#encoding=utf-8

import tensorflow as tf

def should_continue(t, timestaps, *args):
  return t < timestaps

def get_extend_source_ids(source_words, source_ids, source_unk_id, oringin_vocab_size):
  new_source_ids = []
  source_oov_words_list = []

  unk_num_list = []
  extend_source_ids = []
  for source_id, source_word in zip(tf.unstack(source_ids), tf.unstack(source_words)):
    source_len = tf.shape(source_id)[0]
    unk_bool = tf.equal(source_id, source_unk_id)
    unk_index = tf.where(unk_bool)
    unk_index = tf.cast(unk_index, tf.int32)
    unk_words = tf.gather(source_word, unk_index)
    unk_nums = tf.reduce_sum(tf.cast(unk_bool, tf.int32))
    unk_num_list.append(unk_nums)
    updates = tf.expand_dims(tf.range(oringin_vocab_size, oringin_vocab_size + unk_nums), 1)
    new_shape = tf.convert_to_tensor([source_len, 1])
    new_source_id = tf.scatter_nd(unk_index, updates, new_shape)
    new_source_id = tf.reshape(new_source_id, (-1,))
    new_source_id = tf.where(unk_bool, new_source_id, source_id)
    extend_source_ids.append(new_source_id)
    source_oov_words_list.append(unk_words)

  extend_source_ids = tf.convert_to_tensor(extend_source_ids, dtype=tf.int32)
  return extend_source_ids, source_oov_words_list


def get_extend_target_ids(extend_source_ids, source_words, target_words, target_ids, target_len,
                          target_unk_id, target_to_source_aliment, prefer_local="first"):
  unstack_target_ids = tf.unstack(target_ids)
  extend_target_ids = []
  aliments = target_to_source_aliment
  target_unk_token_nums = []

  def get_target_id(t, max_t, seq_target_word, seq_target_id, extend_target_id, seq_extend_source_id, seq_source_word,
                    target_unk_id):
    cur_target_word = seq_target_word[t]
    cur_target_id = seq_target_id[t]
    if tf.equal(cur_target_id, target_unk_id):
      extend_target_id.write(t, cur_target_id)
    else:
      equal_bool = tf.equal(seq_source_word, cur_target_word)
      if tf.reduce_sum(tf.cast(equal_bool, tf.int32)) == 0:
        extend_target_id.write(t, target_unk_id)
      else:
        equal_index = tf.reduce_min(tf.where(equal_bool))
        new_target_id = seq_extend_source_id[equal_index]
        extend_target_id.write(t, new_target_id)
    return t + 1, extend_target_id

  for i, seq_target_id in enumerate(unstack_target_ids):  # loop batch
    extend_source_id = extend_source_ids[i]
    new_seq_target_ids = []
    extend_target_id = tf.TensorArray(dynamic_size=True, dtype=tf.int32)
    tlen = target_len[i]
    t = 0
    args = (
    t, tlen, target_words[i], seq_target_id, extend_target_id, extend_source_ids[i], source_words[i], target_unk_id)
    extend_target_id = tf.while_loop(should_continue, get_target_id, loop_vars=args)
    unk_token_nums = tf.reduce_sum(tf.cast(tf.equal(extend_target_id, target_unk_id), tf.int32))
    target_unk_token_nums.append(unk_token_nums)
    extend_target_id = tf.convert_to_tensor(extend_target_id)
    extend_target_ids.append(extend_target_id)

  return tf.convert_to_tensor(extend_target_ids), tf.convert_to_tensor(target_unk_token_nums, dtype=tf.int32)

if __name__ == "__main__":
  source_ids = tf.constant([[1, 10, 20, 10], [2, 10, 30, -1]], dtype=tf.int32)
  source_words = tf.constant([["a1", "a10", "a20", "a10"], ["a2", "a10", "a30", "a-1"]], dtype=tf.string)
  target_ids = tf.constand([[1, 10, 12, 20, 2], [-1, 30, 12, -1, -1]])
  target_words = tf.constand([["a1","b10", "b20", "a20", "c10"], ["a-1", "a30", "bq", "cd", "qy"]], dtype=tf.string)
  unk_id = -1
  vocab_size = 200
  extend_source_ids, source_oov_words_list = get_extend_source_ids(source_words,source_ids, unk_id, oringin_vocab_size=vocab_size)
  target_len = target_words.shape(1).value
  extend_target_ids, target_unk_token_nums = get_extend_target_ids(extend_source_ids,source_words,target_words,target_ids, target_len, unk_id)
  with tf.Session() as sess:
    [n_source_ids, n_target_ids, target_oov_nums] = sess.run(extend_source_ids, extend_target_ids, target_unk_token_nums)
    print("new source ids:")
    print(n_source_ids)
    print("\nnew target ids:")
    print(n_target_ids)
    print("\ntarget oov nums:")
    print(target_oov_nums)

