#encoding=utf-8

import tensorflow as tf

def make_induces(x, y):
  """return [ [(0,0), (0,1),...,(0,y-1)], [(1,0),...], [(x-1, 0), (x-1, 1), ..., (x-1, y-1)]
  """
  index_x = tf.expand_dims(tf.range(0, x), 1)
  index_y = tf.expand_dims(tf.range(0, y), 0)
  index_x = tf.tile(index_x, [1, y])
  index_y = tf.tile(index_y, [x, 1])
  induces = tf.stack([index_x, index_y], axis=2)
  return induces

def get_aliments(self, features, labels):
  source_tokens = features["source_tokens"]
  source_len = features["source_len"]
  target_tokens = labels["target_tokens"]
  target_len = labels["target_len"]
  max_source_len = tf.reduce_max(source_len)
  max_target_len = tf.reduce_max(target_len)
  batch_size = source_tokens.get_shape().as_list()[0] or tf.shape(source_tokens)[0]

  source_token_list = tf.unstack(source_tokens)
  source_len_list = tf.unstack(source_len)
  target_token_list = tf.unstack(target_tokens)
  target_len_list = tf.unstack(target_len)

  max_mask_shape = tf.convert_to_tensor([max_target_len, max_source_len])

  batch_mask = []
  batch_aliments = []
  for source_token, slen, target_token, tlen in zip(source_token_list, source_len_list, target_token_list,
                                                    target_len_list):
    x = tf.expand_dims(source_token, 0)
    y = tf.expand_dims(target_token, 1)
    rx = tf.tile(x, [tlen, 1])
    ry = tf.tile(y, [1, slen])
    aliments = tf.cast(tf.equal(rx, ry),
                       tf.int32)  # [tlen, slen], every row is one target token match the source_tokens
    induces = make_induces(tlen, slen)
    aliments = tf.scatter_nd(induces, aliments, (max_target_len, max_source_len))
    mask = tf.scatter_nd(induces, tf.ones([tlen, slen]),
                         (max_target_len, max_source_len))  # 1 when in [0, tlen) x [0, slen), res is 0
    batch_mask.append(mask)
    batch_aliments.append(aliments)

  batch_mask = tf.convert_to_tensor(batch_mask)
  batch_aliments = tf.convert_to_tensor(batch_aliments)

  return batch_aliments, batch_mask
