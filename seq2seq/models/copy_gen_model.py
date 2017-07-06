# coding=utf-8

"""
use copy model to seq2seq
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

__author__ = "liyi"
__date__ = "2017-07-04"


import os
import sys
import shutil
import tensorflow as tf

from seq2seq.data import vocab
from seq2seq.models import AttentionSeq2Seq
from seq2seq import graph_utils

def should_continue(t, timestaps, *args):
  return t < timestaps

def source_seq_iteration(t, max_t, seq_source_ids, cur_source_ids, cur_oov_list, source_seq_words,
                        source_unk_id, oringin_vocab_size):
  word_id = tf.gather(seq_source_ids, t)
  if tf.not_equal(word_id, source_unk_id):
    cur_source_ids.write(t, word_id)
  else:
    word = tf.gather(source_seq_words, t)
    word_equal_index = tf.where(tf.equal(cur_oov_list, word))
    tf.assert_less_equal(len(word_equal_index), 1)
    cur_oov_num = cur_source_ids.size()
    if len(word_equal_index) == 0:
      cur_oov_list.write(cur_oov_num, word)
      cur_source_ids.write(t,
                           cur_oov_num + oringin_vocab_size)  # article's oov id is range( max_vocab_size, max_vocab_size + len(cur_oov_list) )
    else:
      cur_source_ids.write(t, word_equal_index[0][0] + oringin_vocab_size)

  return t + 1, max_t, cur_source_ids, cur_oov_list

class CopyGenSeq2Seq(AttentionSeq2Seq):

  def __init__(self,  params, mode, pointer_gen = True, coverage = False, name="copy_gen_seq2seq"):

    self._pointer_gen = pointer_gen
    self._coverage = coverage
    super(CopyGenSeq2Seq, self).__init__(params, mode, name) #final self._params will be the params override the default_params

  @staticmethod
  def default_params():
    """call in configurable class, return default params
    """
    params = AttentionSeq2Seq.default_params().copy()
    params.update({
        "pointer_gen": True,
        "coverage": False,
        "embedding.share": True,
        "attention.class": "AttentionLayerBahdanau",
        "attention.params": {}, # Arbitrary attention layer parameters
        "bridge.class": "seq2seq.models.bridges.ZeroBridge",
        "encoder.class": "seq2seq.encoders.BidirectionalRNNEncoder",
        "encoder.params": {},  # Arbitrary parameters for the encoder
        "decoder.class": "seq2seq.decoders.AttentionDecoder",
        "decoder.params": {}  # Arbitrary parameters for the decoder
    })
    return params

  def load_vocab(self, source_vocab_path, target_vocab_path):

    # Create vocabulary lookup for source
    source_vocab_to_id, source_id_to_vocab, source_word_to_count, source_origin_vocab_size = \
      vocab.create_vocabulary_lookup_table(self.source_vocab_info.path)

    # Create vocabulary look for target
    target_vocab_to_id, target_id_to_vocab, target_word_to_count, target_origin_vocab_size = \
      vocab.create_vocabulary_lookup_table(self.target_vocab_info.path)

    # Add vocab tables to graph colection so that we can access them in
    # other places.
    graph_utils.add_dict_to_collection({
        "source_vocab_to_id": source_vocab_to_id,
        "source_id_to_vocab": source_id_to_vocab,
        "source_word_to_count": source_word_to_count,
        "target_vocab_to_id": target_vocab_to_id,
        "target_id_to_vocab": target_id_to_vocab,
        "target_word_to_count": target_word_to_count
    }, "vocab_tables")

    self._source_vocab_to_id = source_vocab_to_id
    self._source_id_to_vocab = source_id_to_vocab
    self._target_vocab_to_id = target_vocab_to_id
    self._target_id_to_vocab = target_id_to_vocab
    self._source_origin_vocab_size = source_origin_vocab_size
    self._target_origin_vocab_size = target_origin_vocab_size

  def _preprocess(self, features, labels):
    """Model-specific preprocessing for features and labels:

    - Creates vocabulary lookup tables for source and target vocab
    - Converts tokens into vocabulary ids
    """
    # Create vocabulary lookup for source
    source_vocab_to_id, source_id_to_vocab, source_word_to_count, source_origin_vocab_size = \
      vocab.create_vocabulary_lookup_table(self.source_vocab_info.path)

    # Create vocabulary look for target
    target_vocab_to_id, target_id_to_vocab, target_word_to_count, target_origin_vocab_size = \
      vocab.create_vocabulary_lookup_table(self.target_vocab_info.path)

    # Add vocab tables to graph colection so that we can access them in
    # other places.
    graph_utils.add_dict_to_collection({
        "source_vocab_to_id": source_vocab_to_id,
        "source_id_to_vocab": source_id_to_vocab,
        "source_word_to_count": source_word_to_count,
        "target_vocab_to_id": target_vocab_to_id,
        "target_id_to_vocab": target_id_to_vocab,
        "target_word_to_count": target_word_to_count
    }, "vocab_tables")

    # Slice source to max_len
    if self.params["source.max_seq_len"] is not None:
      features["source_tokens"] = features["source_tokens"][:, :self.params[
          "source.max_seq_len"]]
      features["source_len"] = tf.minimum(features["source_len"],
                                          self.params["source.max_seq_len"])

    # Look up the source ids in the vocabulary
    features["source_ids"] = source_vocab_to_id.lookup(features["source_tokens"])
    features["source_ids"] = tf.Print(features["source_ids"], [ features["source_ids"] ], message="source_ids", summarize=10)

    source_unk_id = self.source_vocab_info.special_vocab.UNK
    target_unk_id = self.target_vocab_info.special_vocab.UNK

    new_source_ids, source_oov_words_list = self.get_re_id_source(
                                                                  features["source_tokens"], features["source_ids"],
                                                                  source_unk_id, source_origin_vocab_size
                                                                  )

    ##maintain the max article oov words for the vocab union for decoder's copy and gen
    source_oov_words_num = [tf.shape(v)[0] for v in source_oov_words_list]
    self.source_max_oov_words = tf.reduce_max(source_oov_words_num, axis=0)

    features["extend_source_ids"] = new_source_ids
    features["source_max_oov_words"] = self.source_max_oov_words
    features["source_oov_words_num"] = source_oov_words_num #(batch, article oov word nums)


    # Maybe reverse the source
    if self.params["source.reverse"] is True:
      features["source_ids"] = tf.reverse_sequence(
          input=features["source_ids"],
          seq_lengths=features["source_len"],
          seq_dim=1,
          batch_dim=0,
          name=None)

    features["source_len"] = tf.to_int32(features["source_len"])
    tf.summary.histogram("source_len", tf.to_float(features["source_len"]))
    # tf.summary.histogram("source_oov_len", tf.to_float(features["source_oov_words_num"]))
    # tf.summary.scalar("sample_max_oov_words", features["max_oov_words"])

    if labels is None:
      return features, None

    labels = labels.copy()

    # Slices targets to max length
    if self.params["target.max_seq_len"] is not None:
      labels["target_tokens"] = labels["target_tokens"][:, :self.params[
          "target.max_seq_len"]]
      labels["target_len"] = tf.minimum(labels["target_len"],
                                        self.params["target.max_seq_len"])

    # Look up the target ids in the vocabulary
    labels["target_ids"] = target_vocab_to_id.lookup(labels["target_tokens"])
    labels["target_len"] = tf.to_int32(labels["target_len"])

    # labels["extend_target_ids"], labels["extend_oov_word_num"] = self.get_re_id_target(source_oov_words_list, labels["target_tokens"], labels["target_ids"],
    #                                                     target_unk_id,  target_origin_vocab_size)

    tf.summary.histogram("target_len", tf.to_float(labels["target_len"]))
    # tf.summary.histogram("extend_oov_word_num", tf.to_float(labels["extend_oov_word_num"])) #(batch_size, oov word num in extend vocab with source)

    # Keep track of the number of processed tokens
    num_tokens = tf.reduce_sum(labels["target_len"])
    num_tokens += tf.reduce_sum(features["source_len"])
    token_counter_var = tf.Variable(0, "tokens_counter")
    total_tokens = tf.assign_add(token_counter_var, num_tokens)
    tf.summary.scalar("num_tokens", total_tokens)

    with tf.control_dependencies([total_tokens]):
      features["source_tokens"] = tf.identity(features["source_tokens"])

    # Add to graph collection for later use
    graph_utils.add_dict_to_collection(features, "features")
    if labels:
      graph_utils.add_dict_to_collection(labels, "labels")

    return features, labels

  def get_re_id_source(self, source_words, source_ids, source_unk_id, oringin_vocab_size):

    source_ids = tf.Print(source_ids, [tf.shape(source_ids)], message="source_ids_shape:")
    unstack_source_ids = tf.unstack(source_ids)
    new_source_ids = []
    source_oov_words_list = []


    for i, seq_source_ids in enumerate(unstack_source_ids):#loop batch
      counter = 0
      max_t = tf.shape(seq_source_ids)[0]
      source_seq_words = source_words[i,:]
      initial_new_source_ids = tf.TensorArray(dtype=tf.int32, size=max_t)
      initial_cur_oov_words = tf.TensorArray(dtype=tf.string, dynamic_size=True)
      initial_cur_oov_ids = tf.TensorArray(dtype=tf.int32, size=max_t)
      initial_t = tf.Variable(0, dtype=tf.int32)
      t, max_time_stamps, new_seq_source_ids, cur_oov_words = \
        tf.while_loop( should_continue,  source_seq_iteration, [initial_t, max_t, seq_source_ids,
                                                               initial_new_source_ids, initial_cur_oov_words,
                                                               source_seq_words, source_unk_id, oringin_vocab_size])
      new_seq_source_ids = new_seq_source_ids.pack()
      cur_oov_words = cur_oov_words.pack()
      new_source_ids.append(new_seq_source_ids)
      source_oov_words_list.append(cur_oov_words)

    new_source_ids_tensor = tf.convert_to_tensor(new_source_ids, dtype=tf.int32)

    return new_source_ids_tensor, source_oov_words_list

  def get_re_id_target(self, source_oov_words_list, target_words, target_ids, target_unk_id, oringin_vocab_size):

    unstack_target_ids = tf.unstack(target_ids)
    tf.assert_equal(len(source_oov_words_list), len(unstack_target_ids))

    new_target_ids = []
    target_unk_token_nums = []
    for i, one_seq_target_ids in enumerate(unstack_target_ids): #loop batch
      new_seq_target_ids = []
      unk_token_nums = 0
      target_ids_list = tf.unstack(one_seq_target_ids)
      for j, word_id in enumerate(target_ids_list):
        if tf.equal(word_id, target_unk_id):
          word = target_words[i,j]
          for k, source_oov_word in enumerate(source_oov_words_list[i]):
            if tf.equal(word, source_oov_word):
              # abstract 's word is out of origin vocab, but in article, use the index of article oov word + origin_vocab_size as new id
              article_oov_word_id = oringin_vocab_size + k
              new_seq_target_ids.append(article_oov_word_id)
            else:
              unk_token_nums += 1
              new_seq_target_ids.append(target_unk_id)
        else:
          unk_token_nums += 1
          new_seq_target_ids.append(target_unk_id)

        target_unk_token_nums.append(unk_token_nums)
        new_seq_target_ids = tf.convert_to_tensor(new_seq_target_ids)
        new_target_ids.append(new_seq_target_ids)

    return tf.convert_to_tensor(new_target_ids), target_unk_token_nums






