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

from pydoc import locate

import os
import sys
import shutil
import tensorflow as tf
from seq2seq import decoders
from seq2seq.data import vocab
from seq2seq.models import AttentionSeq2Seq
from seq2seq import graph_utils
from seq2seq.features.aliments import get_aliments
from seq2seq.graph_utils import templatemethod
from pydoc import locate
from seq2seq import losses as seq2seq_losses
from seq2seq.models import bridges
from seq2seq.contrib.seq2seq import helper as tf_decode_helper

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

    # new_source_ids, source_oov_words_list = self.get_re_id_source(
    #                                                               features["source_tokens"], features["source_ids"],
    #                                                               source_unk_id, source_origin_vocab_size
    #                                                               )
    #
    # ##maintain the max article oov words for the vocab union for decoder's copy and gen
    # source_oov_words_num = [tf.shape(v)[0] for v in source_oov_words_list]
    # self.source_max_oov_words = tf.reduce_max(source_oov_words_num, axis=0)

    # features["extend_source_ids"] = new_source_ids
    # features["source_max_oov_words"] = self.source_max_oov_words
    # features["source_oov_words_num"] = source_oov_words_num #(batch, article oov word nums)


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

    #cal aliments and mask for copy-gen model, [b, max_target_len, max_source_len]
    features["aliments"], features["masks"] = get_aliments(features, labels)

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



  def _create_decoder(self, encoder_output, features, _labels):
    attention_class = locate(self.params["attention.class"]) or \
      getattr(decoders.attention, self.params["attention.class"])
    attention_layer = attention_class(
        params=self.params["attention.params"], mode=self.mode)

    # If the input sequence is reversed we also need to reverse
    # the attention scores.
    reverse_scores_lengths = None
    if self.params["source.reverse"]:
      reverse_scores_lengths = features["source_len"]
      if self.use_beam_search:
        reverse_scores_lengths = tf.tile(
            input=reverse_scores_lengths,
            multiples=[self.params["inference.beam_search.beam_width"]])

    return self.decoder_class(
        params=self.params["decoder.params"],
        mode=self.mode,
        vocab_size=self.target_vocab_info.total_size,
        attention_values=encoder_output.attention_values,
        attention_values_length=encoder_output.attention_values_length,
        attention_keys=encoder_output.outputs,
        attention_fn=attention_layer,
        reverse_scores_lengths=reverse_scores_lengths)

  @templatemethod("encode")
  def encode(self, features, labels):
    source_embedded = tf.nn.embedding_lookup(self.source_embedding,
                                             features["source_ids"])
    encoder_fn = self.encoder_class(self.params["encoder.params"], self.mode)
    return encoder_fn(source_embedded, features["source_len"])

  @templatemethod("decode")
  def decode(self, encoder_output, features, labels):
    decoder = self._create_decoder(encoder_output, features, labels)
    if self.use_beam_search:
      decoder = self._get_beam_search_decoder(decoder)

    bridge = self._create_bridge(
      encoder_outputs=encoder_output,
      decoder_state_size=decoder.cell.state_size)
    if self.mode == tf.contrib.learn.ModeKeys.INFER:
      return self._decode_infer(decoder, bridge, encoder_output, features,
                                labels)
    else:
      return self._decode_train(decoder, bridge, encoder_output, features,
                                labels)

  def _create_bridge(self, encoder_outputs, decoder_state_size):
    """Creates the bridge to be used between encoder and decoder"""
    bridge_class = locate(self.params["bridge.class"]) or \
                   getattr(bridges, self.params["bridge.class"])
    return bridge_class(
      encoder_outputs=encoder_outputs,
      decoder_state_size=decoder_state_size,
      params=self.params["bridge.params"],
      mode=self.mode)

  def _decode_train(self, decoder, bridge, _encoder_output, _features, labels):
    """Runs decoding in training mode"""
    target_embedded = tf.nn.embedding_lookup(self.target_embedding,
                                             labels["target_ids"])
    helper_train = tf_decode_helper.TrainingHelper(
      inputs=target_embedded[:, :-1],
      sequence_length=labels["target_len"] - 1)
    decoder_initial_state = bridge()
    return decoder(decoder_initial_state, helper_train)

  def _decode_infer(self, decoder, bridge, _encoder_output, features, labels):
    """Runs decoding in inference mode"""
    batch_size = self.batch_size(features, labels)
    if self.use_beam_search:
      batch_size = self.params["inference.beam_search.beam_width"]

    target_start_id = self.target_vocab_info.special_vocab.SEQUENCE_START
    helper_infer = tf_decode_helper.GreedyEmbeddingHelper(
      embedding=self.target_embedding,
      start_tokens=tf.fill([batch_size], target_start_id),
      end_token=self.target_vocab_info.special_vocab.SEQUENCE_END)
    decoder_initial_state = bridge()
    return decoder(decoder_initial_state, helper_infer)


  def compute_loss(self, decoder_output, _features, labels):
    """Computes the loss for this model.

    Returns a tuple `(losses, loss)`, where `losses` are the per-batch
    losses and loss is a single scalar tensor to minimize.
    """
    #pylint: disable=R0201
    # Calculate loss per example-timestep of shape [B, T]
    losses = seq2seq_losses.cross_entropy_sequence_loss(
        logits=decoder_output.logits[:, :, :],
        targets=tf.transpose(labels["target_ids"][:, 1:], [1, 0]),
        sequence_length=labels["target_len"] - 1)

    # Calculate the average log perplexity
    loss = tf.reduce_sum(losses) / tf.to_float(
        tf.reduce_sum(labels["target_len"] - 1))

    return losses, loss

  def _build(self, features, labels, params):
    # Pre-process features and labels
    features, labels = self._preprocess(features, labels)

    encoder_output = self.encode(features, labels)
    decoder_output, _, = self.decode(encoder_output, features, labels)

    if self.mode == tf.contrib.learn.ModeKeys.INFER:
      predictions = self._create_predictions(
          decoder_output=decoder_output, features=features, labels=labels)
      loss = None
      train_op = None
    else:
      losses, loss = self.compute_loss(decoder_output, features, labels)

      train_op = None
      if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
        train_op = self._build_train_op(loss)

      predictions = self._create_predictions(
          decoder_output=decoder_output,
          features=features,
          labels=labels,
          losses=losses)

    # We add "useful" tensors to the graph collection so that we
    # can easly find them in our hooks/monitors.
    graph_utils.add_dict_to_collection(predictions, "predictions")

    #here return 3 elements is ok, in estimator, it will be atomatically into model_fn_lib.ModelFnOps
    return predictions, loss, train_op

  def __call__(self, features, labels, params):
    """Creates the model graph. See the model_fn documentation in
    tf.contrib.learn.Estimator class for a more detailed explanation.
    """
    with tf.variable_scope("model"):
      with tf.variable_scope(self.name):
        return self._build(features, labels, params)














