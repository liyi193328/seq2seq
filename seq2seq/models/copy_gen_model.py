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
from seq2seq.features.aliments import get_tensor_aliments
from seq2seq.graph_utils import templatemethod
from pydoc import locate
from seq2seq import losses as seq2seq_losses
from seq2seq.models import bridges
from seq2seq.contrib.seq2seq import helper as tf_decode_helper

class CopyGenSeq2Seq(AttentionSeq2Seq):

  def __init__(self,  params, mode, vocab_instance, pointer_gen = True, coverage = True, name="copy_gen_seq2seq"):

    self._pointer_gen = pointer_gen
    self._coverage = coverage
    self._vocab_instance = vocab_instance
    super(CopyGenSeq2Seq, self).__init__(params, mode, name) #final self._params will be the params override the default_params

  @staticmethod
  def default_params():
    """call in configurable class, return default params
    """
    params = AttentionSeq2Seq.default_params().copy()
    params.update({
        "pointer_gen": True,
        "coverage": True,
        "embedding.share": True,
        "attention.class": "AttentionLayerBahdanau",
        "attention.params": {}, # Arbitrary attention layer parameters
        "bridge.class": "seq2seq.models.bridges.ZeroBridge",
        "encoder.class": "seq2seq.encoders.BidirectionalRNNEncoder",
        "encoder.params": {},  # Arbitrary parameters for the encoder
        "decoder.class": "seq2seq.decoders.CopyGenDecoder",
        "decoder.params": {}  # Arbitrary parameters for the decoder
    })
    return params

  def create_lookup_table(self):

    # Create vocabulary lookup for source
    source_vocab_to_id, source_id_to_vocab, source_word_to_count, source_origin_vocab_size = \
      vocab.create_tensor_vocab(self._vocab_instance)

    # Create vocabulary look for target
    target_vocab_to_id, target_id_to_vocab, target_word_to_count, target_origin_vocab_size = \
      source_vocab_to_id, source_id_to_vocab, source_word_to_count, source_origin_vocab_size

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
    self.create_lookup_table()

    # int64_keys = ["source_len", "source_ids", "extend_source_ids", "source_oov_nums", "target_ids", "extend_target_ids"]

    # Slice source to max_len
    ###here can't
    if self.params["source.max_seq_len"] is not None:
      features["source_tokens"] = features["source_tokens"][:, :self.params[
          "source.max_seq_len"]]
      features["source_len"] = tf.minimum(features["source_len"],
                                          self.params["source.max_seq_len"])

    # Look up the source ids in the vocabulary

    graph_source_ids = self._source_vocab_to_id.lookup(features["source_tokens"]) #every sequence contains sequence end flag
    tf.assert_equal(graph_source_ids, features["source_ids"])

    features["source_oov_nums"] = tf.cast(features["source_oov_nums"], tf.int32)
    features["source_max_oov_num"] = tf.reduce_max(features["source_oov_nums"])

    # Maybe reverse the source
    if self.params["source.reverse"] is True:
      raise NotImplemented("reverse func is not stable now")
      reverse_keys = ["source_ids", "extend_source_ids", "source_pos_ids", "source_tfidfs", "source_ner_ids"]
      for key in reverse_keys:
        features[key] = tf.reverse_sequence(
            input=features[key],
            seq_lengths=features["source_len"],
            seq_dim=1,
            batch_dim=0,
            name=None)

    features["source_len"] = tf.cast(features["source_len"], tf.int32)

    tf.summary.histogram("source_len", features["source_len"])
    tf.summary.histogram("source_oov_nums", features["source_oov_nums"])
    tf.summary.scalar("batch_max_oov_words", features["source_max_oov_num"])

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
    graph_target_ids = self._target_vocab_to_id.lookup(labels["target_tokens"])
    tf.assert_equal(graph_target_ids, labels["target_ids"])

    labels["target_len"] = tf.to_int32(labels["target_len"])
    tf.summary.histogram("target_len", tf.to_float(labels["target_len"]))

    # Keep track of the number of processed tokens
    num_tokens = tf.reduce_sum(labels["target_len"])
    num_tokens += tf.reduce_sum(features["source_len"])
    token_counter_var = tf.Variable(0, "tokens_counter", dtype=tf.int32)
    total_tokens = tf.assign_add(token_counter_var, num_tokens)
    tf.summary.scalar("num_tokens", total_tokens)

    with tf.control_dependencies([total_tokens]):
      features["source_tokens"] = tf.identity(features["source_tokens"])

    # Add to graph collection for later use
    graph_utils.add_dict_to_collection(features, "features")
    if labels:
      graph_utils.add_dict_to_collection(labels, "labels")

    return features, labels

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
    source_word_embedded = tf.nn.embedding_lookup(self.source_embedding,
                                             features["source_ids"])

    encoder_fn = self.encoder_class(self.params["encoder.params"], self.mode)
    return encoder_fn(source_word_embedded, features["source_len"])

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

  def _calc_final_dist(self, decoder_output, features):
    """Calculate the final distribution, for the pointer-generator model

    Args:
      vocab_dists: The vocabulary distributions. List length max_dec_steps of (batch_size, vsize) arrays. The words are in the order they appear in the vocabulary file.
      attn_dists: The attention distributions. List length max_dec_steps of (batch_size, attn_len) arrays

    Returns:
      final_dists: The final distributions. List length max-dec_steps of (batch_size, extended_vsize) arrays.
    """

    vocab_dists = decoder_output.logits
    attn_dists = decoder_output.attention_scores
    p_gens = decoder_output.pgens

    vocab_total_size = self.source_vocab_info.total_size

    max_t = tf.shape(vocab_dists)[0]
    batch_size = vocab_dists.get_shape().as_list()[1] or tf.shape(vocab_dists)[1]
    batch_source_max_oovs = tf.reduce_max(features["source_oov_nums"])
    extended_vsize = vocab_total_size + batch_source_max_oovs
    tf.assert_equal(tf.shape(vocab_dists)[0], tf.shape(attn_dists)[0])
    tf.assert_equal(tf.shape(vocab_dists)[0], tf.shape(p_gens)[0])

    final_dists = tf.TensorArray(tf.float32, size=max_t, dynamic_size=True)

    with tf.variable_scope('final_distribution'):

      def should_continue(now_t, max_t, *args, **kwargs):
        return tf.less(now_t, max_t)

      def body(t, max_t, final_dists, *args, **kwargs):
        # p_gen = tf.gather(p_gens, t)
        # vocab_dist = tf.gather(vocab_dists, t)
        # attn_dist = tf.gather(attn_dists, t)
        p_gen = p_gens[t,:]
        vocab_dist = vocab_dists[t, :, :]
        attn_dist = attn_dists[t, :, :]
        vocab_dist, attn_dist = p_gen * vocab_dist , (1-p_gen) * attn_dist
        zeros_shape = tf.convert_to_tensor((batch_size, batch_source_max_oovs))
        extra_zeros = tf.zeros(zeros_shape)
        vocab_dist_extended = tf.concat(axis=1, values=[vocab_dist, extra_zeros]) #[b, origin_total_vocab_size + extend_vocab_size]
        batch_nums = tf.range(0, limit=batch_size, dtype=tf.int32)
        batch_nums = tf.expand_dims(batch_nums, 1)
        attn_len = tf.shape(features["source_ids"])[1]
        batch_nums =  tf.tile(batch_nums, [1, attn_len])
        indices = tf.stack( ( batch_nums, tf.to_int32(features["extend_source_ids"]) ), axis=2)
        final_shapes = [batch_size, extended_vsize]
        #place attend score to corresponding id(extend): total_vocab is 10, extend_source_ids=[1,11, 12, 11, 3], attn_scores=[0.1, 0.2, 0.4, 0.25, 0.05]
        #place the 1's score 0.1 to 1th index value, index 11th value is 0.2,..., the same word id 's properbility is summed
        attn_dist_projected = tf.scatter_nd(indices, attn_dist, final_shapes)
        final_dist = attn_dist_projected + vocab_dist_extended #add copy(attn) score to
        #avoid nan
        final_dist += tf.float32.min
        final_dists = final_dists.write(t, final_dist)
        return t+1, max_t, final_dists

      now_t = tf.constant(0)
      _, _, final_dists = tf.while_loop(
        should_continue,
        body,
        loop_vars=[now_t, max_t, final_dists]
      )
      return final_dists

  def compute_loss(self, decoder_output, _features, labels):
    """Computes the loss for this model.

    Returns a tuple `(losses, loss)`, where `losses` are the per-batch
    losses and loss is a single scalar tensor to minimize.
    """
    #pylint: disable=R0201
    # Calculate loss per example-timestep of shape [B, T]


    final_dists = self._calc_final_dist(decoder_output, _features)
    final_dists = final_dists.stack() #

    # losses = seq2seq_losses.cross_entropy_sequence_loss(
    #     logits=decoder_output.logits[:, :, :],
    #     targets=tf.transpose(labels["target_ids"][:, 1:], [1, 0]),
    #     sequence_length=labels["target_len"] - 1)

    losses = seq2seq_losses.cross_entropy_sequence_loss(
        logits=final_dists,
        targets=tf.transpose(labels["extend_target_ids"][:, 1:], [1, 0]),
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














