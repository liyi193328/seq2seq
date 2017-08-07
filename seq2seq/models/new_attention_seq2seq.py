# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Sequence to Sequence model with attention
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pydoc import locate

import tensorflow as tf
from seq2seq.graph_utils import templatemethod
from tensorflow.python.layers import core as layers_core
from seq2seq.contrib.seq2seq import helper as tf_decode_helper

from seq2seq import decoders
from seq2seq.models.basic_seq2seq import BasicSeq2Seq


class NewAttentionSeq2Seq(BasicSeq2Seq):
  """Sequence2Sequence model with attention mechanism.

  Args:
    source_vocab_info: An instance of `VocabInfo`
      for the source vocabulary
    target_vocab_info: An instance of `VocabInfo`
      for the target vocabulary
    params: A dictionary of hyperparameters
  """

  def __init__(self, params, mode, name="att_seq2seq"):
    super(NewAttentionSeq2Seq, self).__init__(params, mode, name)

  @staticmethod
  def default_params():
    params = BasicSeq2Seq.default_params().copy()
    params.update({
        "attention.class": "LuongAttention",
        "attention_units":128,
        "attention.params": {}, # Arbitrary attention layer parameters
        "bridge.class": "seq2seq.models.bridges.ZeroBridge",
        "encoder.class": "seq2seq.encoders.BidirectionalRNNEncoder",
        "encoder.params": {},  # Arbitrary parameters for the encoder
        "decoder.class": "seq2seq.decoders.NewAttentionDecoder",
        "decoder.params": {}  # Arbitrary parameters for the decoder
    })
    return params

  def _create_decoder(self, encoder_output, features, _labels):

    attention_mechanism_class = locate(self.params["attention.class"]) or \
      getattr(decoders.attention, self.params["attention.class"])

    # If the input sequence is reversed we also need to reverse
    # the attention scores.
    reverse_scores_lengths = None
    if self.params["source.reverse"]:
      reverse_scores_lengths = features["source_len"]
      if self.use_beam_search:
        reverse_scores_lengths = tf.tile(
            input=reverse_scores_lengths,
            multiples=[self.params["inference.beam_search.beam_width"]])

    attention_units = self.params["attention_units"]
    #encoder.outputs
    if self.use_beam_search:
      attention_values = tf.contrib.seq2seq.tile_batch(encoder_output.outputs, multiplier=self.params["inference.beam_search.beam_width"])
      memory_sequence_length = tf.contrib.seq2seq.tile_batch(encoder_output.attention_values_length, multiplier=self.params["inference.beam_search.beam_width"])
    else:
      attention_values = encoder_output.outputs
      memory_sequence_length = encoder_output.attention_values_length
    attention_mechanism = attention_mechanism_class(attention_units, attention_values,
                                                         memory_sequence_length=memory_sequence_length)
    self.attention_mechanism = attention_mechanism
    return self.decoder_class(
        params=self.params["decoder.params"],
        mode=self.mode,
        attention_units=attention_units,
        vocab_size=self.target_vocab_info.total_size,
        output_layer = layers_core.Dense(self.target_vocab_info.total_size),
        attention_mechanism=attention_mechanism,
        reverse_scores_lengths=reverse_scores_lengths)

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

  @templatemethod("decode")
  def decode(self, encoder_output, features, labels):

    decoder = self._create_decoder(encoder_output, features, labels)
    if self.mode == tf.contrib.learn.ModeKeys.INFER:
      if self.use_beam_search:
        decoder = self._get_beam_search_decoder(decoder)

      bridge = self._create_bridge(
        encoder_outputs=encoder_output,
        decoder_state_size=decoder.cell.state_size)
      return self._decode_infer(decoder, bridge, encoder_output, features,
                                labels)
    else:
      bridge = self._create_bridge(
        encoder_outputs=encoder_output,
        decoder_state_size=decoder.cell.state_size.cell_state)
      return self._decode_train(decoder, bridge, encoder_output, features,
                                labels)