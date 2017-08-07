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
A basic sequence decoder that performs a softmax based on the RNN state.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import namedtuple
import tensorflow as tf
from seq2seq.decoders.rnn_decoder import RNNDecoder
from tensorflow.python.layers import core as layers_core
from seq2seq.contrib.seq2seq import helper as tf_decode_helper
from seq2seq.contrib.seq2seq.helper import CustomHelper
from seq2seq.contrib.seq2seq.attention_wrapper import AttentionWrapper
from seq2seq.contrib.seq2seq.attention_wrapper import AttentionWrapperState
from seq2seq.contrib.seq2seq.decoder import Decoder, dynamic_decode

class NewAttentionDecoderOutput(
    namedtuple("DecoderOutput", [
        "logits", "predicted_ids", "cell_output", "attention_scores",
        "attention_context"
    ])):

  """Augmented decoder output that also includes the attention scores.
  """
  pass


class NewAttentionDecoder(RNNDecoder):
  """An RNN Decoder that uses attention over an input sequence.

  Args:
    cell: An instance of ` tf.contrib.rnn.RNNCell`
    helper: An instance of `tf.contrib.seq2seq.Helper` to assist decoding
    initial_state: A tensor or tuple of tensors used as the initial cell
      state.
    vocab_size: Output vocabulary size, i.e. number of units
      in the softmax layer
    attention_keys: The sequence used to calculate attention scores.
      A tensor of shape `[B, T, ...]`.
    attention_values: The sequence to attend over.
      A tensor of shape `[B, T, input_dim]`.
    attention_values_length: Sequence length of the attention values.
      An int32 Tensor of shape `[B]`.
    attention_fn: The attention function to use. This function map from
      `(state, inputs)` to `(attention_scores, attention_context)`.
      For an example, see `seq2seq.decoder.attention.AttentionLayer`.
    reverse_scores: Optional, an array of sequence length. If set,
      reverse the attention scores in the output. This is used for when
      a reversed source sequence is fed as an input but you want to
      return the scores in non-reversed order.
  """

  def __init__(self,
               params,
               mode,
               attention_units,
               vocab_size,
               attention_mechanism,
               output_layer,
               reverse_scores_lengths=None,
               name="attention_decoder"):
    self.vocab_size = vocab_size
    self.attention_mechanism = attention_mechanism
    super(NewAttentionDecoder, self).__init__(params, mode, name)
    self.reverse_scores_lengths = reverse_scores_lengths
    self.cell = AttentionWrapper(self.cell, self.attention_mechanism, attention_layer_size=attention_units)
    self._output_layer = output_layer

  @property
  def output_size(self):
    return NewAttentionDecoderOutput(
        logits=self.vocab_size,
        predicted_ids=tf.TensorShape([]),
        cell_output=self.cell.output_size,
        attention_scores=tf.shape(self.attention_mechanism.values)[1:-1],
        attention_context=self.attention_mechanism.values.get_shape()[-1])

  @property
  def output_dtype(self):
    return NewAttentionDecoderOutput(
        logits=tf.float32,
        predicted_ids=tf.int32,
        cell_output=tf.float32,
        attention_scores=tf.float32,
        attention_context=tf.float32)

  def initialize(self, name=None):

    # finished, first_inputs = self.helper.initialize()
    #
    # # Concat empty attention context
    # attention_context = tf.zeros([
    #     tf.shape(first_inputs)[0],
    #     self.attention_values.get_shape().as_list()[-1]
    # ])
    # first_inputs = tf.concat([first_inputs, attention_context], 1)
    #
    # return finished, first_inputs, self.initial_state

    """Initialize the decoder.

    Args:
      name: Name scope for any created operations.

    Returns:
      `(finished, first_inputs, initial_state)`.
    """
    return self.helper.initialize() + (self.initial_state,)

  def _setup(self, initial_state, helper):
    self.helper = helper
    if not isinstance(initial_state, AttentionWrapperState):
      initial_state = self.cell.zero_state(self.helper.batch_size, tf.float32).clone(cell_state=initial_state)
    self.initial_state = initial_state


  def step(self, time_, inputs, state, name=None):

    cell_output, cell_state = self.cell(inputs, state) #call self.cell.call(inputs, state)
    #class AttentionWrapperState(
    # collections.namedtuple("AttentionWrapperState",
    #                        ("cell_state", "attention", "time", "alignments",
    #                         "alignment_history")))

    if self._output_layer is not None:
      logits = self._output_layer(cell_output)

    attention_scores = cell_state.alignments
    attention_context = cell_state.attention

    if self.reverse_scores_lengths is not None:
      attention_scores = tf.reverse_sequence(
          input=attention_scores,
          seq_lengths=self.reverse_scores_lengths,
          seq_dim=1,
          batch_dim=0)

    sample_ids = self.helper.sample(
        time=time_, outputs=logits, state=cell_state)

    outputs = NewAttentionDecoderOutput(
        logits=logits,
        predicted_ids=sample_ids,
        cell_output=cell_state.cell_state,
        attention_scores=attention_scores,
        attention_context=attention_context)

    finished, next_inputs, next_state = self.helper.next_inputs(
        time=time_, outputs=outputs, state=cell_state, sample_ids=sample_ids)

    return (outputs, next_state, next_inputs, finished)

  def _build(self, initial_state, helper):
    if not self.initial_state:
      self._setup(initial_state, helper)

    scope = tf.get_variable_scope()
    scope.set_initializer(tf.random_uniform_initializer(
        -self.params["init_scale"],
        self.params["init_scale"]))

    maximum_iterations = None
    if self.mode == tf.contrib.learn.ModeKeys.INFER:
      maximum_iterations = self.params["max_decode_length"]

    outputs, final_state, final_sequence_lengths = dynamic_decode(
        decoder=self,
        output_time_major=True,
        impute_finished=False,
        maximum_iterations=maximum_iterations)

    return (outputs, final_state)