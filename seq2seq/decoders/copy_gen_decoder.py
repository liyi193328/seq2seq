#encoding=utf-8

__author__ = "turingli"
__date__ = "2017-07-19"

import tensorflow as tf

from collections import namedtuple
from seq2seq.decoders import RNNDecoder
from seq2seq.contrib.seq2seq.helper import CustomHelper

from seq2seq.models.seq2seq_model import Seq2SeqModel


class CopyGenDecoderOutput(
    namedtuple("DecoderOutput", [
        "logits", "predicted_ids", "cell_output", "attention_scores",
        "attention_context", "pgen"
    ])):
  pass

class CopyGenDecoder(RNNDecoder):

  def __init__(self,
               params,
               mode,
               vocab_size,
               attention_keys,
               attention_values,
               attention_values_length,
               attention_fn,
               source_embedding=None,
               reverse_scores_lengths=None,
               name="CopyGenDecoder"):
    super(CopyGenDecoder, RNNDecoder).__init__(params, mode, name)
    self.vocab_size = vocab_size
    self.source_embedding = source_embedding
    self.attention_keys = attention_keys
    self.attention_values = attention_values
    self.attention_values_length = attention_values_length
    self.attention_fn = attention_fn
    self.reverse_scores_lengths = reverse_scores_lengths

  @property
  def output_size(self):
    return CopyGenDecoderOutput(
        logits=self.vocab_size,
        predicted_ids=tf.TensorShape([]),
        cell_output=self.cell.output_size,
        attention_scores=tf.shape(self.attention_values)[1:-1],
        attention_context=self.attention_values.get_shape()[-1],
        pgen= tf.shape(self.pgen)[-1]
    )

  @property
  def output_dtype(self):
    return CopyGenDecoderOutput(
        logits=tf.float32,
        predicted_ids=tf.int32,
        cell_output=tf.float32,
        attention_scores=tf.float32,
        attention_context=tf.float32,
        pgen=tf.float32
    )

  def initialize(self, name=None):

    finished, first_inputs = self.helper.initialize()

    # Concat empty attention context
    attention_context = tf.zeros([
        tf.shape(first_inputs)[0],
        self.attention_values.get_shape().as_list()[-1]
    ])
    first_inputs = tf.concat([first_inputs, attention_context], 1)

    return finished, first_inputs, self.initial_state

  def cal_gen_probability(self, out_features):
    #probability of generation
    pgen = tf.contrib.layers.fully_connected(
      inputs=out_features,
      num_outputs=1,
      activation_fn=tf.nn.tanh,
    )
    return pgen

  def compute_output(self, cell_output):
    """Computes the decoder outputs."""

    # Compute attention
    att_scores, attention_context = self.attention_fn(
        query=cell_output,
        keys=self.attention_keys,
        values=self.attention_values,
        values_length=self.attention_values_length)

    # TODO: Make this a parameter: We may or may not want this.
    # Transform attention context.
    # This makes the softmax smaller and allows us to synthesize information
    # between decoder state and attention context
    # see https://arxiv.org/abs/1508.04025v5

    decode_out_features = tf.concat([cell_output, attention_context], 1)
    softmax_input = tf.contrib.layers.fully_connected(
        inputs=decode_out_features,
        num_outputs=self.cell.output_size,
        activation_fn=tf.nn.tanh,
        scope="attention_mix")

    # Softmax computation
    logits = tf.contrib.layers.fully_connected(
        inputs=softmax_input,
        num_outputs=self.vocab_size,
        activation_fn=None,
        scope="logits")

    #generation probability
    pgen = self.cal_gen_probability(decode_out_features)

    return softmax_input, logits, att_scores, attention_context, pgen

  def _setup(self, initial_state, helper):

    self.initial_state = initial_state

    self.W_u = 0
    wout_dim = 2 * self.params["decoder.params"]["rnn_cell"]["cell_params"]["num_units"]  # dim(context_vector + hidden states)
    # word_dim = self.source_embedding.shape[1].value
    # with tf.variable_scope("copy_gen_project_wout"):
    #   self.wout_proj = tf.get_variable("project_wout", shape=[word_dim, word_dim], dtype=tf.float32, initializer=tf.truncated_normal_initializer)
    #   self.wout = tf.tanh(tf.matmul(self.source_embedding, self.wout_proj)) #[v,d] * [d,d] = [v,d]

    def att_next_inputs(time, outputs, state, sample_ids, name=None):
      """Wraps the original decoder helper function to append the attention
      context.
      """
      finished, next_inputs, next_state = helper.next_inputs(
          time=time,
          outputs=outputs,
          state=state,
          sample_ids=sample_ids,
          name=name)
      next_inputs = tf.concat([next_inputs, outputs.attention_context], 1)
      return (finished, next_inputs, next_state)

    self.helper = CustomHelper(
        initialize_fn=helper.initialize,
        sample_fn=helper.sample,
        next_inputs_fn=att_next_inputs)

  def step(self, time_, inputs, state, name=None):
    cell_output, cell_state = self.cell(inputs, state)
    cell_output_new, logits, attention_scores, attention_context, pgen = \
      self.compute_output(cell_output)

    if self.reverse_scores_lengths is not None:
      attention_scores = tf.reverse_sequence(
          input=attention_scores,
          seq_lengths=self.reverse_scores_lengths,
          seq_dim=1,
          batch_dim=0)

    sample_ids = self.helper.sample(
        time=time_, outputs=logits, state=cell_state)

    outputs = CopyGenDecoderOutput(
        logits=logits,
        predicted_ids=sample_ids,
        cell_output=cell_output_new,
        attention_scores=attention_scores,
        attention_context=attention_context,
        pgen=pgen
    )

    finished, next_inputs, next_state = self.helper.next_inputs(
        time=time_, outputs=outputs, state=cell_state, sample_ids=sample_ids)


