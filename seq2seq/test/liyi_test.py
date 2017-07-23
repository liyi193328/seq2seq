# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import codecs
import yaml
import numpy as np
import tensorflow as tf

from tensorflow import gfile
from seq2seq.data.vocab import create_vocabulary_lookup_table
from seq2seq.data import vocab, input_pipeline
from seq2seq.training import utils as training_utils
from seq2seq.test import utils as test_utils
from seq2seq.models import BasicSeq2Seq, AttentionSeq2Seq, CopyGenSeq2Seq
from seq2seq.data.vocab import Vocab

def t(file_path, default_value=None):
  x = create_vocabulary_lookup_table(file_path)
  print(x)
  return x


from seq2seq.graph_utils import templatemethod

class test_property:

  def __init__(self, x):
    self._x = x

  @property
  @templatemethod("trial")
  def trial(self):
      print("call trial here")
      w = tf.get_variable('w', [])
      return tf.reduce_sum(self._x) * w

def my_trail(x, share_variable_name):
  var1 = tf.get_variable(share_variable_name, shape=[])
  return tf.reduce_sum(x) * var1

template_my = tf.make_template("template_my", my_trail, share_variable_name="my_v")

def test_model(source_path, target_path, vocab_path):

  tf.logging.set_verbosity(tf.logging.INFO)
  batch_size = 2

  # Build model graph
  mode = tf.contrib.learn.ModeKeys.TRAIN
  params_ = AttentionSeq2Seq.default_params().copy()
  params_.update({
      "vocab_source": vocab_path,
      "vocab_target": vocab_path,
    }
  )
  model = AttentionSeq2Seq( params = params_, mode = mode)

  tf.logging.info(vocab_path)

  input_pipeline_ = input_pipeline.ParallelTextInputPipeline(
    params={
      "source_files":[source_path],
      "target_files":[target_path]
    },
    mode=mode
  )
  input_fn = training_utils.create_input_fn(
    pipeline=input_pipeline_, batch_size=batch_size)
  features, labels = input_fn()
  fetches = model(features, labels, None)

  fetches = [_ for _ in fetches if _ is not None]

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    sess.run(tf.tables_initializer())
    with tf.contrib.slim.queues.QueueRunners(sess):
      fetches_ = sess.run(fetches)

  return model, fetches_

def test_copy_gen_model(record_path, vocab_path=None):

  tf.logging.set_verbosity(tf.logging.INFO)

  vocab = Vocab(vocab_path)
  batch_size = 2

  # Build model graph
  mode = tf.contrib.learn.ModeKeys.TRAIN
  params_ = CopyGenSeq2Seq.default_params().copy()
  params_.update({
      "vocab_source": vocab_path,
      "vocab_target": vocab_path,
    }
  )
  model = CopyGenSeq2Seq( params = params_, mode = mode, vocab_instance=vocab)

  tf.logging.info(vocab_path)

  input_pipeline_ = input_pipeline.FeaturedTFRecordInputPipeline(
    params={
      "files":[record_path],
      "shuffle": True
    },
    mode=mode
  )
  input_fn = training_utils.create_input_fn(
    pipeline=input_pipeline_, batch_size=batch_size)
  features, labels = input_fn()
  fetches = model(features, labels, None)
  fetches = [_ for _ in fetches if _ is not None]
  from tensorflow.python import debug as tf_debug

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    sess.run(tf.tables_initializer())
    with tf.contrib.slim.queues.QueueRunners(sess):
      # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
      fetches_ = sess.run(fetches)

  return model, fetches_

if __name__ == "__main__":

  vocab_path = "/home/bigdata/active_project/test_seq2seq_py2/yard_seq2seq/q2q_sim_95/data/vocab/shared.vocab.txt"
  source_path = "/home/bigdata/active_project/test_seq2seq_py2/yard_seq2seq/q2q_sim_95/data/debug/sources.txt"
  target_path = "/home/bigdata/active_project/test_seq2seq_py2/yard_seq2seq/q2q_sim_95/data/debug/targets.txt"
  record_path = "/home/bigdata/active_project/test_seq2seq_py2/yard_seq2seq/q2q_sim_95/data/debug/debug.tfrecord"
  from os.path import join
  test_copy_gen_model(record_path,
                     vocab_path=vocab_path)

  # test_model(source_path, target_path, vocab_path)


