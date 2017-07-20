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


def t(file_path, default_value=None):
  x = create_vocabulary_lookup_table(file_path)
  print(x)
  return x


from seq2seq.graph_utils import templatemethod

@templatemethod("trial")
def trial(x):
    w0 = tf.get_variable('w', [])
    w1 = tf.get_variable("w", [])
    assert  w0 is w1
    return tf.reduce_sum(x) * w0

def my_trail(x, share_variable_name):
  var1 = tf.get_variable(share_variable_name, shape=[])
  return tf.reduce_sum(x) * var1

template_my = tf.make_template("template_my", my_trail, share_variable_name="my_v")

def test_trial():
  y = tf.placeholder(tf.float32, [None])
  z = tf.placeholder(tf.float32, [None])

  a_y = trial(y)
  # a_z = trial(z)

  # a_y_t = template_my(y)
  # a_z_t = template_my(z)

  s = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  print(tf.global_variables())


  # print(a_y_t.eval(feed_dict={y: [1.1, 1.9]}
  #                  ))
  print(a_y.eval(feed_dict={y: [1.1, 1.9]}))
  # print(a_z_t.eval(feed_dict={y: [1.1, 1.9]}))
  # print(a_z.eval(feed_dict={z: [1.9, 1.1]}))

def get_vocab_list(vocab_path):
  lines = codecs.open(vocab_path, "r", "utf-8")
  vocab = []
  for line in lines:
    vocab.append(line.strip().split("\t")[0])
  return vocab

def test_copy_gen_model(source_path = None, target_path = None, vocab_path=None):

  tf.logging.set_verbosity(tf.logging.INFO)
  batch_size = 2
  input_depth = 4
  sequence_length = 10

  if vocab_path is None:
    # Create vocabulary
    vocab_list = [str(_) for _ in range(10)]
    vocab_list += ["笑う", "泣く", "了解", "はい", "＾＿＾"]
    vocab_size = len(vocab_list)
    vocab_file = test_utils.create_temporary_vocab_file(vocab_list)
    vocab_info = vocab.get_vocab_info(vocab_file.name)
    vocab_path = vocab_file.name
    tf.logging.info(vocab_file.name)
  else:
    vocab_info = vocab.get_vocab_info(vocab_path)
    vocab_list = get_vocab_list(vocab_path)

  extend_vocab = vocab_list + ["中国", "爱", "你"]

  tf.contrib.framework.get_or_create_global_step()
  source_len = sequence_length + 5
  target_len = sequence_length + 10
  source = " ".join(np.random.choice(extend_vocab, source_len))
  target = " ".join(np.random.choice(extend_vocab, target_len))

  is_tmp_file = False
  if source_path is None and target_path is None:
    is_tmp_file = True
    sources_file, targets_file = test_utils.create_temp_parallel_data(
      sources=[source], targets=[target])
    source_path = sources_file.name
    target_path = targets_file.name

  # Build model graph
  mode = tf.contrib.learn.ModeKeys.TRAIN
  params_ = CopyGenSeq2Seq.default_params().copy()
  params_.update({
      "vocab_source": vocab_path,
      "vocab_target": vocab_path,
    }
  )
  model = CopyGenSeq2Seq(params = params_, mode = mode)

  tf.logging.info(source_path)
  tf.logging.info(target_path)

  input_pipeline_ = input_pipeline.ParallelTextInputPipeline(
    params={
      "source_files": [source_path],
      "target_files": [target_path]
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

  if is_tmp_file:
    sources_file.close()
    targets_file.close()

  return model, fetches_

if __name__ == "__main__":

  # test_trial()
  vocab_path = "/home/bigdata/active_project/test_seq2seq_py2/yard_seq2seq/q2q_sim_95_80w/data/vocab/shared.vocab.txt"
  # t(vocab_path)
  dir = "/home/bigdata/active_project/run_tasks/text_sum/debug"
  from os.path import join
  test_copy_gen_model(source_path=join(dir,"source.txt"), target_path=join(dir, "target.txt"),
                     vocab_path=join(dir,"vocab.txt"))

  # test_trial()


