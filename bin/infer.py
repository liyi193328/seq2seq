#! /usr/bin/env python
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

""" Generates model predictions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pydoc import locate

import os
import yaml
import seq2seq
import codecs

from six import string_types

import tensorflow as tf
from tensorflow import gfile

from seq2seq import tasks, models
from seq2seq.configurable import _maybe_load_yaml, _deep_merge_dict
from seq2seq.data import input_pipeline
from seq2seq.inference import create_inference_graph
from seq2seq.training import utils as training_utils

tf.flags.DEFINE_string("tasks", "{}", "List of inference tasks to run.")
tf.flags.DEFINE_string("model_params", "{}", """Optionally overwrite model
                        parameters for inference""")

tf.flags.DEFINE_integer("num_threads", None,"num threads[None]")
tf.flags.DEFINE_string("config_path", None,
                       """Path to a YAML configuration file defining FLAG
                       values and hyperparameters. Refer to the documentation
                       for more details.""")

tf.flags.DEFINE_string("input_pipeline", None,
                       """Defines how input data should be loaded.
                       A YAML string.""")

tf.flags.DEFINE_string("model_dir", None, "directory to load model from")
tf.flags.DEFINE_string("all_model_list_path", None, "every line ia s model path, task_index's work infer the correspond model")
tf.flags.DEFINE_string("checkpoint_path", None,
                       """Full path to the checkpoint to be loaded. If None,
                       the latest checkpoint in the model dir is used.""")
tf.flags.DEFINE_boolean("single_machine", True, "in one machine or not[True]")
tf.flags.DEFINE_integer("batch_size", 32, "the train/dev batch size")
tf.flags.DEFINE_string("save_pred_path", None, "save pred path[None], None is print only")
tf.flags.DEFINE_string("job_name", None, "None | worker | ps")
tf.flags.DEFINE_integer("task_index", None, "distributed worker index to infer")
tf.flags.DEFINE_string("data_parts",None, "data parts, split by ,; every time infer data_parts[task_index]'s source data")
FLAGS = tf.flags.FLAGS

modelpathAndprefix = None
if FLAGS.all_model_list_path is not None:
  modelpathAndprefix = [path.strip().split() for path in codecs.open(FLAGS.all_model_list_path, "r", "utf-8").readlines()]
  tf.logging.info("modelname_prefix: {}".format(modelpathAndprefix))

if FLAGS.task_index is not None:
  assert FLAGS.model_dir is None
  assert  modelpathAndprefix is not None
  FLAGS.__setattr__("model_dir", modelpathAndprefix[FLAGS.task_index][0])
  pred_dir = "/mnt/yardcephfs/mmyard/g_wxg_td_prc/mng/turingli/query_rewrite/43w_86w_new_data_infer"
  save_name = modelpathAndprefix[FLAGS.task_index][1] + "." + os.path.basename(FLAGS.model_dir) #prefix.model_name
  FLAGS.__setattr__("save_pred_path", os.path.join(pred_dir, save_name))

data_index = None
if FLAGS.job_name == "worker" and FLAGS.task_index is not None:
  all_data_parts = FLAGS.data_parts.split(",")
  data_index = all_data_parts[FLAGS.task_index]
  if FLAGS.single_machine is True:
    cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices is not None:
      gpu_devices = cuda_visible_devices.split(",")
      if FLAGS.task_index >= len(gpu_devices):
        gpu_th = ""
      else:
        gpu_th = str(gpu_devices[FLAGS.task_index])
      os.environ["CUDA_VISIBLE_DEVICES"] = gpu_th

print("cuda_visible_devices:{}".format(os.getenv("CUDA_VISIBLE_DEVICES")))
print ("data_index:{}".format(data_index))

def main(_argv):
  """Program entry point.
  """

  # Load flags from config file
  if FLAGS.config_path:
    with gfile.GFile(FLAGS.config_path) as config_file:
      config_flags = yaml.load(config_file)
      for flag_key, flag_value in config_flags.items():
        setattr(FLAGS, flag_key, flag_value)

  if isinstance(FLAGS.tasks, string_types):
    FLAGS.tasks = _maybe_load_yaml(FLAGS.tasks)

  if isinstance(FLAGS.input_pipeline, string_types):
    FLAGS.input_pipeline = _maybe_load_yaml(FLAGS.input_pipeline)

  if data_index is not None:
    source_prefix = FLAGS.input_pipeline["params"]["source_files"][0]
    FLAGS.input_pipeline["params"]["source_files"][0] = source_prefix + "_part_{}".format(data_index)

  input_pipeline_infer = input_pipeline.make_input_pipeline_from_def(
      FLAGS.input_pipeline, mode=tf.contrib.learn.ModeKeys.INFER,
      shuffle=False, num_epochs=1)

  # Load saved training options
  train_options = training_utils.TrainOptions.load(FLAGS.model_dir)

  # Create the model
  model_cls = locate(train_options.model_class) or \
    getattr(models, train_options.model_class)
  model_params = train_options.model_params
  model_params = _deep_merge_dict(
      model_params, _maybe_load_yaml(FLAGS.model_params))
  model = model_cls(
      params=model_params,
      mode=tf.contrib.learn.ModeKeys.INFER)

  checkpoint_path = FLAGS.checkpoint_path
  if not checkpoint_path or checkpoint_path == "None":
    checkpoint_path = tf.train.latest_checkpoint(FLAGS.model_dir)
    ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
    global_steps = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
  else:
    #get static global steps from checkpoint path
    global_steps = int(os.path.basename(checkpoint_path).split('-')[1])
  if FLAGS.save_pred_path is not None:
    if data_index is not None:
      FLAGS.save_pred_path = FLAGS.save_pred_path + "_pred_part_{}".format(data_index)
    FLAGS.save_pred_path = FLAGS.save_pred_path + "." + str(global_steps)

  if os.path.exists(FLAGS.save_pred_path):
    tf.logging.warning("{} exists before, exit infer".format(FLAGS.save_pred_path))
    return
  tf.logging.warning("will write to {}".format(FLAGS.save_pred_path))

  # Load inference tasks
  hooks = []
  for tdict in FLAGS.tasks:
    if not "params" in tdict:
      tdict["params"] = {}
    if tdict["class"] == "DecodeText":
      tdict["params"]["save_pred_path"] = FLAGS.save_pred_path
    task_cls = locate(tdict["class"]) or getattr(tasks, tdict["class"])
    task = task_cls(tdict["params"])
    hooks.append(task)

  # Create the graph used for inference
  predictions, _, _ = create_inference_graph(
      model=model,
      input_pipeline=input_pipeline_infer,
      batch_size=FLAGS.batch_size)

  saver = tf.train.Saver()

  def session_init_op(_scaffold, sess):
    saver.restore(sess, checkpoint_path)
    tf.logging.info("Restored model from %s", checkpoint_path)

  scaffold = tf.train.Scaffold(init_fn=session_init_op)
  session_creator = tf.train.ChiefSessionCreator(scaffold=scaffold,config=tf.ConfigProto(intra_op_parallelism_threads=FLAGS.num_threads))
  with tf.train.MonitoredSession(
      session_creator=session_creator,
      hooks=hooks) as sess:

    # Run until the inputs are exhausted
    while not sess.should_stop():
      sess.run([])

if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
