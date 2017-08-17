#encoding=utf-8

import codecs
import os
import sys
import csv
import numpy as np
import logging
import tensorflow as tf
import click
import six
import pyltp
from collections import OrderedDict
import seq2seq.features.nlp as NLP
from seq2seq.data import vocab
from seq2seq.features import global_vars, utils
from seq2seq.features import SpecialWords, SpecialWordsIns

logger = logging.getLogger(__name__)

def words_to_id(tokens, vocab_cls):
  token_ids = []
  for token in tokens:
    token_id = vocab_cls.word2id(token)
    token_ids.append(token_id)
  return token_ids

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def join_str(array, delimeter=" "):

  if type(array) != list:
    array = [array]
  def convert_unicode(v):
    if type(v) == float or type(v) == int:
      return str(v)
    return v

  array = [convert_unicode(v) for v in array]
  return delimeter.join(array)

def get_extend_source_ids(source_tokens, source_ids, vocab_cls, unique=False):

  assert len(source_tokens) == len(source_ids), len(source_tokens)
  source_oov_list = []
  extend_source_ids = []
  oov_start_id = vocab_cls.size()

  for i, source_id in enumerate(source_ids):
    token = source_tokens[i]
    if source_id == vocab_cls.special_vocab.UNK:
      if unique is True:
        try:
          ix = source_oov_list.index(token)
          new_id = ix + oov_start_id
        except ValueError:
          new_id = len(source_oov_list) + oov_start_id
          source_oov_list.append(token)
      else:
        new_id = len(source_oov_list) + oov_start_id
        source_oov_list.append(token)
      extend_source_ids.append(new_id)
    else:
      extend_source_ids.append(source_id)

  assert len(extend_source_ids) == len(source_ids), (len(extend_source_ids), len(source_ids))

  return extend_source_ids, source_oov_list

def get_extend_target_ids(extend_source_ids, source_tokens, target_tokens, target_ids, target_unk_id):
  extend_target_ids = []
  for i, target_id in enumerate(target_ids):
    if target_id == target_unk_id:
      find_match = False
      for j, source_token in enumerate(source_tokens):
        if target_tokens[i] == source_token:
          extend_target_ids.append(extend_source_ids[j])
          find_match = True
          break
      if not find_match:
        extend_target_ids.append(target_unk_id)
    else:
      extend_target_ids.append(target_id)
  return extend_target_ids

class Preprocess(object):

  def __init__(self, vocab_path, pos_path, ner_path, tfidf_path, char_path=None, **kwargs):
    self._vocab_path = vocab_path
    self._pos_path = pos_path
    self._ner_path = ner_path
    self._tfidf_path = tfidf_path
    self._char_path = char_path
    self._word_vocab = vocab.Vocab(vocab_path)
    self._pos_vocab = vocab.Vocab(pos_path)
    self._ner_vocab = vocab.Vocab(ner_path)
    self._tfidf_vocab = NLP.Tfidf(tfidf_path, special_words=SpecialWords, default=0.0)
    self._char_path = None
    if self._char_path is not None:
      self._char_cls = vocab.Vocab(self._char_path)

  def get_source_side_features(self, source_lines):
    pass

  def get_target_side_features(self, target_lines):
    pass

  def get_aliment_extend_features(self, ):
    pass

  def get_train_features_from_files(self, parallel_text_path):
    pass

  def get_infer_features_from_files(self, source_paths):
    paths = utils.get_dir_or_file_path(source_paths)
    infer_features_list = []
    for i, path in enumerate(paths):
      lines = codecs.open(path, "r", "utf-8")
      lines = [v.strip() for v in lines]
      infer_features = self.get_infer_features_given_input(lines)
      infer_features_list.append(infer_features)
    return infer_features_list

  def get_infer_features_given_input(self, source_lines):
    pass

def get_features(parallel_text_path, save_path, vocab_cls, pos_cls, ner_cls, tfidf_cls, delimeter=" ",
                 copy_source_unique=False, source_sentence_split=False, target_sentence_split=False, tfrecord_out_nums=999):

  """get source features or both(TODO)
  :param save_path:
  :param vocab_cls:
  :param source_path:
  :param target_path:
  :param delimeter:
  :return:
  """
  '''
  Add special symbols in every sentences' first pos in SEQUENCE_START and SEQUENCE_END in sentence's end
  [s0, s1] ->[ SEQUENCE_START s0 SEQUENCE_END, SEQUENCE_START s1 SEQUENCE_END]
  '''


  f = codecs.open(parallel_text_path, "r", "utf-8")
  all_features = []
  cnt = 0
  writer = None

  for st in f:

    if cnt == 0:
      writer = tf.python_io.TFRecordWriter(save_path + ".0")
    elif (cnt + 1) % tfrecord_out_nums == 0:
      writer.close()
      x = (cnt + 1) / tfrecord_out_nums + 1
      path = save_path + ".{}".format(x)
      writer = tf.python_io.TFRecordWriter(path)

    source_line, target_line = st.strip().split("\t")
    source_line = source_line.strip()
    target_line = target_line.strip()
    source_tokens = source_line.split(delimeter)
    target_tokens = target_line.split(delimeter)
    source_sentences, target_sentences = None, None

    if source_sentence_split is True:
      source_sentences = list(pyltp.SentenceSplitter.split(source_line))
    if target_sentence_split is True:
      target_sentences = list(pyltp.SentenceSplitter.split(target_line))
    #TO DO: will convert to class and handle every sentence, every word and so on

    ##get raw source nlp features: words, pos, ner, tfidf
    source_postags = NLP.Postags(source_tokens)
    source_ners = NLP.NamedEntityRecogize(source_tokens, source_postags)

    source_tokens.append(SpecialWordsIns.SEQUENCE_END)  # special id
    source_ners.append(SpecialWordsIns.SEQUENCE_END)
    source_postags.append(SpecialWordsIns.SEQUENCE_END)

    source_ids = words_to_id(source_tokens, vocab_cls)
    source_ner_ids = [ner_cls.word2id(ner) for ner in source_ners]
    source_pos_ids = [pos_cls.word2id(pos) for pos in source_postags]
    source_tfidfs = tfidf_cls.encode(source_tokens)

    ##get raw target nlp features: words, ner
    target_ners = NLP.NamedEntityRecogize(target_tokens)
    target_tokens.insert(0, SpecialWordsIns.SEQUENCE_START)
    target_tokens.append(SpecialWordsIns.SEQUENCE_END)
    target_ners.insert(0, SpecialWordsIns.SEQUENCE_START)
    target_ners.append(SpecialWordsIns.SEQUENCE_END)

    target_ids = words_to_id(target_tokens, vocab_cls)
    target_ner_ids = [ner_cls.word2id(ner) for ner in target_ners]

    extend_source_ids, source_oov_list = get_extend_source_ids(source_tokens, source_ids, vocab_cls, unique=copy_source_unique)
    source_oov_nums = len(source_oov_list)
    extend_target_ids = get_extend_target_ids(extend_source_ids, source_tokens, target_tokens, target_ids, vocab_cls.special_vocab.UNK)

    assert len(source_ids) == len(source_tokens)
    assert len(source_ids) == len(extend_source_ids)
    assert len(target_ids) == len(target_tokens)
    assert len(target_ids) == len(extend_target_ids)

    source_keys = [
      "source_tokens", "source_ids", "extend_source_ids","source_oov_list","source_oov_nums",
      "source_ner_ids", "source_pos_ids", "source_tfidfs", "source_ners", "source_postags"
    ]
    target_keys = [
      "target_tokens", "target_ids", "extend_target_ids", "target_ner_ids", "target_ners"
    ]

    keys = source_keys + target_keys

    int64_keys = ["source_ids", "extend_source_ids","source_oov_nums","source_ner_ids", "source_pos_ids", "target_ids", "extend_target_ids", "target_ner_ids"]
    float_keys = ["source_tfidfs"]
    bytes_keys = ["source_ners", "source_postags","source_tokens", "source_oov_list", "target_tokens", "target_ners"]
    features = OrderedDict()
    vars = locals()
    ex = tf.train.Example()
    for key in keys:
      var = vars[key]
      features[key] = var
      if key in int64_keys:
        if type(var) != list:
          var = [var]
        ex.features.feature[key].int64_list.value.extend(var)
      elif key in float_keys:
        if type(var) != list:
          var = [var]
        ex.features.feature[key].float_list.value.extend(var)
      elif key in bytes_keys:
        s = join_str(var).encode("utf-8")
        ex.features.feature[key].bytes_list.value.extend([s])
      else:
        raise ValueError("{} not in int64_keys,float_keys, bytes_keys".format(key))

    writer.write(ex.SerializeToString())

    all_features.append(features)

    cnt += 1

  writer.close()

  return all_features

def read_and_decode_single_example(filename):
  # first construct a queue containing a list of filenames.
  # this lets a user split up there dataset in multiple files to keep
  # size down
  filename_queue = tf.train.string_input_producer([filename],
                                                  num_epochs=None)
  # Unlike the TFRecordWriter, the TFRecordReader is symbolic
  reader = tf.TFRecordReader()
  # One can read a single serialized example from a filename
  # serialized_example is a Tensor of type string.
  _, serialized_example = reader.read(filename_queue)
  # The serialized example is converted back to actual values.
  # One needs to describe the format of the objects to be returned

  source_keys_to_features = global_vars.source_feature_keys
  target_keys_to_features = global_vars.target_feature_keys

  #tf.FixedLenFeature
  features_cls = {}
  all_keys = set(list(source_keys_to_features.keys()) + list(target_keys_to_features.keys()))
  for key in all_keys:
    if key in source_keys_to_features:
      features_cls[key] = source_keys_to_features
    elif key in target_keys_to_features:
      features_cls[key] = target_keys_to_features[key]

  features = tf.parse_single_example(
    serialized_example,
    features= features_cls)

  return features


def print_values(values, keys, np_array=True):
  format_values = {}
  for key in keys:
    if np_array is False:
      v = values[key].values
    else:
      v = values[key]
    if v.ndim == 0:
      v = [v]
    if type(v[0]) != np.int64 and type(v[0]) != np.float32:
      v = np.char.decode(v.astype("S"), "utf-8")
      # s = " ".join([x.encode("utf-8") for x in v])
      s = " ".join(v)
    else:
      s = " ".join([str(x) for x in v])
    format_values[key] = v

    print(key)
    print(s)

  return format_values

@click.group()
def cli():
  pass

@click.command()
@click.argument("parallel_text_path")
@click.argument("save_path")
@click.argument("vocab_path")
@click.option("--pos_path", type=str)
@click.option("--ner_path", type=str)
@click.option("--tfidf_path", type=str)
@click.option("--char_path", type=str)
@click.option("--out_nums", type=int, default=10000, help="record nums every tf record file")
def generate_features(parallel_text_path, save_path, vocab_path, copy_source_unique=True,
           pos_path=None, ner_path=None, tfidf_path=None, char_path=None, out_nums=10000):

  word_vocab = vocab.Vocab(vocab_path)
  # char_vocab = vocab.Vocab(char_path)
  pos_vocab = vocab.Vocab(pos_path)
  ner_vocab = vocab.Vocab(ner_path)
  tfidf_vocab = NLP.Tfidf(tfidf_path, special_words=SpecialWords, default=0.0)

  features = get_features(parallel_text_path, save_path, word_vocab, pos_vocab,  ner_vocab, tfidf_vocab ,
                          copy_source_unique=copy_source_unique, tfrecord_out_nums=out_nums)
  # from  pprint import pprint
  # pprint(features)

@click.command()
@click.argument("load_path")
def load_features(load_path):
  from pprint import pprint
  features = read_and_decode_single_example(load_path)
  with tf.Session() as sess:
    # Required. See below for explanation
    init = tf.global_variables_initializer()
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)
    values = sess.run(features)
    keys = ["source_tokens", "source_ids", "extend_source_ids", "source_oov_list", "source_oov_nums","target_tokens", "target_ids",
            "extend_target_ids"]
    format_values = {}
    for key in keys:
      v = values[key].values
      v = np.char.decode( v.astype("S"), "utf-8" )
      format_values[key] = v[0].split()
      print(key)
      print(np.char.encode(v[0], "utf-8"))
    assert len(format_values["source_ids"]) == len(format_values["extend_source_ids"])
    print("\n")
    values = sess.run(features)
    for key in keys:
      v = values[key].values
      v = np.char.decode( v.astype("S"), "utf-8" )
      print("{}:{}".format(key, np.char.encode(v[0], "utf-8")))

@click.command()
@click.argument("load_path")
def pipeline_debug(load_path):
  from seq2seq.data import input_pipeline
  from seq2seq.features import global_vars
  pipeline = input_pipeline.FeaturedTFRecordInputPipeline(
    params={
      "files": [load_path],
      "num_epochs": 2,
      "shuffle": True
    }, mode=tf.contrib.learn.ModeKeys.TRAIN
  )
  data_provider = pipeline.make_data_provider()
  features = pipeline.read_from_data_provider(data_provider)

  keys = global_vars.source_feature_keys + global_vars.target_feature_keys

  with tf.train.MonitoredSession() as sess:
    tf.train.start_queue_runners(sess=sess)
    while not sess.should_stop():
      res = sess.run(features)
      print_values(res, keys)
      print("\n")

cli.add_command(generate_features)
cli.add_command(load_features)
cli.add_command(pipeline_debug)

if __name__ == "__main__":
  cli()
