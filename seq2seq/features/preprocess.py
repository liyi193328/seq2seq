#encoding=utf-8

import codecs
import os
import sys
import csv
import numpy
import tensorflow as tf
from seq2seq.data import vocab


def words_to_id(tokens, vocab_cls):
  token_ids = []
  for token in tokens:
    token_id = vocab_cls.word2id(token)
    token_ids.append(token_id)
  token_ids_string = " ".join([str(v) for v in token_ids])
  return token_ids_string

def save():
  ex = tf.train.Example()
  # pylint: disable=E1101
  ex.features.feature["source"].bytes_list.value.extend(
    [source.encode("utf-8")])
  ex.features.feature["target"].bytes_list.value.extend(
    [target.encode("utf-8")])
  writer.write(ex.SerializeToString())

def get_features(save_path, vocab_cls, source_path, target_path=None, delimeter=" "):
  """get source features or both(TODO)
  :param save_path:
  :param vocab_cls:
  :param source_path:
  :param target_path:
  :param delimeter:
  :return:
  """
  fs = codecs.open(source_path, "r", "utf-8")
  ft = codecs.open(target_path, "r", "utf-8")
  source_lines = fs.readlines()
  target_lines = ft.readlines()
  assert  len(source_lines) == len(target_lines), (len(source_lines), len(target_lines))
  writer = tf.python_io.TFRecordWriter(save_path)

  for source_line, target_line in zip(source_lines, target_lines):

    source_tokens = source_line.strip().split(delimeter)
    target_tokens = target_line.strip().split(delimeter)

    source_ids = words_to_id(source_tokens, vocab_cls)
    target_ids = words_to_id(target_tokens, vocab_cls)

    ex = tf.train.Example()
    ex.features.feature["source"].bytes_list.value.extend([source_line.encode("utf-8")])
    ex.features.feature["target"].bytes_list.value.extend([target_line.encode("utf-8")])



vocab_instance = vocab.Vocab("/home/bigdata/active_project/test_seq2seq_py2/yard_seq2seq/q2q_sim_95/data/vocab/shared.vocab.txt")

print(vocab_instance.special_vocab)
