#encoding=utf-8

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
"""Vocabulary related functions.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""
total vocab = special_words + actual vocab(special words first)  
"""

import codecs
import six
import csv
import collections
import tensorflow as tf
from tensorflow import gfile
from seq2seq.features import SpecialWords

#["PAD","UNK", "SEQUENCE_START", "SEQUENCE_END", "PARA_START", "PARA_END"]
SpecialVocab = collections.namedtuple("SpecialVocab", SpecialWords)

class VocabInfo(
    collections.namedtuple("VocabInfo",
                           ["path", "vocab_size", "special_vocab"])):
  """Convenience structure for vocabulary information.
  """

  @property
  def total_size(self):
    """Returns size the the base vocabulary plus the size of extra vocabulary"""
    return self.vocab_size + len(self.special_vocab)


def get_vocab_info(vocab_path, special_words=SpecialWords):
  """Creates a `VocabInfo` instance that contains the vocabulary size and
    the special vocabulary for the given file.

  Args:
    vocab_path: Path to a vocabulary file with one word per line.

  Returns:
    A VocabInfo tuple.
  """
  with gfile.GFile(vocab_path) as file:
    vocab_size = sum(1 for _ in file)
  special_vocab = get_special_vocab(0, special_words)
  return VocabInfo(vocab_path, vocab_size, special_vocab)


def get_special_vocab(first_index, special_words):
  """Returns the `SpecialVocab` instance for a given vocabulary size.
  """
  SpecialVocab = collections.namedtuple("SpecialVocab", special_words)
  return SpecialVocab(*range(first_index, first_index + len(SpecialVocab._fields)))

def create_tensor_vocab(vocab_instance):

  assert isinstance(vocab_instance, Vocab)
  word_to_ids = vocab_instance._word_to_id
  word_to_count = vocab_instance._word_to_count

  vocab, ids = list(word_to_ids.keys()), list(word_to_ids.values())
  vocab_size = len(vocab)

  counts = [word_to_count[v] for v in word_to_count]
  count_tensor = tf.constant(counts, dtype=tf.int64)
  vocab_tensor = tf.constant(vocab)
  vocab_idx_tensor = tf.constant(ids, dtype=tf.int64)

  # Create ID -> word mapping
  id_to_vocab_init = tf.contrib.lookup.KeyValueTensorInitializer(
      vocab_idx_tensor, vocab_tensor, tf.int64, tf.string)
  id_to_vocab_table = tf.contrib.lookup.HashTable(id_to_vocab_init, "UNK")

  # Create word -> id mapping
  vocab_to_id_init = tf.contrib.lookup.KeyValueTensorInitializer(
      vocab_tensor, vocab_idx_tensor, tf.string, tf.int64)
  vocab_to_id_table = tf.contrib.lookup.HashTable(vocab_to_id_init,
                                                  vocab_instance.special_vocab.UNK)

  # Create word -> count mapping
  word_to_count_init = tf.contrib.lookup.KeyValueTensorInitializer(
      vocab_tensor, count_tensor, tf.string, tf.int64)
  word_to_count_table = tf.contrib.lookup.HashTable(word_to_count_init, -1)

  return vocab_to_id_table, id_to_vocab_table, word_to_count_table, vocab_size

def create_vocabulary_lookup_table(filename, specaial_words = SpecialWords, default_value=None):

  #old version
  """Creates a lookup table for a vocabulary file.

  Args:
    filename: Path to a vocabulary file containg one word per line.
      Each word is mapped to its line number.
    default_value: UNK tokens will be mapped to this id.
      If None, UNK tokens will be mapped to [vocab_size]

    Returns:
      A tuple (vocab_to_id_table, id_to_vocab_table,
      word_to_count_table, vocab_size). The vocab size does not include
      the UNK token.
    """

  if not gfile.Exists(filename):
    raise ValueError("File does not exist: {}".format(filename))

  # Add special vocabulary items
  special_vocab = get_special_vocab(0, specaial_words)

  vocab = list(SpecialVocab._fields)
  counts = [-1. for _ in list(special_vocab._fields)]

  # Load vocabulary into memory
  with gfile.GFile(filename) as file:
    or_vocab = list(line.strip("\n") for line in file)
  has_counts = (len(or_vocab[0].split("\t")) == 2)
  if has_counts:
    or_vocab, or_counts = zip(*[_.split("\t") for _ in or_vocab])
    or_counts = [float(_) for _ in or_counts]
    or_vocab = list(or_vocab)
  else:
    or_counts = [-1. for _ in or_vocab]

  counts.extend(or_counts)
  vocab.extend(or_vocab)
  vocab_size = len(vocab)

  if default_value is None:
    default_value = special_vocab.UNK

  tf.logging.info("Creating vocabulary lookup table of size %d", vocab_size)

  vocab_tensor = tf.constant(vocab)
  count_tensor = tf.constant(counts, dtype=tf.float32)
  vocab_idx_tensor = tf.range(vocab_size, dtype=tf.int64)

  # Create ID -> word mapping
  id_to_vocab_init = tf.contrib.lookup.KeyValueTensorInitializer(
      vocab_idx_tensor, vocab_tensor, tf.int64, tf.string)
  id_to_vocab_table = tf.contrib.lookup.HashTable(id_to_vocab_init, "UNK")

  # Create word -> id mapping
  vocab_to_id_init = tf.contrib.lookup.KeyValueTensorInitializer(
      vocab_tensor, vocab_idx_tensor, tf.string, tf.int64)
  vocab_to_id_table = tf.contrib.lookup.HashTable(vocab_to_id_init,
                                                  default_value)

  # Create word -> count mapping
  word_to_count_init = tf.contrib.lookup.KeyValueTensorInitializer(
      vocab_tensor, count_tensor, tf.string, tf.float32)
  word_to_count_table = tf.contrib.lookup.HashTable(word_to_count_init, -1)

  return vocab_to_id_table, id_to_vocab_table, word_to_count_table, vocab_size

class Vocab(object):
  """Vocabulary class for mapping between words and ids (integers)"""

  def __init__(self, vocab_file, special_words=SpecialWords, max_size=None):
    """Creates a vocab of up to max_size words, reading from the vocab_file. If max_size is 0, reads the entire vocab file.

    Args:
      vocab_file: path to the vocab file, which is assumed to contain "<word> <frequency>" on each line, sorted with most frequent word first. This code doesn't actually use the frequencies, though.
      max_size: integer. The maximum size of the resulting Vocabulary."""
    self._vocab_file = vocab_file
    self._word_to_id = {}
    self._id_to_word = {}
    self._word_to_count = {}
    self._count = 0 # keeps track of total number of words in the Vocab

    if special_words is None:
      special_words = []
    special_vocab = get_special_vocab(0, special_words)
    self.special_vocab = special_vocab

    special_words = list(SpecialVocab._fields)

    print("Special words: {}".format(" ".join(special_words)))

    for w in SpecialVocab._fields:
      self._word_to_id[w] = self._count
      self._word_to_count[w] = -1
      self._id_to_word[self._count] = w
      self._count += 1

    have_nums = False
    # Read the vocab file and add words up to max_size
    with codecs.open(vocab_file, 'r', "utf-8") as vocab_f:
      for line in vocab_f:
        pieces = line.split()
        if len(pieces) == 2:
          have_nums = True
          self._word_to_count[pieces[0]] = int(pieces[1])
        w = pieces[0]
        if not have_nums:
          self._word_to_count[w] = -1
        if w in special_words:
          s = " ".join(special_words)
          raise Exception('{} shouldn\'t be in the vocab file, but {} is'.format(s,w))
        if w in self._word_to_id:
          raise Exception('Duplicated word in vocabulary file: %s' % w)
        self._word_to_id[w] = self._count
        self._id_to_word[self._count] = w
        self._count += 1
        if max_size is not None and self._count >= max_size:
          print("max_size of vocab was specified as %i; we now have %i words. Stopping reading." % (max_size, self._count))
          break

    last_word = self._id_to_word[self._count-1]
    if six.PY2:
      last_word = last_word.encode("utf-8")
    print("Finished constructing vocabulary of {} total words. Last word added: {}".format(self._count, last_word))

  def word2id(self, word):
    """Returns the id (integer) of a word (string). Returns [UNK] id if word is OOV."""
    if word not in self._word_to_id:
      return self._word_to_id[SpecialWords.UNK]
    return self._word_to_id[word]

  def id2word(self, word_id):
    """Returns the word (string) corresponding to an id (integer)."""
    if word_id not in self._id_to_word:
      raise ValueError('Id not found in vocab: %d' % word_id)
    return self._id_to_word[word_id]

  def size(self):
    """Returns the total size of the vocabulary"""
    return self._count

  def write_metadata(self, fpath):
    """Writes metadata file for Tensorboard word embedding visualizer as described here:
      https://www.tensorflow.org/get_started/embedding_viz

    Args:
      fpath: place to write the metadata file
    """
    print("Writing word embedding metadata file to %s..." % (fpath))
    with codecs.open(fpath, "w", "utf-8") as f:
      fieldnames = ['word']
      writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
      for i in range(self.size()):
        word = self._id_to_word[i]
        f.write(word+"\n")


if __name__ == "__main__":

  vocab_cls = Vocab("/home/bigdata/active_project/test_seq2seq_py2/yard_seq2seq/q2q_sim_95/data/vocab/shared.vocab.txt")
  sess = tf.InteractiveSession()
  vocab_to_id_table, id_to_vocab_table, word_to_count_table, vocab_size = create_tensor_vocab(vocab_cls)
  words = ["如何", "的", "凤仙花"]
  words = tf.constant(words, dtype=tf.string)
  ids = vocab_to_id_table.lookup(words)
  counts = word_to_count_table.lookup(words)

  sess.run(tf.global_variables_initializer())
  sess.run(tf.tables_initializer())

  print(ids.eval())
  print(counts.eval())




