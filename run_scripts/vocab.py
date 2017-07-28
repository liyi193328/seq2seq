#encoding=utf-8

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

__author__ = "liyi"
__date__ = "2017-07-06"

import os
import sys
import argparse
import collections
import logging
import codecs
import charset


def generate_vocab(source_paths, save_path, delimiter=" ", max_vocab_size=150000, min_freq=10,
                   filter_en=True, filter_num=True, verb=True):

  # Counter for all tokens in the vocabulary
  vocab_cnt = collections.Counter()

  for i, path in enumerate(source_paths):
    f = codecs.open(path, "r", "utf-8")
    while True:
      line = f.readline()
      if not line:
        break
      if delimiter == "":
        tokens = list(line.strip())
      else:
        tokens = line.strip().split(delimiter)
      tokens = [_ for _ in tokens if len(_) > 0]
      vocab_cnt.update(tokens)

  ##filter vocab
  if filter_en is True or filter_num is True:
    new_vocab_cnt = collections.Counter()
    for word in vocab_cnt:
      skip = False
      for index, char in enumerate(word):
        if filter_en and charset.is_alphabet(char):
          skip = True
        elif filter_num and charset.is_number(char):
          skip = True
        elif charset.is_chinese_punctuation(char):  ##solve 。榜样
          if len(word) > 1:
            print("{} is not right".format(word))
            skip = True
        if skip is True:
          break
      if skip is False:
        new_vocab_cnt[word] = vocab_cnt[word]
    vocab_cnt = new_vocab_cnt

  logging.info("Found %d unique tokens in the vocabulary.", len(vocab_cnt))

  # Filter tokens below the frequency threshold
  if min_freq > 0:
    filtered_tokens = [(w, c) for w, c in vocab_cnt.most_common()
                       if c > min_freq]
    cnt = collections.Counter(dict(filtered_tokens))

  logging.info("Found %d unique tokens with frequency > %d.",
               len(vocab_cnt), min_freq)

  # Sort tokens by 1. frequency 2. lexically to break ties
  word_with_counts = vocab_cnt.most_common()
  word_with_counts = sorted(
      word_with_counts, key=lambda x: (x[1], x[0]), reverse=True)

  # Take only max-vocab
  if max_vocab_size is not None:
    word_with_counts = word_with_counts[:max_vocab_size]

  if save_path is not None:
    save_path = os.path.abspath(save_path)
    if os.path.exists(os.path.dirname(save_path)) == False:
        os.makedirs(os.path.dirname(save_path))
    with codecs.open(save_path, "w", "utf-8") as f:
      for word, count in word_with_counts:
        # print("{}\t{}".format(word, count))
        f.write("{}\t{}\n".format(word, count))
    print("generate vocab path {}".format(save_path))
  return word_with_counts

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Generate vocabulary for a tokenized text file.")
  parser.add_argument(
      "--min_frequency",
      dest="min_frequency",
      type=int,
      default=0,
      help="Minimum frequency of a word to be included in the vocabulary.")
  parser.add_argument(
      "--max_vocab_size",
      dest="max_vocab_size",
      type=int,
      help="Maximum number of tokens in the vocabulary")
  parser.add_argument(
      "--downcase",
      dest="downcase",
      type=bool,
      help="If set to true, downcase all text before processing.",
      default=False)
  parser.add_argument(
      "infile",
      nargs="+",
      type=str,
      help="Input tokenized text file to be processed.")
  parser.add_argument(
      "--delimiter",
      dest="delimiter",
      type=str,
      default=" ",
      help="Delimiter character for tokenizing. Use \" \" and \"\" for word and char level respectively."
  )
  args = parser.parse_args()
