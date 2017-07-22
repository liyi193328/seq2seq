#encoding=utf-8

import codecs
import os
import sys
import csv
import numpy
from seq2seq.data import vocab


def get_side_features(tokens, vocab_cls):
  token_ids = []
  for token in tokens:
    token_id = vocab_cls.word2id(token)
    token_ids.append(token_id)
  features = {}

vocab_instance = vocab.Vocab("/home/bigdata/active_project/test_seq2seq_py2/yard_seq2seq/q2q_sim_95/data/vocab/shared.vocab.txt")

print(vocab_instance.special_vocab)
