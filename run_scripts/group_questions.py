#encoding=utf-8

import os
import sys
import codecs
import make_seq2seq_data
import click



if __name__ == "__main__":
  '''
  make("../data/q2q_pos.train", "../data/q2q_12w_cancel_dup_dual", keep=None, add_dual=True)
  get_unique_ques("../data/q2q_pos.train", "../data/all_ques.txt", add_dual=True)
  '''
  make()


