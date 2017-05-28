#encoding=utf-8

import os
import sys
import codecs
import make_seq2seq_data


def get_source_target(qs, keys, xs, ys,keep_one=False):
  sources = []
  targets = []
  for key in keys:
    vs = qs[key]
    if keep_one is True:
      vs = [vs[0]]
    sources.extend([xs[v] for v in vs])
    targets.extend([ys[v] for v in vs])
  return sources, targets

def make(file_path, save_dir, ratio_list=None):
  from os.path import join
  if os.path.exists(save_dir) == False:
    os.makedirs(save_dir)
  lines = codecs.open(file_path,"r","utf-8").readlines()
  qs = {}
  xs = []
  ys = []
  for i, line in enumerate(lines):
   t = line.strip().split("\t")
   x = t[0].strip()
   y = t[1].strip()
   xs.append(x)
   ys.append(y)
   if x not in qs:
     qs[x] = []
   qs[x].append(i)
  if ratio_list is None:
   ratio_list = [0.93, 0.95, 1.0]

  q_num = len(qs)
  ques_list = list(qs.keys())
  train_index = int(q_num * ratio_list[0])
  eval_index = int(q_num * ratio_list[1])
  test_index = int(q_num * ratio_list[2])

  train_dir = join(save_dir, "train")
  eval_dir = join(save_dir, "dev")
  test_dir = join(save_dir, "test")

  train_keys, eval_keys, test_keys = ques_list[0:train_index], \
                                     ques_list[train_index:eval_index],\
                                     ques_list[eval_index:test_index]

  train_s, train_t = get_source_target(qs, train_keys, xs, ys)
  eval_s, eval_t = get_source_target(qs, eval_keys, xs, ys, keep_one=True)
  test_s, test_t = get_source_target(qs, test_keys, xs, ys, keep_one=True)

  make_seq2seq_data.write_some_data(train_s, train_t, train_dir)
  make_seq2seq_data.write_some_data(eval_s, eval_t, eval_dir)
  make_seq2seq_data.write_some_data(test_s, test_t,test_dir)


if __name__ == "__main__":
  make("../data/q2q_pos.train", "../data/q2q_12w_cancel_dup")



