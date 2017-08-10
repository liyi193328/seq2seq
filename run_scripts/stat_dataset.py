# -*- coding: utf-8 -*-

import os
import sys
import codecs
import click
import utils
import pandas as pd


@click.command()
@click.argument("path_or_dir")
@click.option("--result_path", default=None, help="save_result path")
def stat_parallel_dataset(path_or_dir, result_path=None):
  paths = utils.get_dir_or_file_path(path_or_dir)
  vocab_dict = set()
  s_len = []
  t_len = []
  overlap = []
  for i, path in enumerate(paths):
    print("reading from {}".format(path))
    f = codecs.open(path, "r", "utf-8")
    for line in f:
      if not line:
        break
      try:
        s, t = line.strip().split("\t")
      except Exception:
        continue
      s_token = s.strip().split(" ")
      t_token = t.strip().split(" ")
      s_len.append(len(s_token))
      t_len.append(len(t_token))
      ov = 0
      for x in s_token:
        if x in t_token:
          ov += 1
      overlap.append(ov)
      for x in s_token:
        vocab_dict.add(x)
      for x in t_token:
        vocab_dict.add(x)
  vocab_size = len(vocab_dict)
  result = {
    "vocab_size": vocab_size,
    "s_ave_len": sum(s_len) / float(len(s_len)),
    "t_ave_len": sum(t_len) / float(len(t_len)),
    "ave_overlap": sum(overlap) / float(len(overlap))
  }
  print(result)

  return result

if __name__ == "__main__":
  stat_parallel_dataset()
