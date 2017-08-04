#encoding=utf-8

import os
import click
import codecs
import utils
from collections import OrderedDict

def get_q2q_dict(path):
  q2q_dict = OrderedDict()
  all_paths = utils.get_dir_or_file_path(path)
  for path in all_paths:
    f = codecs.open(path, "r", "utf-8")
    lines = f.readlines()
    for i, line in enumerate(lines):
      st = line.strip().split("\t")
      st = [v.strip() for v in st]
      if len(st) < 2:
        continue
      try:
        s, t, = st[0], st[1]
      except Exception:
        print(st)
        print(line)
      if s not in q2q_dict:
        q2q_dict[s] = []
      q2q_dict[s].append(t)
  return q2q_dict

def No_key_Overlap(d1, d2):
  q = set(list(d1.keys())).intersection(list(d2.keys()))
  if len(q) != 0:
    print(len(q))
    print(list(q)[0:10])
  if len(q) > 0:
    print("\n".join(q))
    return False
  return True

@click.command()
@click.argument("train_paths", nargs=-1)
@click.argument("dev_paths", nargs=1)
@click.argument("test_paths", nargs=1)
def check_no_overlap(train_paths, dev_paths, test_paths):
  def ques_set(paths):
    keys = set()
    if type(paths) not in (tuple, list):
      paths = [paths]
    for path in paths:
      for line in codecs.open(path, "r", "utf_8"):
        keys.add(line.strip())
    return keys
  train_keys = ques_set(train_paths)
  dev_keys = ques_set(dev_paths)
  test_keys = ques_set(test_paths)
  x = train_keys.intersection(dev_keys)
  if len(x) > 0:
    print("train overlap with dev:")
    print("\n".join(list(x)))
  else:
    print("train No dev")
  y = train_keys.intersection(test_keys)
  if len(y) > 0:
    print("train overlap with test:")
    print("\n".join(list[y]))
  else:
    print("train No test")
  z = dev_keys.intersection(test_keys)
  if len(z) > 0:
    print("dev with test:")
    print("\n".join(list(z)))
  else:
    print("dev No test")

if __name__ == "__main__":
  check_no_overlap()