#encoding=utf-8
import os
import sys
import codecs
import json

def get_dir_or_file_path(dir_or_path, max_deep=1):
  if os.path.isdir(dir_or_path):
    all_paths = [os.path.join(dir_or_path, name) for name in os.listdir(dir_or_path)]
  else:
    all_paths = [dir_or_path]
  return all_paths

def jsonWrite(d, file_path, indent=2):
  with codecs.open(file_path, "w", "utf-8") as f:
    json.dump(d,f,ensure_ascii=False, indent=indent)

def split_join(s):
  return "".join(s.strip().split())

def read_pred_result(pred_path):
  """从预测模型出的转写文件读取出数据
  :param pred_path: 
  :return: 原始问题和预测结果
  """
  source_list, pred_list = [], []
  f = codecs.open(pred_path, "r", "utf-8")
  while True:

    # handle two conditions:
    # {s0}\t{s1}  ||  {s0}\n{s1}\n
    line0 = f.readline()
    if not line0:
      break
    s = line0
    t = f.readline()
    if not t:
      break
    s = s.replace("SEQUENCE_END", "")
    t = t.replace("SEQUENCE_END", "")
    tokenized_s, tokenized_t = s.strip(), t.strip()
    source_list.append(tokenized_s)
    pred_list.append(tokenized_t)
    pair_split_line = f.readline()
    if not pair_split_line:
      break
  return source_list, pred_list