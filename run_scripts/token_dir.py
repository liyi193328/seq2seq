#encoding=utf-8

import os
import sys
import codecs
import pyltp
import click
import utils
import multiprocessing as MP
from pyltp import Segmentor
from pyltp import SentenceSplitter

# if "LTP_DATA_DIR" not in os.environ:
#   print("must set LTP_DATA_DIR environment")
#   sys.exit(-1)

LTP_DATA_DIR = os.environ.get("LTP_DATA_DIR", "/home/bigdata/software/LTP/ltp_data")
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')
segmentor = Segmentor()  # 初始化实例
segmentor.load(cws_model_path)  # 加载模型

def token_file(path, save_path, delimiter="\t"):
  f = codecs.open(path, "r", "utf-8")
  fout = codecs.open(save_path, "w", "utf-8")
  while True:
    line = f.readline()
    if not line:
      break
    t = line.strip().split(delimiter)
    token_str_list = []
    for v in t:
      tokens = segmentor.segment(v)
      token_str_list.append(" ".join(tokens))
    fout.write(delimiter.join(token_str_list) + "\n")
  print("tokenize from {} to {}".format(path, save_path))
  return True

def parallel_token_dir(file_path_or_dir, save_dir, suffix=".token", pnums = MP.cpu_count() - 1, delimiter="\t"):
  file_paths = utils.get_dir_or_file_path(file_path_or_dir)
  pros = []
  pnums = min(len(file_paths), pnums)
  print("use {} to tokenize".format(pnums))
  pool = MP.Pool(pnums)
  for path in file_paths:
    file_name = os.path.basename(path).split(".")[0]
    save_path = os.path.join(save_dir, file_name + suffix)
    pro = pool.apply_async(token_file, args=(path, save_path), kwds={"delimiter": delimiter})
    pros.append(pro)
  for i, pro in enumerate(pros):
    result = pro.get()
  print("done")

@click.command()
@click.argument("file_path_or_dir")
@click.argument("save_dir")
@click.option("--delimiter", default="\t", type=str, help="split every line, and tokenize every element in them")
@click.option("--pnums", type=int, help="num of process to tokenize file or dir", default=MP.cpu_count()-1)
def cli(file_path_or_dir, save_dir, suffix=".token", pnums = MP.cpu_count() - 1, delimiter="\t"):
  parallel_token_dir(file_path_or_dir, save_dir, suffix=".token", pnums = MP.cpu_count() - 1, delimiter="\t")

if __name__ == "__main__":
  cli()