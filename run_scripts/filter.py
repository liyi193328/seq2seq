#encoding=utf-8

import sys
import glob
if sys.version_info[0] == 2:
  reload(sys)
  sys.setdefaultencoding("utf-8")

import os
import codecs
import json
import click
import utils

"""
1. 过滤问题： filter_questions
2. 给定了相似度的json文件，过滤相似度低的预测对，filter_low_sim_from_json
"""


@click.group()
def cli():
  pass

@click.command()
@click.argument("path")
@click.argument("save_path")
@click.option("--least_token_num", type=int, default=3, help="this sentence least token num")
@click.option("--max_token_num", type=int, default=15, help="max token num in sentence")
def filter_questions(path,save_path, least_token_num=3, max_token_num=15):
  f = codecs.open(path, "r", "utf-8")
  fo = codecs.open(save_path, "w", "utf-8")
  xcnt = 0
  ques_set = set()
  while True:
    line = f.readline()
    if not line:
      break
    x = line.strip().split(" ")
    if len(x) >= least_token_num and len(x) <= max_token_num:
      if line.strip() not in ques_set:
        ques_set.add(line.strip())
        fo.write(line)
        xcnt += 1
  fo.close()
  f.close()
  print ("now have {} sentences".format(xcnt))

@click.command(name="merge_pred_dir")
@click.argument("pred_dirs", nargs=-1)
@click.argument("save_path", type=str)
@click.argument("all_ques_path_or_dir", type=str)
@click.option("--unk_path",type=str, default=None, help="save pred with unk")
@click.option("--res_ques_path",type=str, default=None, help="remain ques not done")
def merge_and_unique_pred_result(pred_dirs, save_path, all_ques_path_or_dir, unk_path=None, res_ques_path=None):
  all_ques_paths = utils.get_dir_or_file_path(all_ques_path_or_dir)
  print("getting ques set")
  ques_set = set()
  for ques_path in all_ques_paths:
    print("reading from {}".format(ques_path))
    lines = codecs.open(ques_path, "r", "utf-8").readlines()
    for line in lines:
      ques_set.add(line.strip())
  print("all ques num: {}".format(len(ques_set)))
  all_pred_path = []
  for pred_dir in pred_dirs:
    all_pred_path.extend(utils.get_dir_or_file_path(pred_dir))
  done_ques = set()
  done_cnt = 0
  fout = codecs.open(save_path, "w", "utf-8")
  funk = None
  unk_pred_nums = 0
  if unk_path is not None:
    funk = codecs.open(unk_path, "w", "utf-8")
  with click.progressbar(all_pred_path,label="reading pred path") as bar:
    for pred_path in bar:
      source_list, pred_list = utils.read_pred_result(pred_path)
      for source, pred in zip(source_list, pred_list):
        if source not in done_ques:
          done_ques.add(source)
          done_cnt += 1
          wst = "\n".join([source, pred]) + "\n\n"
          if unk_path is not None:
            if "UNK" in pred:
              unk_pred_nums += 1
              funk.write(wst)
            else:
              fout.write(wst)
          else:
            fout.write(wst)
  if unk_path is not None:
    print("rewrite's pred contain unk: {}".format(unk_pred_nums))
  print("rewrite ques num: {}".format(done_cnt))

  res_ques = ques_set - done_ques
  print("remain ques num: {}".format(len(res_ques)))
  if res_ques_path is not None:
    with codecs.open(res_ques_path, "w", "utf-8") as f:
      for res_que in res_ques:
        f.write(res_que + "\n")
    print("save res ques to {}".format(res_ques_path))

@click.command()
@click.argument("json_path", "json path from get_q2q_sim.py, every cell contain source, predict, score")
@click.option("--save_path", default=None, help="store filter result path, every line is tab.join(s,p,score), None print on screen")
@click.option("--sim_threshold", default=0.95,help="lowest sim threshold[0.95]")
def filter_low_sim_from_json(json_path, save_path=None, sim_threshold=0.95):
  data = json.load(codecs.open(json_path, "r", "utf-8"))
  fout = codecs.open(save_path, "w", "utf-8")
  for i, cell in enumerate(data):
    if i % 100000 == 0:
      print("finished {}".format(i/len(data)))
    try:
      if cell["score"] >= sim_threshold:
        s = "\t".join([cell["source"], cell["predict"], str(cell["score"])]) + "\n"
        fout.write(s)
    except KeyError:
      print(cell)
      continue
  fout.close()

cli.add_command(filter_questions)
cli.add_command(filter_low_sim_from_json)
cli.add_command(merge_and_unique_pred_result)

if __name__ == "__main__":
  # filter_low_sim_from_json()
  cli()