
#encoding=utf-8



import os
import sys
import json
import codecs
import copy
import get_q2q_sim
import pandas as pd
import click

@click.command()
@click.argument("path")
@click.argument("all_ques_path")
@click.option("--save_path", default=None, help="if none, not save")
@click.option("--is_json", type=bool, is_flag=True, help="if path is json file")
def describ(path, all_ques_path, is_json, save_path=None):
  all_ques = set()
  with codecs.open(all_ques_path, "r", "utf-8") as f:
    ques = [v.strip() for v in f.readlines()]
    ques = set(ques)
  print("all ques nums: {}".format(len(ques)))
  with codecs.open(path, "r", "utf-8") as f:
    if is_json:
        data = json.load(f)
    else:
      data = []
      while True:
        s = f.readline().strip()
        if not s:
          break
        t = f.readline().strip()
        f.readline()
        s = s.replace("SEQUENCE_END", "")
        t = t.replace("SEQUENCE_END", "")
        s, t = s.strip(), t.strip()
        data.append({"source":s,"predict":t})
  score_list = []
  new_data = []
  gen_nums = 0
  for i, t in enumerate(data):
    if "score" in t:
      score_list.append(t["score"])
    xt = copy.copy(t)
    if t["predict"] == t["source"]:
      xt["in_ques_set"] = True
    elif t["predict"] not in ques:
      xt["in_ques_set"] = False
      gen_nums += 1
    else:
      xt["in_ques_set"] = True
    new_data.append(xt)
  series_score = pd.Series(score_list)
  print(series_score.describe())
  print("predict sents: {}, no dup sents: {}".format(len(data), gen_nums))
  if save_path is not None:
    get_q2q_sim.jsonWrite(new_data, save_path)

if __name__ == "__main__":
  describ()
  # describ(r"E:\active_project\run_tasks\seq2seq\q2q_12w\all_ques.sim.json", r"E:\active_project\Dataset\all_ques.txt", r"E:\active_project\run_tasks\seq2seq\q2q_12w\new.sim.json")




