
#encoding=utf-8



import os
import sys
import json
import codecs
import copy
import get_q2q_sim
import pandas as pd

def describ(json_path, all_ques_path, save_path):
  all_ques = set()
  with codecs.open(all_ques_path, "r", "utf-8") as f:
    ques = [get_q2q_sim.split_join(v) for v in f.readlines()]
    ques = set(ques)
  print("all ques nums: {}".format(len(ques)))
  with codecs.open(json_path, "r", "utf-8") as f:
    data = json.load(f)
  score_list = []
  new_data = []
  gen_nums = 0
  for i, t in enumerate(data):
    score_list.append(t["score"])
    xt = copy.copy(t)
    if t["predict"] not in ques:
      xt["in_ques_set"] = False
      gen_nums += 1
    else:
      xt["in_ques_set"] = True
    new_data.append(xt)
  series_score = pd.Series(score_list)
  print(series_score.describe())
  print("predict sents: {}, generate sents: {}".format(len(data), gen_nums))
  get_q2q_sim.jsonWrite(new_data, save_path)

if __name__ == "__main__":
  describ(r"E:\active_project\run_tasks\seq2seq\q2q_12w\all_ques.sim.json", r"E:\active_project\Dataset\all_ques.txt", r"E:\active_project\run_tasks\seq2seq\q2q_12w\new.sim.json")




