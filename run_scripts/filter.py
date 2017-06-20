#encoding=utf-8

import sys
if sys.version_info[0] == 2:
  reload(sys)
  sys.setdefaultencoding("utf-8")

import os
import codecs
import json
import click

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

if __name__ == "__main__":
  # filter_low_sim_from_json()
  cli()