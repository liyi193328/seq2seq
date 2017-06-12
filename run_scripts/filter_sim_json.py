#encoding=utf-8

import sys
if sys.version_info[0] == 2:
  reload(sys)
  sys.setdefaultencoding("utf-8")

import os
import codecs
import json
import click

@click.command()
@click.argument("json_path", "json path from get_q2q_sim.py, every cell contain source, predict, score")
@click.argument("save_path", "store filter result path, every line is tab.join(s,p,score)")
@click.option("--sim_threshold", default=0.95,help="lowest sim threshold[0.95]")
def filter_low_sim_from_json(json_path, save_path, sim_threshold=0.95):
  data = json.load(codecs.open(json_path, "r", "utf-8"))
  fout = codecs.open(save_path, "w", "utf-8")
  for i, cell in enumerate(data):
    if cell["score"] >= sim_threshold:
      s = "\t".join([cell["source"], cell["predict"], cell["score"]]) + "\n"
      fout.write(s)
  fout.close()

if __name__ == "__main__":
  filter_low_sim_from_json()