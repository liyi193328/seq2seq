#encoding=utf-8

"""
function:
1. get file text's similiary through q2q service
  get q2q similariy from service
  curl -X POST -d '{"query":"你知罪吗", "question":"你知道错了吗"}' http://10.191.15.89:40919/cgi-bin/ranker/q2qsimilarity
  warp this comand for whole file

author: liyi
"""

import os
import sys
if sys.version_info[0] == 2:
  reload(sys)
  sys.setdefaultencoding("utf-8")
import time
import json
import codecs
import click
import argparse
import traceback
import utils
import multiprocessing as MP
import subprocess

def get_q2q_sim(q0, q1):
  cmd = ''' curl -X POST -d '{{"query":"{}", "question":"{}" }}' http://10.191.15.89:40919/cgi-bin/ranker/q2qsimilarity '''.format(q0, q1)
  # print(cmd)
  pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
  outputs, errs = pro.communicate()
  return outputs

def jsonWrite(d, file_path, indent=2):
  with codecs.open(file_path, "w", "utf-8") as f:
    json.dump(d,f,ensure_ascii=False, indent=indent)

def split_join(s):
  return "".join(s.strip().split())

@click.command()
@click.argument("file_path", type=str)
@click.argument("save_path", type=str)
@click.option("--in_one_line", is_flag=True, help=r"question pair in one line, or in two lines(the end split")
@click.option("--parallels", default=max(1, MP.cpu_count() - 5), type=int, help="parallels[cpu.count - 5]")
def get_q2q_file_sinle_pro(file_path, save_path, parallels=1, time_dealy=2,
                           delimiter="\t",in_one_line=False):
  # file_path is tokenized file
  f = codecs.open(file_path, "r","utf-8")
  results = []
  pros = []

  begin = time.time()
  while True:
    # handle two conditions:
    # {s0}\t{s1}  ||  {s0}\n{s1}\n
    line0 = f.readline()
    if not line0:
      break
    if in_one_line is True:
      try:
        s, t = line0.split(delimiter)
      except Exception:
        print("errors happen in {}".format(line0))
        continue
    else:
      s = line0
      t = f.readline()
      f.readline()
    tokenized_s, tokenized_t = s.strip(), t.strip()
    s = split_join(s)
    t = split_join(t)
    s = s.replace("SEQUENCE_END", "")
    t = t.replace("SEQUENCE_END", "")
    outputs = get_q2q_sim(s,t)
    tmp_dict = {"source":tokenized_s, "predict":tokenized_t}
    try:
      rj = json.loads(outputs.strip())
      if "data" not in rj:
        tmp_dict["score"] = -1
      else:
        if str(rj["data"]["error"]) == "0":
          tmp_dict["score"] = rj["data"]["score"]
        if "score" not in tmp_dict:
          tmp_dict["score"] = -1
    except Exception:
      tmp_dict["score"] = -1
      traceback.print_exc()
    if tmp_dict["score"] == -1:
      print("{}\t{}".format(tmp_dict["source"], tmp_dict["predict"]))
    results.append(tmp_dict)
    if len(results) % 200000 == 0:
      print("finished {} sents".format(len(results)))
      now_time = time.time()
      print("cost {}".format(now_time - begin))
      time.sleep(time_dealy)
      jsonWrite(results, save_path, indent=2)

  jsonWrite(results, save_path , indent=2)


def get_q2q_file(file_path, save_path, parallels=MP.cpu_count() - 2, time_dealy=1,
                 in_one_line=True, delimiter="\t"):

  # file_path is tokenized
  f = codecs.open(file_path, "r","utf-8")
  results = []
  pool = MP.Pool(parallels)
  pros = []

  while True:
    # handle two conditions:
    # {s0}\t{s1}  ||  {s0}\n{s1}\n
    line0 = f.readline()
    if not line0:
      break
    if in_one_line is True:
      try:
        s, t = line0.split(delimiter)
      except Exception:
        print("errors happen in {}".format(line0))
        continue
    else:
      s = line0
      t = f.readline()
      f.readline()
    tokenized_s , tokenized_t = s.strip(), t.strip()
    s = split_join(tokenized_s)
    t = split_join(tokenized_t)
    s = s.replace("SEQUENCE_END", "")
    t = t.replace("SEQUENCE_END", "")
    pro = pool.apply_async( get_q2q_sim, args=(s, t,) )
    pros.append(pro)
    if len(pros) % 30000 == 0:
      print("{}th process starts...".format(len(pros)))
    results.append({"source":tokenized_s, "predict":tokenized_t})
    if len(pros) % 100000 == 0:
      print("waiting for {} secs".format(time_dealy))
      time.sleep(time_dealy)

  nums = len(pros)
  for i, pro in enumerate(pros):
    outputs = pro.get().strip()
    try:
      rj = json.loads(outputs.strip())
      if "data" not in rj:
        print("errors, {}".format(rj))
        results[i]["score"] = -1
      else:
        if str(rj["data"]["error"]) == "0":
            results[i]["score"] = rj["data"]["score"]
        if "score" not in results[i]:
          results[i]["score"] = -1
    except Exception:
      results[i]["score"] = -1
      print(results[i])
      traceback.print_exc()
    if i and i % 10000:
      print("finined {} sents and ratio {}".format(i, i/float(nums)))

    # if i and i % 300000 == 0:
    #   print("finished {}".format(i/nums))
    #   jsonWrite(results[0:i], save_path, indent=2)
  jsonWrite(results, save_path, indent=2)
  flat_results = []
  for result in results:
    flat_results.append([ result["source"], result["predict"], result["score"] ])
  utils.write_list_to_file(flat_results, save_path.replace(".json", ".txt") )

@click.command()
@click.argument("source_dir", type=str)
@click.argument("save_prefix", type=str)
@click.option("--in_one_line", is_flag=True, help=r"question pair in one line, or in two lines(the end split")
@click.option("--parallels", default=max(1, MP.cpu_count() - 5), type=int, help="parallels[cpu.count - 5]")
def get_q2q_sim_dir(source_dir, save_prefix, parallels=MP.cpu_count() - 2, time_dealy=1,
                 in_one_line=False, delimiter="\t"):
    score_dir = os.path.dirname(save_prefix)
    if os.path.exists(score_dir) == False:
      os.makedirs(score_dir)
    source_paths = utils.get_dir_or_file_path(source_dir)

    with click.progressbar(source_paths, label="get score") as bar:
      for i, source_path in enumerate(bar):
        save_path = os.path.join(save_prefix, "_part_{}.json".format(i))
        get_q2q_file(source_path, save_path, parallels=parallels, time_dealy=time_dealy, in_one_line=in_one_line,delimiter=delimiter)
        print("{} done!".format(source_path))
    final_score_path = save_prefix + ".final"
    utils.merge_dir(score_dir, final_score_path, suffix=".txt")

if __name__ == "__main__":

  # get_q2q_sim("我爱中国", "我爱中华人民共和国")

  # parser = argparse.ArgumentParser()
  # parser.add_argument("file_path", type=str, help="model predict path")
  # parser.add_argument("save_path", type=str, help="save result path")
  # parser.add_argument("--in_one_line", type=str, default="False", help=r"question pair in one line, or in two lines(the end split")
  # parser.add_argument("--pnums", default=max(1, MP.cpu_count() - 5), type=int, help="parallels[cpu.count - 5]")
  # args = parser.parse_args()
  # args.in_one_line = args.in_one_line in ["Yes", "y", "Y", "yes", "true", "True"]
  # if args.pnums <= 1:
  #   get_q2q_file_sinle_pro(args.file_path, args.save_path, in_one_line=args.in_one_line)
  # else:
  #   get_q2q_file(args.file_path, args.save_path,parallels=args.pnums, in_one_line=args.in_one_line)

  get_q2q_sim_dir()

