#encoding=utf-8

"""
get q2q similariy from service
curl -X POST -d '{"query":"你知罪吗", "question":"你知道错了吗"}' http://10.191.15.89:40919/cgi-bin/ranker/q2qsimilarity
warp this comand for whole file
"""

import os
import sys
if sys.version_info[0] == 2:
  reload(sys)
  sys.setdefaultencoding("utf-8")
import time
import json
import codecs
import argparse
import traceback
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

def get_q2q_file_sinle_pro(file_path, save_path, parallels=1, time_dealy=2, join_space=False,
                           delimiter="\t",in_one_line=True):
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
    s, t = s.strip(), t.strip()
    if join_space:
      s = split_join(s)
      t = split_join(t)
    s = s.replace("SEQUENCE_END", "")
    t = t.replace("SEQUENCE_END", "")
    outputs = get_q2q_sim(s,t)
    tmp_dict = {"source":s, "pred":t}
    try:
      rj = json.loads(outputs.strip())
      if "data" not in rj:
        print("errors, {}".format(rj))
        tmp_dict["score"] = -1
      else:
        if str(rj["data"]["error"]) == "0":
          tmp_dict["score"] = rj["data"]["score"]
        if "score" not in tmp_dict:
          tmp_dict["score"] = -1
    except Exception:
      tmp_dict["score"] = -1
      print(outputs)
      traceback.print_exc()
    results.append(tmp_dict)
    if len(results) % 100000 == 0:
      print("finished {} sents".format(len(results)))
      now_time = time.time()
      print("cost {}".format(now_time - begin))
      time.sleep(time_dealy)

  jsonWrite(results, save_path , indent=2)

def get_q2q_file(file_path, save_path, parallels=MP.cpu_count() - 2, time_dealy=2,
                 in_one_line=False, delimiter="\t", join_space=False):

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
    s , t = s.strip(), t.strip()
    if join_space:
      s = split_join(s)
      t = split_join(t)
    s = s.replace("SEQUENCE_END", "")
    t = t.replace("SEQUENCE_END", "")
    pro = pool.apply_async( get_q2q_sim, args=(s,t,) )
    pros.append(pro)
    print("{}th process starts...".format(len(pros)))
    results.append({"source":s, "predict":t})
    if len(pros) % 3000 == 0:
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
    if i and i % 100000 == 0:
      print("finished {}".format(i/nums))
      jsonWrite(results[0:i], save_path, indent=2)

  jsonWrite(results,save_path,indent=2)


if __name__ == "__main__":

  # get_q2q_sim("我爱中国", "我爱中华人民共和国")

  parser = argparse.ArgumentParser()
  parser.add_argument("file_path", type=str, help="model preidct path")
  parser.add_argument("save_path", type=str, help="save result path")
  parser.add_argument("--pnums", default=max(1, MP.cpu_count() - 5), type=int, help="parallels[cpu.count - 5]")
  args = parser.parse_args()
  if args.pnums <= 1:
    get_q2q_file_sinle_pro(args.file_path, args.save_path)
  else:
    get_q2q_file(args.file_path, args.save_path,parallels=args.pnums)

