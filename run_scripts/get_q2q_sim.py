#encoding=utf-8

"""
get q2q similariy from service
curl -X POST -d '{"query":"你知罪吗", "question":"你知道错了吗"}' http://10.191.15.89:40919/cgi-bin/ranker/q2qsimilarity
warp this comand for whole file
"""
import os
import sys
import time
import json
import codecs
import argparse
import multiprocessing as MP
import subprocess


def get_q2q_sim(q0, q1):
  q0 = split_join(q0)
  q1 = split_join(q1)
  q0 = q0.replace("SEQUENCE_END", "")
  q1 = q1.replace("SEQUENCE_END", "")
  cmd = '''curl -X POST -d '{"''' + """query":"{}", "question":"{}" """.format(q0, q1) + """}' http://10.191.15.89:40919/cgi-bin/ranker/q2qsimilarity"""
  print(cmd)
  pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
  outputs, errs = pro.communicate()
  return outputs

def jsonWrite(file_path, d, indent=2):
  with codecs.open(file_path, "w", "utf-8") as f:
    json.dump(f,d,ensure_ascii=False, indent=indent)

def split_join(s):
  return "".join(s.strip().split())

def get_q2q_file(file_path, save_path, parallels=MP.cpu_count() - 2, time_dealy=2):
  f = codecs.open(file_path, "r","utf-8")
  results = []
  pool = MP.Pool(parallels)
  pros = []

  while True:
    s = f.readline()
    if not s:
      break
    t = f.readline()
    f.readline()
    pro = pool.apply_async( get_q2q_sim, args=(s,t,) )
    pros.append(pro)
    print("{}th process starts...".format(len(pros)))
    results.append({"source":s, "predict":t})
    if len(pros) % 10000 == 0:
      print("waiting for {} secs".format(time_dealy))
      time.sleep(time_dealy)
  for i, pro in enumerate(pros):
    outputs = pro.get().strip()
    rj = json.loads(outputs.strip())
    if str(rj["data"]["error"]) == "0":
        results[i]["score"] = rj["data"]["score"]
    if "score" not in results[i]:
      results[i]["score"] = -1

  jsonWrite(file_path, results, indent=2)

if __name__ == "__main__":

  # get_q2q_sim("我爱中国", "我爱中华人民共和国")

  parser = argparse.ArgumentParser()
  parser.add_argument("file_path", type=str, help="model preidct path")
  parser.add_argument("save_path", type=str, help="save result path")
  parser.add_argument("--pnums", default=max(1, MP.cpu_count() - 5), type=int, help="parallels")
  args = parser.parse_args()

  get_q2q_file(args.file_path, args.save_path,parallels=args.pnums)