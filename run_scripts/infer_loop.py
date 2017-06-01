#encoding=utf-8


import os
import sys
import subprocess
import argparse
import time
from datetime import datetime, timedelta

real_path = os.path.dirname( os.path.abspath(__file__) ) #os.getcwd()
project_root = os.path.dirname(real_path)
my_env = os.environ.copy()
Python = sys.executable
print(Python)

def loop_infer(sleep_secs, total_secs, yaml_conf_path, source_path, model_dir, save_dir, config, save_name, checkpoint_path="None"):
  start = time.time()

  while time.time() - start < total_secs:
    call_infer_fn(yaml_conf_path, source_path, model_dir, save_dir, config, save_name, checkpoint_path=checkpoint_path)
    time.sleep(sleep_secs)
  return

def call_infer_fn(yaml_conf_path, source_path, model_dir, save_dir, config, save_name, checkpoint_path="None"):

  yaml_conf_path = os.path.abspath(yaml_conf_path)
  source_path = os.path.abspath(source_path)
  model_dir = os.path.abspath(model_dir)
  save_dir = os.path.abspath(save_dir)
  if os.path.exists(save_dir) == False:
    os.makedirs(save_dir)
  cmd = """
  export CUDA_VISIBLE_DEVICES=""; \
  {Python} -m bin.infer \
  --config_path={yaml_conf_path} \
  --model_dir={model_dir} \
  --checkpoint_path={checkpoint_path} \
  --input_pipeline="
    class: ParallelTextInputPipeline
    params:
      source_files:
        - {source_path}" \
  --save_pred_path={save_dir}/{save_name}
  """.format(
    Python=Python,
    yaml_conf_path=yaml_conf_path,
    source_path=source_path,
    save_dir=save_dir,
    checkpoint_path=checkpoint_path,
    model_dir=model_dir,
    save_name=save_name
  )
  print("lanch: {}".format(cmd))

  pro = subprocess.Popen(cmd,stdout=subprocess.PIPE, stdin=subprocess.PIPE, shell=True,universal_newlines=True,cwd=project_root, env=my_env)
  res = pro.communicate()
  print(res)

  return

if __name__ == "__main__":
  model_name="add_residual_connections"
  model_dir = "/home/bigdata/active_project/test_seq2seq_py2/yard_seq2seq/data/q2q_12w_cancel_dup/model/{}".format(model_name)
  save_dir = "/home/bigdata/active_project/test_seq2seq_py2/yard_seq2seq/data/q2q_12w_cancel_dup/predict/{}".format(model_name)
  source_path = "/home/bigdata/active_project/test_seq2seq_py2/yard_seq2seq/data/q2q_12w_cancel_dup/data/test/sources.txt"
  parser = argparse.ArgumentParser()
  parser.add_argument("yaml_conf_path", type=str, help="pred yaml conf path")
  parser.add_argument("--infer_once",type=str, help="infer once or infer loop", default="true")
  parser.add_argument("--sleep_secs", type=int, default=20*60, help="sleep secs after one infer")
  parser.add_argument("--total_secs", type=int, default=20*3600, help="total secs for loop infer")
  parser.add_argument("--source_path", type=str, default=source_path)
  parser.add_argument("--model_dir", type=str, default=model_dir)
  parser.add_argument("--checkpoint_path", type=str, default="None", help="if specify checkpoint_path, model_dir will be ignore[None]")
  parser.add_argument("--save_dir", type=str, default=save_dir)
  parser.add_argument("--save_name", type=str, default="predictions.txt", help="final pred results saved to {save_dir}/{save_name}")
  args = parser.parse_args()
  config = {
    "beam_width": 5
  }
  if args.infer_once in ["true", "True", "Y", "yes"]:
    call_infer_fn(args.yaml_conf_path, args.source_path, args.model_dir, args.save_dir, config, args.save_name, checkpoint_path=args.checkpoint_path)
  else:
    loop_infer(args.sleep_secs, args.total_secs, args.yaml_conf_path, args.source_path, args.model_dir, args.save_dir, config, args.save_name, checkpoint_path=args.checkpoint_path)