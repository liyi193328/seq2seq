#encoding=utf-8


import os
import sys
import subprocess
import argparse

real_path = os.path.dirname( os.path.abspath(__file__) ) #os.getcwd()
project_root = os.path.dirname(real_path)
my_env = os.environ.copy()
Python = sys.executable
print(Python)

def call_infer_fn(yaml_conf_path, source_path, model_dir, save_dir, config, save_name):

  cmd = """
  {Python} -m bin.infer \
  --config_path={yaml_conf_path}
  --model_dir {model_dir} \
  --input_pipeline "
    class: ParallelTextInputPipeline
    params:
      source_files:
        - {source_path}" \
  > {save_dir}/{save_name}
  """.format(
    Python=Python,
    config_path=yaml_conf_path,
    source_path=source_path,
    save_dir=save_dir,
    model_dir=model_dir,
    beam_width=config["beam_width"],
    save_name=save_name
  )
  print("lanch: {}".format(cmd))

  res = subprocess.run(cmd,shell=True,universal_newlines=True,cwd=project_root, env=my_env)
  print(res)

  return

if __name__ == "__main__":
  model_dir = "/home/bigdata/active_project/test_seq2seq_py2/yard_seq2seq/data/q2q_12w/model/test0"
  save_dir = "/home/bigdata/active_project/test_seq2seq_py2/yard_seq2seq/data/q2q_12w/predict/test0"
  source_path = "/home/bigdata/active_project/test_seq2seq_py2/yard_seq2seq/data/q2q_12w/data/test/sources.txt"
  parser = argparse.ArgumentParser()
  parser.add_argument("yaml_conf_path", type=str, help="pred yaml conf path")
  parser.add_argument("--save_name", type=str, default="predictions.txt", help="final pred results saved to {save_dir}/{save_name}")
  args = parser.parse_args()
  config = {
    "beam_width": 5
  }
  call_infer_fn(source_path, model_dir, save_dir, config, args.save_name)