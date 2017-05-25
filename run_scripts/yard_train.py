#encoding=utf-8


import os
import sys
import argparse
import subprocess
import shutil
import logging
from os.path import join

DATA_DIR = None
DATA_ROOT = os.environ.get("DATA_ROOT",None)
APP_NAME = os.environ.get("APP_NAME",None)
CONFIG_NAME = os.environ.get("CONFIG_NAME", None)
if DATA_ROOT is not None and APP_NAME is not None:
    DATA_DIR = os.path.join(DATA_ROOT, APP_NAME)
DATA_DIR = os.environ.get("DATA_DIR", DATA_DIR)

logging.warn("must run in project dir: python run_scripts/train.py --xx --yy")

def train_model(data_dir, save_model_dir, config_name, clear_output_dir="True", vocab_path=None,
                batch_size = 32, cloud="True", schedule="default", train_steps=10000,
                ps_hosts="", worker_hosts="", job_name=None, task_index=0, project_path=None
                ):

    if project_path is None:
        project_path = os.getcwd()
    logging.info("cur run path: {}".format(project_path))
    if data_dir is None:
        raise ValueError("must provide data dir in command or in environment variable: DATA_DIR|DATA_ROOT/APP_NAME")

    data_dir = os.path.abspath(data_dir)
    common_path = os.path.dirname(data_dir)

    #{}/example_configs/{config_name}
    config_dir = os.path.join(project_path, os.path.join("example_configs", config_name))
    save_model_dir = os.path.abspath(save_model_dir)

    if vocab_path is None:
        source_vocab_path=join(data_dir, "vocab/shared.vocab.txt")
        target_vocab_path=join(data_dir, "vocab/shared.vocab.txt")

    train_source_path= join(data_dir, "train/sources.txt")
    train_target_path = join(data_dir, "train/targets.txt")
    dev_source_path =join(data_dir, "dev/sources.txt")
    dev_target_path =join(data_dir, "dev/targets.txt")

    cmd = '''
    python -m bin.train \
      --ps_hosts={ps_hosts} \
      --worker_hosts={worker_hosts} \
      --job_name={job_name} \
      --task_index={task_index} \
      --config_paths="
          {config_dir}/nmt_small.yml,
          {config_dir}/train_seq2seq.yml,
          {config_dir}/text_metrics_bpe.yml" \
      --model_params="
          vocab_source: {source_vocab_path}
          vocab_target: {target_vocab_path}" \
      --cloud={cloud} \
      --schedule={schedule}
      --batch_size={batch_size} \
      --train_steps={train_steps} \
      --output_dir={save_model_dir} \
      --clear_output_dir={clear_output_dir} \
      '''.format(
        **locals()
        )
    print("runing: {}".format(cmd))
    r = subprocess.run(cmd, cwd=project_path, shell=True, check=True)
    print(r)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("save_model_dir", help="save model dir", type=str)
    parser.add_argument("config_name", help="app config name, config file will search in [./example_configs/{config_name/]", type=str)
    parser.add_argument("--data_dir", default=DATA_DIR, help="run task data dir", type=str)
    parser.add_argument("--clear_output_dir", default="True", help="clear model or not before train model[True]", type=str)
    parser.add_argument("--batch_size", default=32, help="batch_size[32]", type=int)
    parser.add_argument("--ps_hosts", default="", help="ps_hosts,common split", type=str)
    parser.add_argument("--worker_hosts", default="", help="worker_hosts,common split", type=str)
    parser.add_argument("--job_name", default="None", help="ps, worker['None']", type=str)
    parser.add_argument("--task_index", default=0, help="worker index [0]", type=int)
    parser.add_argument("--cloud", default="True", help="distributed mode?[True]", type=str)
    parser.add_argument("--schedule", default="default", help="schedule for this server[default]", type=str)
    parser.add_argument("--train_steps", default=10000, help="global max train steps[10000]", type=int)
    parser.add_argument("--vocab_path", default=None, help="target vocab path[{data_dir}/vocab/shared.vocab.txt]", type=str)
    args = parser.parse_args()
    train_model(
        args.data_dir, args.save_model_dir, args.config_name, clear_output_dir=args.clear_output_dir, vocab_path=args.vocab_path,
        batch_size=args.batch_size, cloud=args.cloud, schedule=args.schedule, train_steps=args.train_steps,
        ps_hosts=args.ps_hosts, worker_hosts=args.worker_hosts, job_name=args.job_name,
        task_index=args.task_index
        )



