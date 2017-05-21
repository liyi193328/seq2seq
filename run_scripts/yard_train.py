#encoding=utf-8


import os
import sys
import argparse
import subprocess
import shutil
from os.path import join


def train_model(seq2seq_dir, data_dir , save_model_dir, clear_model=True, source_vocab_path=None, target_vocab_path=None, batch_size = 32, train_steps=10000):

    data_dir = os.path.abspath(data_dir)
    common_path = os.path.dirname(data_dir)
    print(common_path, data_dir)
    
    save_model_dir = os.path.abspath(save_model_dir)
    source_vocab_path = os.path.abspath(source_vocab_path)
    target_vocab_path = os.path.abspath(target_vocab_path)
    seq2seq_dir = os.path.abspath(seq2seq_dir)
    
    VOCAB_SOURCE = source_vocab_path
    VOCAB_TARGET = target_vocab_path
    if source_vocab_path is None:
        VOCAB_SOURCE=join(data_dir, "train/vocab.sources.txt")
    if target_vocab_path is None:
        VOCAB_TARGET=join(data_dir, "train/vocab.targets.txt")
    TRAIN_SOURCES=join(data_dir, "train/sources.txt")
    TRAIN_TARGETS=join(data_dir, "train/targets.txt")
    DEV_SOURCES=join(data_dir, "dev/sources.txt")
    DEV_TARGETS=join(data_dir, "dev/targets.txt")

    DEV_TARGETS_REF=join(data_dir, "dev/targets.txt")

    TRAIN_STEPS=train_steps

    if clear_model is True:
        if os.path.exists(save_model_dir) is True:
            shutil.rmtree(save_model_dir)
        os.mkdir(save_model_dir)

    SEQ2SEQ_PROJECT_DIR=seq2seq_dir
    # os.chdir(SEQ2SEQ_PROJECT_DIR)

    cmd = '''
    python -m bin.train \
      --config_paths="
          {data_dir}/nmt_medium.yml,
          {data_dir}/train_seq2seq.yml,
          {data_dir}/text_metrics_bpe.yml" \
      --model_params "
          vocab_source: {VOCAB_SOURCE}
          vocab_target: {VOCAB_TARGET}" \
      --input_pipeline_train "
        class: ParallelTextInputPipeline
        params:
          source_files:
            - {TRAIN_SOURCES}
          target_files:
            - {TRAIN_TARGETS}" \
      --input_pipeline_dev "
        class: ParallelTextInputPipeline
        params:
           source_files:
            - {DEV_SOURCES}
           target_files:
            - {DEV_TARGETS}" \
      --batch_size {batch_size} \
      --train_steps {TRAIN_STEPS} \
      --output_dir {MODEL_DIR}
      '''.format(
        data_dir=data_dir,
        VOCAB_SOURCE=VOCAB_SOURCE,
        VOCAB_TARGET=VOCAB_TARGET,
        TRAIN_SOURCES=TRAIN_SOURCES,
        TRAIN_TARGETS=TRAIN_TARGETS,
        DEV_SOURCES=DEV_SOURCES,
        DEV_TARGETS=DEV_TARGETS,
        MODEL_DIR=save_model_dir,
        TRAIN_STEPS = TRAIN_STEPS,
        batch_size = batch_size
        )
    print("runing: {}".format(cmd))
    r = subprocess.run(cmd, cwd=seq2seq_dir, shell=True, check=True)
    print(r)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="run task data dir", type=str)
    parser.add_argument("save_model_dir", help="save model dir", type=str)
    parser.add_argument("--seq2seq_dir", default="/home/bigdata/active_project/seq2seq", help="seq2seq install dir[/home/bigdata/active_project/seq2seq]", type=str)
    parser.add_argument("--clear_model", default="True", help="clear model or not before train model[True]", type=str)
    parser.add_argument("--batch_size", default=32, help="batch_size[32]", type=int)
    parser.add_argument("--train_steps", default=10000, help="global max train steps[10000]", type=int)
    parser.add_argument("--source_vocab_path", default=None, help="source vocab path[{data_dir}/train/train.vocab.txt]", type=str)
    parser.add_argument("--target_vocab_path", default=None, help="target vocab path[{data_dir}/train/target.vocab.txt]", type=str)
    args = parser.parse_args()
    args.clear_model = args.clear_model in ["True", "Yes", "1", "y", "Y", "true"]
    train_model(args.seq2seq_dir, args.data_dir, args.save_model_dir, clear_model = args.clear_model, 
        batch_size = args.batch_size, train_steps = args.train_steps,
        source_vocab_path=args.source_vocab_path, target_vocab_path = args.target_vocab_path
        )



