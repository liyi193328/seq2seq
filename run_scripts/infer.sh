#!/usr/bin/env bash

MODEL_DIR="/home/bigdata/active_project/test_seq2seq_py2/yard_seq2seq/data/q2q_12w/model/test0"
PRED_DIR="/home/bigdata/active_project/test_seq2seq_py2/yard_seq2seq/data/q2q_12w/predict/test0"
DEV_SOURCES="/home/bigdata/active_project/test_seq2seq_py2/yard_seq2seq/data/q2q_12w/data/test/sources.txt"

PROJECT_DIR=${PWD}

beam_width=5

mkdir -p $PRED_DIR

cd ${PROJECT_DIR}

python -m bin.infer \
  --tasks "
    - class: DecodeText
      params:
        unk_replace: True" \
  --model_dir $MODEL_DIR \
  --input_pipeline "
    class: ParallelTextInputPipeline
    params:
      source_files:
        - $DEV_SOURCES" \
  > ${PRED_DIR}/predictions.txt