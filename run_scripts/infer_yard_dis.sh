#!/usr/bin/env bash

echo "get cloud extra info:"
echo "$@"

echo "getting cluster info:"
echo $1
echo $2
echo $3
echo $4


##changable according to your data dir and task

echo "must enter the project seq2seq dir, then bash run_scripts/run_yard_xx.sh"

echo "TASK_NAME=${TASK_NAME} must contain in config_dir and data_dir"

SEQ2SEQ_PROJECT_DIR=${PWD}
#TASK_NAME=ques_10w
RUN_NAME=${RUN_NAME:=run_try}
TASK_NAME=${TASK_NAME:=stable_single}

ROOT=${ROOT:=/mnt/yardcephfs/mmyard/g_wxg_td_prc/mng/turingli}
DEFAULT_TASK_ROOT=$ROOT/$TASK_NAME
TASK_ROOT=${TASK_ROOT:=$DEFAULT_TASK_ROOT}

#model dir:{root}/{task_name}/model/{run_name}/*
DEFAULT_MODEL_DIR=${TASK_ROOT}/model/${RUN_NAME}
MODEL_DIR=${MODEL_DIR:=$DEFAULT_MODEL_DIR}
mkdir -p ${MODEL_DIR}
echo "MODEL_DIR: $MODEL_DIR"
chmod -R 777 ${MODEL_DIR}

#data dir:{root}/{task_name}/data/[train|dev]
DEFAULT_DATA_DIR=${TASK_ROOT}/data
DATA_DIR=${DATA_DIR:=$DEFAULT_DATA_DIR}
echo "DATA_DIR: ${DATA_DIR}"

#config dir: {root}/{task_name}/config/{run_name}/*
CONFIG_APP_NAME=${CONFIG_APP_NAME:=$RUN_NAME}
MAYBE_CONFIG_DIR=${TASK_ROOT}/config/${CONFIG_APP_NAME}
CONFIG_DIR=${CONFIG_DIR:=$MAYBE_CONFIG_DIR}

echo "config dir: ${CONFIG_DIR}"
echo "seq2seq project dir: ${SEQ2SEQ_PROJECT_DIR}"

CLEAR_OUTPUT_DIR=${CLEAR_OUTPUT_DIR:=False}

VOCAB_SOURCE=$DATA_DIR/vocab/shared.vocab.txt
VOCAB_TARGET=$DATA_DIR/vocab/shared.vocab.txt
TRAIN_SOURCES=$DATA_DIR/train/sources.txt
TRAIN_TARGETS=$DATA_DIR/train/targets.txt
DEV_SOURCES=$DATA_DIR/dev/sources.txt
DEV_TARGETS=$DATA_DIR/dev/targets.txt
TEST_SOURCES=$DATA_DIR/test/sources.txt
TEST_TARGETS=$DATA_DIR/test/targets.txt

TRAIN_STEPS=${TRAIN_STEPS:=5000000}
BATCH_SIZE=${BATCH_SIZE:=64}
EVAL_EVERY_N_STEPS=${EVAL_EVERY_N_STEPS:=10000}
SAVE_CHECK_SECS=${SAVE_CHECK_SECS:=1800}
KEEP_CHECK_MAX=${KEEP_CHECK_MAX:=20}

MODEL_LIST_PATH=${MODEL_LIST_PATH:=None}

echo "#########"
echo "MODEL_LIST_PATH: $MODEL_LIST_PATH"
echo "#########"

export PYTHONPATH=${SEQ2SEQ_PROJECT_DIR}:${PYTHONPATH}
echo "PYTHONPATH: ${PYTHONPATH}"

cd ${SEQ2SEQ_PROJECT_DIR}
echo "now in dir:$PWD, begin to train model"

python -m bin.infer \
  --tasks="
    - class: DecodeText
      params:
        unk_replace: True
        " \
  --model_params="
  inference.beam_search.length_penalty_weight: 1.0
  inference.beam_search.choose_successors_fn: choose_top_k_mask_unk
  inference.beam_search.beam_width: $beam_width " \
  --model_dir=$MODEL_DIR \
  $1 \
  $2 \
  $3 \
  $4 \
  --all_model_list_path=${MODEL_LIST_PATH} \
  --input_pipeline "
    class: ParallelTextInputPipeline
    params:
      source_files:
        - /mnt/yardcephfs/mmyard/g_wxg_td_prc/mng/turingli/query_rewrite/stable_new_dual/data/test/sources.txt
        "