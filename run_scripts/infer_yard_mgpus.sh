#!/usr/bin/env bash

echo "get cloud extra info:"
echo "$@"

mgpus=$1

##changable according to your data dir and task

echo "must enter the project seq2seq dir, then bash run_scripts/infer_yard.sh"
echo "export TASK_NAME=xx; export MODEL_NAME=xx;export SAVE_PRED_NAME=xx;"
echo "[export SOURCE_PRED_PREFIX=xx;] [export SAVE_PRED_PREFIX=xx;] [export DATA_ROOT=xx;] [export SOURCE_NAME=all_ques] [export PRED_DIR=xx]"
echo "bash infer_yard_mgpus.sh"

SEQ2SEQ_PROJECT_DIR=${PWD}
#TASK_NAME=ques_10w
TASK_NAME=${TASK_NAME:=q2q_sim_95}
MODEL_NAME=${MODEL_NAME:=model0}
CONFIG_APP_NAME=${CONFIG_APP_NAME:=q2q_sim_95}
SAVE_PRED_NAME=${SAVE_PRED_NAME:="${MODEL_NAME}_pred.txt"}
echo "task_name: $TASK_NAME"
echo "model_name: $MODEL_NAME"

DATA_ROOT=${DATA_ROOT:=/mnt/yardcephfs/mmyard/g_wxg_td_prc/turingli}
TASK_ROOT=$DATA_ROOT/$TASK_NAME
SOURCE_NAME=${SOURCE_NAME:="all_ques"}
DEFAULT_SOURCE_PREFIX=$TASK_ROOT/ques_parts/$SOURCE_NAME
SOURCE_PRED_PREFIX=${SOURCE_PRED_PREFIX:=$DEFAULT_SOURCE_PREFIX}
echo "source_pred_prefix: $SOURCE_PRED_PREFIX"

DEFAULT_PRED_DIR=$TASK_ROOT/predict
PRED_DIR=${PRED_DIR:=$DEFAULT_PRED_DIR}
SAVE_PRED_PREFIX=${SAVE_PRED_PREFIX:=$PRED_DIR/$SAVE_PRED_NAME}
echo "save_pred_prefix:$SAVE_PRED_PREFIX"

DEFAULT_MODEL_ROOT=$TASK_ROOT/model
MODEL_DIR_ROOT=${MODEL_DIR_ROOT:=$DEFAULT_MODEL_ROOT}
MAYBE_MODEL_DIR=${MODEL_DIR_ROOT}/${MODEL_NAME}
MODEL_DIR=${MODEL_DIR:=$MAYBE_MODEL_DIR}
echo "MODEL_DIR: $MODEL_DIR"

#yard_ques_gen_10w_config
MAYBE_CONFIG_DIR=${SEQ2SEQ_PROJECT_DIR}/example_configs/${CONFIG_APP_NAME}
CONFIG_DIR=${CONFIG_DIR:=$MAYBE_CONFIG_DIR}

echo "seq2seq project dir: ${SEQ2SEQ_PROJECT_DIR}"


VOCAB_SOURCE=$DATA_DIR/vocab/shared.vocab.txt
VOCAB_TARGET=$DATA_DIR/vocab/shared.vocab.txt
TRAIN_SOURCES=$DATA_DIR/train/sources.txt
TRAIN_TARGETS=$DATA_DIR/train/targets.txt
DEV_SOURCES=$DATA_DIR/dev/sources.txt
DEV_TARGETS=$DATA_DIR/dev/targets.txt
TEST_SOURCES=$DATA_DIR/test/sources.txt
TEST_TARGETS=$DATA_DIR/test/targets.txt

BATCH_SIZE=${BATCH_SIZE:=64}
echo "batch_size:$BATCH_SIZE"

export PYTHONPATH=${SEQ2SEQ_PROJECT_DIR}:${PYTHONPATH}
echo "PYTHONPATH: ${PYTHONPATH}"

cd ${SEQ2SEQ_PROJECT_DIR}
echo "now in dir:$PWD, begin to train model"

PROJECT_DIR=${PWD}

beam_width=${BEAM_WIDTH:=10}

#mkdir -p $PRED_DIR

cd ${PROJECT_DIR}

for ((i=0; i<$mgpus; i++))
do
    SOURCE_PATH=${SOURCE_PRED_PREFIX}_part_${i}
    PRED_PATH=${SAVE_PRED_PREFIX}_part_${i}
    echo "source_path:$SOURCE_PATH"
    echo "pred_path:$PRED_PATH"
    python -m bin.infer \
      --tasks "
        - class: DecodeText
        - class: DumpBeams
          params:
            file: ${PRED_DIR}/${i}th_beams.npz" \
      --model_params "
      inference.beam_search.length_penalty_weight: 1.0
      inference.beam_search.choose_successors_fn: choose_top_k_mask_unk
      inference.beam_search.beam_width: $beam_width " \
      --model_dir $MODEL_DIR \
      --mgpus ${mgpus} \
      --gpu_index ${i} \
      --input_pipeline "
        class: ParallelTextInputPipeline
        params:
          source_files:
            - ${SOURCE_PATH}" \
      --save_pred_path ${PRED_PATH} &
done


#  --tasks "
#    - class: DecodeText"