#! /bin/bash/ env

#organized:
#root/task_name:
#        -data
#            -vocab
#            -train
#            -dev
#            -test
#                -sources.txt
#                -sources.txt
#        -model
#            -model_name0
#                ...
#            -model_name1
#                ...
#         -config
#             model1_config
#                -- nmt_small.yml
#        -predict
#            -model_name
#                - prediction.steps0.txt
#                - predictions.steps1.txt


##every time, watch out task_name, model_name, config_app_name##

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

echo "#########"
echo "TRAIN_STEPS:$TRAIN_STEPS"
echo "BATCH_SIZE:$BATCH_SIZE"
echo "EVAL_EVERY_N_STEPS: $EVAL_EVERY_N_STEPS"
echo "CLEAR_OUTPUT_DIR:$CLEAR_OUTPUT_DIR"
echo "#########"

export PYTHONPATH=${SEQ2SEQ_PROJECT_DIR}:${PYTHONPATH}
echo "PYTHONPATH: ${PYTHONPATH}"

cd ${SEQ2SEQ_PROJECT_DIR}
echo "now in dir:$PWD, begin to train model"

python -m bin.train \
  --config_paths="
      $CONFIG_DIR/nmt_small.yml,
      $CONFIG_DIR/train_seq2seq.yml,
      $CONFIG_DIR/text_metrics_slice_text.yml" \
  $1 \
  $2 \
  $3 \
  $4 \
  --allow_soft_placement=True \
  --gpu_allow_growth=True \
  --cloud=True \
  --schedule="default" \
  --batch_size=$BATCH_SIZE \
  --train_steps=$TRAIN_STEPS \
  --eval_every_n_steps=${EVAL_EVERY_N_STEPS} \
  --output_dir=$MODEL_DIR \
  --clear_output_dir=${CLEAR_OUTPUT_DIR} \
  --save_checkpoints_secs=$SAVE_CHECK_SECS \
  --keep_checkpoint_max=$KEEP_CHECK_MAX \
  --set_eval_node=1

#  --model_params="
#      vocab_source: $VOCAB_SOURCE
#      vocab_target: $VOCAB_TARGET" \
#  --input_pipeline_train="
#  class: ParallelTextInputPipeline
#  params:
#    source_files:
#      - $TRAIN_SOURCES
#    target_files:
#      - $TEST_TARGETS
#   " \
#   --input_pipeline_dev="
#  class: ParallelTextInputPipeline
#  params:
#    source_files:
#      - $DEV_SOURCES
#    target_files:
#      - $DEV_TARGETS
#   " \