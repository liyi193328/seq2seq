#! /bin/bash/ env

#data organized:
#data_root/task_name:
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

SEQ2SEQ_PROJECT_DIR=${PWD}
#TASK_NAME=ques_10w
TASK_NAME=${TASK_NAME:=ques_gen_all}
MODEL_NAME=${MODEL_NAME:=model0}

DATA_ROOT=${DATA_ROOT:=/mnt/yardcephfs/mmyard/g_wxg_td_prc/turingli}
DEFAULT_MODEL_ROOT=$DATA_ROOT/${TASK_NAME}/model
MODEL_DIR_ROOT=${MODEL_DIR_ROOT:=$DEFAULT_MODEL_ROOT}
MAYBE_MODEL_DIR={MODEL_DIR_ROOT}/${MODEL_NAME}
MODEL_DIR=${MODEL_DIR:=$MAYBE_MODEL_DIR}
mkdir -p ${MODEL_DIR}
echo "MODEL_DIR: $MODEL_DIR"

#yard_ques_gen_10w_config
CONFIG_APP_NAME=${CONFIG_APP_NAME:=yard_ques_gen_10w_config}
MAYBE_CONFIG_DIR=${SEQ2SEQ_PROJECT_DIR}/example_configs/${CONFIG_APP_NAME}
CONFIG_DIR=${CONFIG_DIR:=$MAYBE_CONFIG_DIR}

echo "TASK_NAME=${TASK_NAME} must contain in config_dir and data_dir"
echo "config dir: ${CONFIG_DIR}"
echo "seq2seq project dir: ${SEQ2SEQ_PROJECT_DIR}"

TASK_ROOT=$DATA_ROOT/$TASK_NAME
DATA_DIR=$TASK_ROOT/data
echo "DATA_DIR: ${DATA_DIR}"

CLEAR_OUTPUT_DIR=${CLEAR_OUTPUT_DIR:=False}

VOCAB_SOURCE=$DATA_DIR/vocab/shared.vocab.txt
VOCAB_TAEGET=$DATA_DIR/vocab/shared.vocab.txt
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
echo "#########"

export PYTHONPATH=${SEQ2SEQ_PROJECT_DIR}:${PYTHONPATH}
echo "PYTHONPATH: ${PYTHONPATH}"

cd ${SEQ2SEQ_PROJECT_DIR}
echo "now in dir:$PWD, begin to train model"

python -m bin.train \
  --config_paths="
      $CONFIG_DIR/nmt_small.yml,
      $CONFIG_DIR/train_seq2seq.yml,
      $CONFIG_DIR/text_metrics_bpe.yml" \
  --input_pipeline_train="
  class: ParallelTextInputPipeline
  params:
    source_files:
      - $TRAIN_SOURCES
    target_files:
      - $TEST_TARGETS
   " \
   --input_pipeline_dev="
  class: ParallelTextInputPipeline
  params:
    source_files:
      - $DEV_SOURCES
    target_files:
      - $DEV_TARGETS
   " \
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


