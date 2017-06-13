#! /bin/bash/ env

SEQ2SEQ_PROJECT_DIR=${SEQ2SEQ_PROJECT_DIR:=/home/bigdata/active_project/seq2seq}
echo "seq2seq_project_dir:$SEQ2SEQ_PROJECT_DIR"

#TASK_NAME=ques_10w
TASK_NAME=${TASK_NAME:=ques_gen_all}
MODEL_NAME=${MODEL_NAME:=model0}

DATA_ROOT=${DATA_ROOT:=/home/bigdata/active_project/test_seq2seq_py2/yard_seq2seq}
DEFAULT_MODEL_ROOT=$DATA_ROOT/${TASK_NAME}/model
MODEL_DIR_ROOT=${MODEL_DIR_ROOT:=$DEFAULT_MODEL_ROOT}
MAYBE_MODEL_DIR=${MODEL_DIR_ROOT}/${MODEL_NAME}
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
VOCAB_TARGET=$DATA_DIR/vocab/shared.vocab.txt
TRAIN_SOURCES=$DATA_DIR/train/sources.txt
TRAIN_TARGETS=$DATA_DIR/train/targets.txt
DEV_SOURCES=$DATA_DIR/dev/sources.txt
DEV_TARGETS=$DATA_DIR/dev/targets.txt
TEST_SOURCES=$DATA_DIR/test/sources.txt
TEST_TARGETS=$DATA_DIR/test/targets.txt


TRAIN_STEPS=${TRAIN_STEPS:=5000000}
BATCH_SIZE=${BATCH_SIZE:=64}
EVAL_EVERY_N_STEPS=${EVAL_EVERY_N_STEPS:=5000}
SAVE_CHECK_SECS=${SAVE_CHECK_SECS:=1800}
KEEP_CHECK_MAX=${KEEP_CHECK_MAX:=20}
echo "#########"
echo "TRAIN_STEPS:$TRAIN_STEPS"
echo "BATCH_SIZE:$BATCH_SIZE"
echo "EVAL_EVERY_N_STEPS: $EVAL_EVERY_N_STEPS"
echo "#########"

LOG_DIR=${TASK_ROOT}/log
mkdir -p ${LOG_DIR}

export CUDA_VISIBLE_DEVICES="";
python $SEQ2SEQ_PROJECT_DIR/bin/train.py \
--config_path="$CONFIG_DIR/nmt_small.yml, $CONFIG_DIR/train_seq2seq.yml, $CONFIG_DIR/text_metrics_bpe.yml" \
--model_params="
      vocab_source: $VOCAB_SOURCE
      vocab_target: $VOCAB_TARGET" \
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
--config_path="$CONFIG_DIR/nmt_small.yml, $CONFIG_DIR/train_seq2seq.yml, $CONFIG_DIR/text_metrics_bpe.yml" \
--ps_hosts="localhost:2222" --worker_hosts="localhost:2223,localhost:2224,localhost:2225" --job_name="ps" --task_index=0 --cloud=True --schedule="default" \
--output_dir="${MODEL_DIR}" --gpu_memory_fraction=1 --eval_every_n_steps=${EVAL_EVERY_N_STEPS} \
--train_steps=$TRAIN_STEPS --batch_size=$BATCH_SIZE --save_checkpoints_secs=1200 \
--keep_checkpoint_max=$KEEP_CHECK_MAX clear_output_dir=False \
> ${LOG_DIR}/ps_${MODEL_NAME}.log 2>&1 &

export CUDA_VISIBLE_DEVICES="0";
python $SEQ2SEQ_PROJECT_DIR/bin/train.py \
--config_path="$CONFIG_DIR/nmt_small.yml, $CONFIG_DIR/train_seq2seq.yml, $CONFIG_DIR/text_metrics_bpe.yml" \
--model_params="
      vocab_source: $VOCAB_SOURCE
      vocab_target: $VOCAB_TARGET" \
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
--ps_hosts="localhost:2222" --worker_hosts="localhost:2223,localhost:2224,localhost:2225" --job_name="worker" --task_index=0 \
--cloud=True --schedule="train" \
--output_dir="${MODEL_DIR}" --gpu_memory_fraction=0.5 --eval_every_n_steps=${EVAL_EVERY_N_STEPS} \
--train_steps=$TRAIN_STEPS --batch_size=$BATCH_SIZE --save_checkpoints_secs=$SAVE_CHECK_SECS \
--keep_checkpoint_max=$KEEP_CHECK_MAX --clear_output_dir=False \
> ${LOG_DIR}/worker0_${MODEL_NAME}.log 2>&1 &

export CUDA_VISIBLE_DEVICES="0";
python $SEQ2SEQ_PROJECT_DIR/bin/train.py \
--model_params="
      vocab_source: $VOCAB_SOURCE
      vocab_target: $VOCAB_TARGET" \
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
--config_path="$CONFIG_DIR/nmt_small.yml, $CONFIG_DIR/train_seq2seq.yml, $CONFIG_DIR/text_metrics_bpe.yml" \
--ps_hosts="localhost:2222" --worker_hosts="localhost:2223,localhost:2224,localhost:2225" --job_name="worker" \
--task_index=1 --cloud=True --schedule="train" \
--output_dir="${MODEL_DIR}" --gpu_memory_fraction=0.5 --eval_every_n_steps=${EVAL_EVERY_N_STEPS} \
--train_steps=$TRAIN_STEPS --batch_size=$BATCH_SIZE --save_checkpoints_secs=$SAVE_CHECK_SECS \
--keep_checkpoint_max=$KEEP_CHECK_MAX --clear_output_dir=False \
> ${LOG_DIR}/worker1_${MODEL_NAME}.log 2>&1 &

export CUDA_VISIBLE_DEVICES="";
python $SEQ2SEQ_PROJECT_DIR/bin/train.py \
    --model_params="
          vocab_source: $VOCAB_SOURCE
          vocab_target: $VOCAB_TARGET" \
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
    --config_path="$CONFIG_DIR/nmt_small.yml, $CONFIG_DIR/train_seq2seq.yml, $CONFIG_DIR/text_metrics_bpe.yml" \
    --ps_hosts="localhost:2222" --worker_hosts="localhost:2223,localhost:2224,localhost:2225" --job_name="worker" \
    --task_index=2 --cloud=True --schedule="continuous_eval" \
    --output_dir="${MODEL_DIR}" --gpu_memory_fraction=1 --eval_every_n_steps=${EVAL_EVERY_N_STEPS} \
    --train_steps=$TRAIN_STEPS --batch_size=$BATCH_SIZE --save_checkpoints_secs=$SAVE_CHECK_SECS \
    --keep_checkpoint_max=$KEEP_CHECK_MAX --clear_output_dir=False \
    > ${LOG_DIR}/worker2_${MODEL_NAME}.log 2>&1 &