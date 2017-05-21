#! /bin/bash/ env

##changable according to your data dir and task

DATA_ROOT=/mnt/yardcephfs/mmyard/g_wxg_td_prc/turingli/data
SEQ2SEQ_PROJECT_DIR=${PWD}/seq2seq
CONFIG_DIR=${SEQ2SEQ_PROJECT_DIR}/example_configs/yard_ques_gen_10w_config
echo "config dir: ${CONFIG_DIR}"
echo "seq2seq project dir: ${SEQ2SEQ_PROJECT_DIR}"

TASK_NAME=ques_10w
TASK_ROOT=$DATA_ROOT/$TASK_NAME
DATA_DIR=$TASK_ROOT/data
VOCAB_SOURCE=$DATA_DIR/vocab/shared.vocab.txt
VOCAB_SOURCE=$DATA_DIR/vocab/shared.vocab.txt
TRAIN_SOURCES=$DATA_DIR/train/sources.txt
TRAIN_TARGETS=$DATA_DIR/train/targets.txt
DEV_SOURCES=$DATA_DIR/dev/sources.txt
DEV_TARGETS=$DATA_DIR/dev/targets.txt
TEST_SOURCES=$DATA_DIR/test/sources.txt
TEST_TARGETS=$DATA_DIR/test/targets.txt
MODEL_DIR=$TASK_ROOT/model_local

echo "MODEL_DIR: $MODEL_DIR"

rm -rf $MODEL_DIR
mkdir -p $MODEL_DIR

TRAIN_STEPS=500000
BATCH_SIZE=64
EVAL_EVERY_N_STEPS=5000

export PYTHONPATH=${PWD}/pyrouge:${SEQ2SEQ_PROJECT_DIR}:$PYTHONPATH
echo "PYTHONPATH: ${PYTHONPATH}"

cd ${SEQ2SEQ_PROJECT_DIR}
echo "now in dir:$PWD, begin to train model"

#####
CUDA_VISIBLE_DEVICES=0
python -m bin.train \
  --ps_hosts="localhost:2222" \
  --worker_hosts="localhost:2223,localhost:2224" \
  --job_name="ps" \
  --task_index=0 \
  --schedule="" \
  --config_paths="
      $CONFIG_DIR/nmt_small.yml,
      $CONFIG_DIR/train_seq2seq.yml,
      $CONFIG_DIR/text_metrics_bpe.yml" \
  --batch_size $BATCH_SIZE \
  --train_steps $TRAIN_STEPS \
  --gpu_memory_fraction=1 \
  --eval_every_n_steps=$EVAL_EVERY_N_STEPS \
  --output_dir $MODEL_DIR  &

CUDA_VISIBLE_DEVICES=1
python -m bin.train \
  --ps_hosts="localhost:2222" \
  --worker_hosts="localhost:2223,localhost:2224" \
  --job_name="worker" \
  --task_index=0 \
  --schedule="" \
  --config_paths="
      $CONFIG_DIR/nmt_small.yml,
      $CONFIG_DIR/train_seq2seq.yml,
      $CONFIG_DIR/text_metrics_bpe.yml" \
  --batch_size $BATCH_SIZE \
  --train_steps $TRAIN_STEPS \
  --eval_every_n_steps=$EVAL_EVERY_N_STEPS \
  --gpu_memory_fraction=1 \
  --output_dir $MODEL_DIR  &

CUDA_VISIBLE_DEVICES=2
python -m bin.train \
  --ps_hosts="localhost:2222" \
  --worker_hosts="localhost:2223,localhost:2224" \
  --job_name="worker" \
  --task_index=1 \
  --schedule="" \
  --config_paths="
      $CONFIG_DIR/nmt_small.yml,
      $CONFIG_DIR/train_seq2seq.yml,
      $CONFIG_DIR/text_metrics_bpe.yml" \
  --batch_size $BATCH_SIZE \
  --train_steps $TRAIN_STEPS \
  --eval_every_n_steps=$EVAL_EVERY_N_STEPS \
  --gpu_memory_fraction=1 \
  --output_dir $MODEL_DIR  &