#! /bin/bash/ env

##changable according to your data dir and task
TASK_NAME=ques_50w
# DATA_ROOT=/mnt/yardcephfs/mmyard/g_wxg_td_prc/turingli
DATA_ROOT=/home/bigdata/active_project/test_seq2seq_py2/yard_seq2seq
SEQ2SEQ_PROJECT_PATH=/home/bigdata/active_project/seq2seq
RUN_PATH=$PWD

PREFIX=$DATA_ROOT/$TASK_NAME
DATA_DIR=$PREFIX/ques_50w_data
CONFIG_DIR=$PREFIX/ques_50w_config
VOCAB_SOURCE=$DATA_DIR/vocab/shared.vocab.txt
VOCAB_TARGET=$DATA_DIR/vocab/shared.vocab.txt
TRAIN_SOURCES=$DATA_DIR/train/sources.txt
TRAIN_TARGETS=$DATA_DIR/train/targets.txt
DEV_SOURCES=$DATA_DIR/dev/sources.txt
DEV_TARGETS=$DATA_DIR/dev/targets.txt
TEST_SOURCES=$DATA_DIR/test/sources.txt
TEST_TARGETS=$DATA_DIR/test/targets.txt
MODEL_DIR=$PREFIX/model
LOG_DIR=$PREFIX/log
mkdir -p $LOG_DIR

echo "MODEL_DIR: $MODEL_DIR"

rm -rf $MODEL_DIR
mkdir -p $MODEL_DIR

TRAIN_STEPS=500000
BATCH_SIZE=128


export PYTHONPATH=${PWD}/pyrouge:${SEQ2SEQ_PROJECT_PATH}:$PYTHONPATH
echo "PYTHONPATH: ${PYTHONPATH}"

cd $SEQ2SEQ_PROJECT_PATH
echo "now in $SEQ2SEQ_PROJECT_PATH"
#####chy pc test
#cd  ~/active_project/seq2seq/
#python ./bin/train.py --config_path="../example_configs/nmt_small.yml, ../example_configs/train_seq2seq.yml, ../example_configs/text_metrics_bpe.yml" --ps_hosts="localhost:2222" --worker_hosts="localhost:2223" --job_name="worker" --task_index=0 --output_dir="/home/bigdata/active_project/test_seq2seq_py2/yard_seq2seq/ques_50w/model"

#####

python -m bin.train \
  --ps_hosts="localhost:2222" \
  --worker_hosts="localhost:2223" \
  --job_name="ps" \
  --task_index=0 \
  --schedule="" \
  --config_paths="
      $CONFIG_DIR/nmt_small.yml,
      $CONFIG_DIR/train_seq2seq.yml,
      $CONFIG_DIR/text_metrics_bpe.yml" \
  --model_params "
      vocab_source: $VOCAB_SOURCE
      vocab_target: $VOCAB_TARGET" \
  --input_pipeline_train "
    class: ParallelTextInputPipeline
    params:
      source_files:
        - $TRAIN_SOURCES
      target_files:
        - $TRAIN_TARGETS" \
  --input_pipeline_dev "
    class: ParallelTextInputPipeline
    params:
       source_files:
        - $DEV_SOURCES
       target_files:
        - $DEV_TARGETS" \
  --batch_size $BATCH_SIZE \
  --train_steps $TRAIN_STEPS \
  --gpu_memory_fraction=1 \
  --eval_every_n_steps=40000 \
  --output_dir $MODEL_DIR  &

python -m bin.train \
  --ps_hosts="localhost:2222" \
  --worker_hosts="localhost:2223" \
  --job_name="worker" \
  --task_index=0 \
  --schedule="" \
  --config_paths="
      $CONFIG_DIR/nmt_small.yml,
      $CONFIG_DIR/train_seq2seq.yml,
      $CONFIG_DIR/text_metrics_bpe.yml" \
  --model_params "
      vocab_source: $VOCAB_SOURCE
      vocab_target: $VOCAB_TARGET" \
  --input_pipeline_train "
    class: ParallelTextInputPipeline
    params:
      source_files:
        - $TRAIN_SOURCES
      target_files:
        - $TRAIN_TARGETS" \
  --input_pipeline_dev "
    class: ParallelTextInputPipeline
    params:
       source_files:
        - $DEV_SOURCES
       target_files:
        - $DEV_TARGETS" \
  --batch_size $BATCH_SIZE \
  --train_steps $TRAIN_STEPS \
  --eval_every_n_steps=40000 \
  --gpu_memory_fraction=0.6 \
  --output_dir $MODEL_DIR  &
