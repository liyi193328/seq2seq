#! /bin/bash/ env

export DATA_DIR="/home/bigdata/active_project/test_seq2seq_py2/yard_seq2seq/data/q2q_12w_cancel_dup"
export CUDA_VISIBLE_DEVICES=""  python train.py --config_path="../example_configs/q2q_12w/nmt_small.yml, ../example_configs/q2q_12w/train_seq2seq.yml, ../example_configs/q2q_12w/text_metrics_bpe.yml" --ps_hosts="localhost:2222" --worker_hosts="localhost:2223,localhost:2224,localhost:2225" --job_name="ps" --task_index=0 --cloud=True --schedule="default" --output_dir="${DATA_DIR}/model/test0" --gpu_memory_fraction=1 --eval_every_n_steps=8000 --train_steps=500000 --batch_size=64 --clear_output_dir=False > ${DATA_DIR}/ps.log 2>&1 &
export CUDA_VISIBLE_DEVICES="0" python train.py --config_path="../example_configs/q2q_12w/nmt_small.yml, ../example_configs/q2q_12w/train_seq2seq.yml, ../example_configs/q2q_12w/text_metrics_bpe.yml" --ps_hosts="localhost:2222" --worker_hosts="localhost:2223,localhost:2224,localhost:2225" --job_name="worker" --task_index=0 --cloud=True --schedule="train" --output_dir="${DATA_DIR}/model/test0" --gpu_memory_fraction=0.5 --eval_every_n_steps=8000 --train_steps=500000 --batch_size=64 --clear_output_dir=False > ${DATA_DIR}/worker0.log 2>&1 &
export CUDA_VISIBLE_DEVICES="0" python train.py --config_path="../example_configs/q2q_12w/nmt_small.yml, ../example_configs/q2q_12w/train_seq2seq.yml, ../example_configs/q2q_12w/text_metrics_bpe.yml" --ps_hosts="localhost:2222" --worker_hosts="localhost:2223,localhost:2224,localhost:2225" --job_name="worker" --task_index=1 --cloud=True --schedule="train" --output_dir="${DATA_DIR}/model/test0" --gpu_memory_fraction=0.5 --eval_every_n_steps=8000 --train_steps=500000 --batch_size=64 --clear_output_dir=False > ${DATA_DIR}/worker1.log 2>&1 &
export CUDA_VISIBLE_DEVICES=""  python train.py --config_path="../example_configs/q2q_12w/nmt_small.yml, ../example_configs/q2q_12w/train_seq2seq.yml, ../example_configs/q2q_12w/text_metrics_bpe.yml" --ps_hosts="localhost:2222" --worker_hosts="localhost:2223,localhost:2224,localhost:2225" --job_name="worker" --task_index=2 --cloud=True --schedule="continuous_eval" --output_dir="${DATA_DIR}/model/test0" --gpu_memory_fraction=1 --eval_every_n_steps=8000 --train_steps=500000 --batch_size=64 --clear_output_dir=False > ${DATA_DIR}/worker2.log 2>&1 &