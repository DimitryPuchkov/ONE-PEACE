#!/usr/bin/env bash

# Activate conda environment
source /home/dmitry/miniconda3/bin/activate one_peace

# Add fairseq to PYTHONPATH
export PYTHONPATH=/home/dmitry/ONE-PEACE-VIDEO-MODALITY/fairseq:$PYTHONPATH

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=6082
export CUDA_VISIBLE_DEVICES=0
export GPUS_PER_NODE=1

config_dir=./
config_name=pretrain_vl_3B
save_dir=../../checkpoints/pretrain/one_peace_vl_3B

torchrun --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../../train.py \
    --config-dir=${config_dir} \
    --config-name=${config_name} \
    checkpoint.save_dir=${save_dir}