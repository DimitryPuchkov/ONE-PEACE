#!/usr/bin/env bash
# Debug run of RU adaptation: 1 epoch, batch 2, on the local GPU (RTX 3090 Ti).
# Usage: bash debug_vl_ru.sh [log_file]

source /home/dmitry/miniconda3/bin/activate one_peace

export PYTHONPATH=/home/dmitry/ONE-PEACE-VIDEO-MODALITY/fairseq:$PYTHONPATH
export MASTER_PORT=6199
export CUDA_VISIBLE_DEVICES=0
export GPUS_PER_NODE=1

config_dir=./
config_name=debug_vl_ru
save_dir=../../checkpoints/pretrain/debug_vl_ru
log_file=${1:-debug_run.log}

mkdir -p ${save_dir}

torchrun --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../../train.py \
    --config-dir=${config_dir} \
    --config-name=${config_name} \
    checkpoint.save_dir=${save_dir} 2>&1 | tee ${log_file}
