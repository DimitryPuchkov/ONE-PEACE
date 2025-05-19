#!/bin/bash
#SBATCH --job-name=ptrain_32_256_precomputed_part5
#SBATCH --error=/userspace/tfv/logs/train_video_adapter_32_256_precomputed_part5.err
#SBATCH --output=/userspace/tfv/logs/train_video_adapter_32_256_precomputed_part5.log
#SBATCH --partition=a100
#SBATCH --cpus-per-task=16
#SBATCH --nodelist=ngpu09

# Define log directory and base name (CHANGE THESE TO UPDATE LOG PATHS)
LOG_DIR="/userspace/tfv/logs"
LOG_NAME="train_video_adapter_32_256_precomputed_part5"
LOG_FILE="${LOG_DIR}/${LOG_NAME}.log"
ERR_FILE="${LOG_DIR}/${LOG_NAME}.err"

# Log start time
echo "Script started at $(date)" >> "$LOG_FILE"
echo "Script started at $(date)" >> "$ERR_FILE"

# Activate conda environment
source /userspace/tfv/miniconda3/etc/profile.d/conda.sh
echo "Loading conda environment" >> "$LOG_FILE"
conda activate dip_test

# Set Conda cache directories to writable locations
export CONDA_PKGS_DIRS=/userspace/tfv/cache/conda_pkgs
export CONDA_CACHE_DIR=/userspace/tfv/cache

# Set environment variables for CUDA and others
export CUDA_HOME=$CONDA_PREFIX
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

export LD_LIBRARY_PATH=/userspace/tfv/nccl/build/lib:$LD_LIBRARY_PATH

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9
export MASTER_PORT=29544
export GPUS_PER_NODE=10

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_DEBUG=ERROR

export PIP_CACHE_DIR=/userspace/tfv/cache
#export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256,garbage_collection_threshold:0.6"
export HF_HOME=/userspace/tfv/cache/huggingface
export OUTLINES_CACHE_DIR=/userspace/tfv/cache/outlines
export NLTK_DATA=/userspace/tfv/cache/nltk_data
export MPLCONFIGDIR=/userspace/tfv/cache/matplotlib
export FASTEMBED_CACHE_PATH=/userspace/tfv/cache/fastembed
export PATH=/userspace/dra/ffmpeg-release:$PATH
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export DEEPSPEED_AUTOTUNE_CACHE=/userspace/tfv/cache/deepspeed
export DEEPSPEED_CACHE_DIR=/userspace/tfv/cache/deepspeed
export TRITON_CACHE_DIR=/userspace/tfv/cache/deepspeed
export CUDA_LAUNCH_BLOCKING=0

echo "Environment variables set" >> "$LOG_FILE"

# Print CUDA details
echo "Printing CUDA details" >> "$LOG_FILE"
ls -d /usr/local/cuda* >> "$LOG_FILE"
nvidia-smi -L >> "$LOG_FILE"
nvcc -V >> "$LOG_FILE"
python -V >> "$LOG_FILE"

#kill -9 4053237 4053238 4053239 4053240

nvidia-smi --query-compute-apps=pid,name,gpu_bus_id --format=csv

echo "GPU status before code execution:" >> "$LOG_FILE"
nvidia-smi >> "$LOG_FILE"


# Define configuration paths
config_dir=.
config_name=pretrain_video_3B_new
save_dir=../../checkpoints/pretrain/one_peace_video_32_256_precomputed
restore_file=/userspace/tfv/diplom/ONE-PEACE-VIDEO-MODALITY/one_peace/checkpoints/pretrain/one_peace_video_32_256_precomputed/checkpoint_last.pt

# Launch training using DeepSpeed
torchrun --nproc_per_node=${GPUS_PER_NODE} --nnodes=${SLURM_NNODES} --master_port=${MASTER_PORT} --rdzv_backend=c10d  /userspace/tfv/diplom/ONE-PEACE-VIDEO-MODALITY/one_peace/train.py \
     --config-dir=${config_dir} \
     --config-name=${config_name} \
     checkpoint.save_dir=${save_dir} \
     checkpoint.restore_file=${restore_file}

echo "GPU status after code execution:" >> "$LOG_FILE"
nvidia-smi >> "$LOG_FILE"

echo "Script ended at $(date)" >> "$LOG_FILE"

