#!/bin/bash

# 直接设置环境变量
export MINESTUDIO_DIR="$HOME/.minestudio"
export HF_ENDPOINT="https://hf-mirror.com"
export http_proxy="http://192.168.101.118:3128"
export https_proxy="http://192.168.101.118:3128"

# 验证环境变量
echo "Environment variables set:"
echo "  MINESTUDIO_DIR=$MINESTUDIO_DIR"
echo "  HF_ENDPOINT=$HF_ENDPOINT"
echo "  http_proxy=$http_proxy"
echo "  https_proxy=$https_proxy"
echo ""

# 激活 conda
source /Users/nanzhang/miniforge3/etc/profile.d/conda.sh
conda activate /Users/nanzhang/miniforge3/envs/rocket2-arm64

# 进入项目目录
cd ~/rocket2

# 运行脚本
python train/collect_grounding_data.py \
    --task-file eval/eval_tasks_paper.yaml \
    --output-dir ./grounding_data_local_v1 \
    --steps-per-task 10 \
    --max-samples 100

