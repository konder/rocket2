#!/bin/zsh

# 以交互式 zsh 运行，加载 ~/.zshrc 并激活 conda 环境

set -e

echo "✅ 环境变量已通过 zsh -i 加载"
echo "   MINESTUDIO_DIR: $MINESTUDIO_DIR"
echo "   http_proxy: $http_proxy"
echo "   HF_ENDPOINT: $HF_ENDPOINT"
echo ""

# 显式激活 conda 环境
echo "🔄 激活 conda 环境: rocket2-arm64"
eval "$(/Users/nanzhang/miniforge3/bin/conda shell.zsh hook)"
conda activate /Users/nanzhang/miniforge3/envs/rocket2-arm64

echo "✅ Conda 环境已激活"
echo ""

# 进入项目目录，准备输出
cd ~/rocket2
mkdir -p grounding_data_local_v1 logs

echo "✅ 项目目录准备完成"
echo ""

# 运行数据收集脚本
echo "🚀 开始运行数据收集..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python train/collect_grounding_data.py \
    --task-file eval/eval_tasks_paper.yaml \
    --output-dir ./grounding_data_local_v1 \
    --steps-per-task 10 \
    --max-samples 100

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ 脚本执行完成"
