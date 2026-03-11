#!/bin/bash

# xiami-2 任务 #1 执行脚本
# 设置环境变量 + 运行数据收集

set -e

# ============================================================
# 1. 环境变量配置
# ============================================================
export MINESTUDIO_DIR="$HOME/.minestudio"
# 注：移除代理设置 - launchd 沙箱环境无法访问内部代理
# 改为使用直接连接

echo "✅ 环境变量已设置"
echo "   MINESTUDIO_DIR: $MINESTUDIO_DIR"
echo ""

# ============================================================
# 2. 激活 Conda 环境
# ============================================================
source /Users/nanzhang/miniforge3/etc/profile.d/conda.sh
conda activate /Users/nanzhang/miniforge3/envs/rocket2-arm64

echo "✅ Conda 环境已激活: rocket2-arm64"
echo ""

# ============================================================
# 3. 进入项目目录，准备输出
# ============================================================
cd ~/rocket2
mkdir -p grounding_data_local_v1 logs

echo "✅ 项目目录准备完成"
echo ""

# ============================================================
# 4. 运行数据收集脚本
# ============================================================
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

