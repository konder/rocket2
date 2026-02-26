# ROCKET-2 Benchmark 部署指南 — Linux GPU 服务器

> 目标：在 Linux GPU 服务器（RTX 5090 / A100 / A800 等）上部署 ROCKET-2 完整评估管线
>
> 组件：MineStudio 模拟器 + ROCKET-2 策略模型 + Molmo-7B 视觉语言模型 + SAM-2 分割模型
>
> 评估任务：16 个论文基准任务 / 76 个 MineStudio 内置任务

---

## 目录

1. [硬件要求与显存预估](#1-硬件要求与显存预估)
2. [基础环境搭建](#2-基础环境搭建)
3. [项目代码与依赖安装](#3-项目代码与依赖安装)
4. [MineStudio 模拟器安装](#4-minestudio-模拟器安装)
5. [SAM-2 安装与权重下载](#5-sam-2-安装与权重下载)
6. [模型权重下载](#6-模型权重下载)
7. [运行基准评估](#7-运行基准评估)
8. [Docker 部署（可选）](#8-docker-部署可选)
9. [常见问题排查](#9-常见问题排查)

---

## 1. 硬件要求与显存预估

### 最低配置

| 项目 | 要求 |
|---|---|
| GPU | NVIDIA GPU，显存 ≥ 24GB（RTX 4090/5090/A5000 以上） |
| CUDA | 12.0+ |
| CPU | 8+ 核 |
| 内存 | 32GB+ RAM |
| 磁盘 | 100GB+ 可用空间（模型权重 + Minecraft） |
| 操作系统 | Ubuntu 20.04 / 22.04 |
| 显示 | 需要 Xvfb（虚拟帧缓冲），无需物理显示器 |

### 显存分配预估（FP16）

| 组件 | 显存占用 |
|---|---|
| Molmo-7B-D（FP16） | ~14 GB |
| ROCKET-2 策略模型（FP16） | ~1.5 GB |
| SAM-2 Base+（FP16） | ~0.4 GB |
| MineStudio 模拟器 | ~0.5 GB（GPU 渲染） |
| **合计** | **~16.5 GB** |

> RTX 5090（32GB）或 A100（40/80GB）完全足够。RTX 4090（24GB）也可以运行。

---

## 2. 基础环境搭建

### 2.1 Conda 环境

```bash
conda create -n rocket2 python=3.10 -y
conda activate rocket2
```

### 2.2 PyTorch（CUDA 12.x）

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

验证：

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.get_device_name(0)}')"
```

### 2.3 系统依赖

```bash
sudo apt-get update && sudo apt-get install -y \
    openjdk-8-jdk \
    xvfb \
    libglew-dev libosmesa6-dev libgl1-mesa-glx libglfw3 \
    mesa-utils libegl1-mesa libgl1-mesa-dev libglu1-mesa-dev \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    gcc g++ unzip wget git
```

验证 Java：

```bash
java -version
# 应显示 openjdk version "1.8.x"
```

---

## 3. 项目代码与依赖安装

### 3.1 克隆项目

```bash
git clone <your-rocket2-repo-url> rocket2
cd rocket2
```

### 3.2 安装 MineStudio

```bash
pip install minestudio==1.1.2
```

### 3.3 安装项目依赖

```bash
pip install -r requirements.txt
```

### 3.4 安装 Molmo 依赖

```bash
# transformers 需要 4.46.x 版本（Molmo 兼容）
pip install transformers==4.46.3 accelerate sentencepiece
```

> **重要**：Molmo-7B-D 的自定义代码与 transformers >= 4.49 不兼容，必须使用 4.46.x。

---

## 4. MineStudio 模拟器安装

### 4.1 下载 Minecraft 模拟器引擎

```bash
python -m minestudio.simulator.entry -y
```

这会下载 Minecraft 1.16.5 客户端和 MineRL 运行时（约 2GB）。

### 4.2 设置环境变量（可选）

```bash
# 如需指定模拟器安装目录
export MINESTUDIO_DIR="/path/to/minestudio_dir"
```

### 4.3 验证模拟器

```bash
python -c "
from minestudio.simulator import MinecraftSim
env = MinecraftSim(obs_size=(224, 224))
obs, info = env.reset()
print(f'Simulator OK, obs shape: {obs[\"image\"].shape}')
env.close()
"
```

---

## 5. SAM-2 安装与权重下载

### 5.1 安装 SAM-2（从 MineStudio 源码）

```bash
cd MineStudio/minestudio/utils/realtime_sam
pip install --no-build-isolation -e .
cd ../../../../
```

### 5.2 下载 SAM-2 权重

```bash
cd MineStudio/minestudio/utils/realtime_sam/checkpoints
bash download_ckpts.sh
cd ../../../../../
```

验证：

```bash
ls MineStudio/minestudio/utils/realtime_sam/checkpoints/
# 应包含：sam2_hiera_base_plus.pt, sam2_hiera_large.pt 等
```

---

## 6. 模型权重下载

### 6.1 ROCKET-2 模型（自动下载）

ROCKET-2 模型首次运行时会从 HuggingFace 自动下载：

```bash
# 默认模型: phython96/ROCKET-2-1.5x-17w
# 或手动下载:
python -c "
from minestudio.models import load_rocket_policy
model = load_rocket_policy('hf:phython96/ROCKET-2-1.5x-17w')
print('ROCKET-2 loaded')
"
```

### 6.2 Molmo-7B-D 模型（~14GB）

```bash
# 方法 1：使用 huggingface-cli 下载到指定目录
huggingface-cli download allenai/Molmo-7B-D-0924 --local-dir ./Molmo-7B-D-0924

# 方法 2：使用 HF 镜像（国内服务器）
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download allenai/Molmo-7B-D-0924 --local-dir ./Molmo-7B-D-0924
```

### 6.3 timm 预训练权重（ROCKET-2 依赖）

```bash
python model.py
```

---

## 7. 运行基准评估

### 7.1 快速验证（单任务）

```bash
# 使用 Mock Molmo（不需要 Molmo 模型，快速验证管线）
python benchmark/benchmark_eval.py \
    --task-file benchmark/eval_tasks_paper.yaml \
    --molmo-mock \
    --episodes 1 \
    --tasks mine_coal \
    --output-dir benchmark/results/ \
    --max-steps 50 \
    --save-video
```

### 7.2 单任务 Molmo+SAM2 评估

```bash
python benchmark/benchmark_eval.py \
    --task-file benchmark/eval_tasks_paper.yaml \
    --molmo-model ./Molmo-7B-D-0924 \
    --sam-path ./MineStudio/minestudio/utils/realtime_sam/checkpoints \
    --episodes 1 \
    --tasks mine_coal \
    --output-dir benchmark/results/ \
    --max-steps 600 \
    --save-video
```

### 7.3 论文基准全量评估（16 任务 × 32 回合）

```bash
python benchmark/benchmark_eval.py \
    --task-file benchmark/eval_tasks_paper.yaml \
    --molmo-model ./Molmo-7B-D-0924 \
    --sam-path ./MineStudio/minestudio/utils/realtime_sam/checkpoints \
    --episodes 32 \
    --output-dir benchmark/results/ \
    --save-video
```

### 7.4 MineStudio 内置任务全量评估（76 任务）

```bash
python benchmark/benchmark_eval.py \
    --task-file benchmark/eval_tasks_full.yaml \
    --molmo-model ./Molmo-7B-D-0924 \
    --sam-path ./MineStudio/minestudio/utils/realtime_sam/checkpoints \
    --episodes 3 \
    --output-dir benchmark/results_full/ \
    --save-video
```

### 7.5 CLI 参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--task-file` | （必填） | 任务 YAML 文件路径 |
| `--ckpt` | `hf:phython96/ROCKET-2-1.5x-17w` | ROCKET-2 模型路径 |
| `--cfg-coef` | `1.5` | Classifier-Free Guidance 系数 |
| `--episodes` | YAML 中的值 | 每个任务的评估回合数 |
| `--max-steps` | YAML 中的值 | 每回合最大步数 |
| `--tasks` | 全部 | 指定任务名（空格分隔） |
| `--molmo-mock` | `false` | 使用 Mock Molmo（调试用） |
| `--molmo-model` | `allenai/Molmo-7B-D-0924` | Molmo 模型 ID 或本地路径 |
| `--sam-path` | `./MineStudio/.../checkpoints` | SAM-2 权重目录 |
| `--sam-variant` | `base` | SAM-2 变体：large/base/small/tiny |
| `--output-dir` | `benchmark/results/` | 输出目录 |
| `--save-video` | `false` | 保存 MP4 回放视频 |

### 7.6 输出文件

```
benchmark/results/
├── benchmark_20260226_215101.json    # 评估结果 JSON
└── videos/
    ├── mine_coal_ep0.mp4             # 回放视频（包含目标缩略图叠加）
    ├── mine_coal_ep1.mp4
    └── ...
```

结果 JSON 示例：

```json
{
  "summary": {
    "overall": {
      "num_tasks": 16,
      "total_episodes": 512,
      "total_success": 156,
      "success_rate": 0.305
    },
    "by_interaction_type": {
      "mine": {"num_tasks": 4, "success_rate": 0.45},
      "hunt": {"num_tasks": 3, "success_rate": 0.28},
      "approach": {"num_tasks": 3, "success_rate": 0.15},
      "harvest": {"num_tasks": 3, "success_rate": 0.35},
      "craft": {"num_tasks": 2, "success_rate": 0.10},
      "combat": {"num_tasks": 1, "success_rate": 0.22}
    }
  }
}
```

---

## 8. Docker 部署（可选）

如果使用 Docker，可以基于项目已有的 Dockerfile：

```bash
# 构建镜像
docker build -t rocket2-benchmark -f Dockerfile .

# 运行评估（需要 NVIDIA Docker Runtime）
docker run --gpus all \
    -v $(pwd)/benchmark/results:/app/ROCKET-2/benchmark/results \
    -v /path/to/Molmo-7B-D-0924:/app/Molmo-7B-D-0924 \
    rocket2-benchmark \
    python benchmark/benchmark_eval.py \
        --task-file benchmark/eval_tasks_paper.yaml \
        --molmo-model /app/Molmo-7B-D-0924 \
        --episodes 32 \
        --output-dir benchmark/results/ \
        --save-video
```

---

## 9. 常见问题排查

### Q1: `xvfb-run: command not found`

```bash
sudo apt-get install -y xvfb
```

MineStudio 在 Linux 上需要虚拟帧缓冲来运行 Minecraft 渲染。

### Q2: `java: command not found` 或版本不对

```bash
sudo apt-get install -y openjdk-8-jdk
# 确认版本
java -version  # 应显示 1.8.x
```

### Q3: `CUDA out of memory`

- 减少 `--episodes`（如 `--episodes 1`）
- 使用较小的 SAM-2 变体：`--sam-variant tiny`
- 确认没有其他进程占用 GPU：`nvidia-smi`

### Q4: Molmo 加载失败 `AttributeError: all_tied_weights_keys`

transformers 版本不兼容，需降级：

```bash
pip install transformers==4.46.3
rm -rf ~/.cache/huggingface/modules/transformers_modules/Molmo*
```

### Q5: `can't set up init inventory`

这是 MineStudio 的已知日志，不影响评估。初始物品会在后续步骤正确设置。

### Q6: HuggingFace 下载超时

设置镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### Q7: 如何只在 macOS 上调试代码逻辑？

使用 Mock 模式，不需要 Molmo 模型：

```bash
python benchmark/benchmark_eval.py \
    --task-file benchmark/eval_tasks_paper.yaml \
    --molmo-mock \
    --episodes 1 \
    --tasks mine_coal \
    --output-dir benchmark/results/ \
    --max-steps 50
```

### Q8: `ModuleNotFoundError: No module named 'sam2'`

SAM-2 未安装，执行：

```bash
cd MineStudio/minestudio/utils/realtime_sam
pip install --no-build-isolation -e .
```

---

## 附录：评估时间预估

基于 RTX 5090（32GB）的参考时间：

| 评估规模 | 任务数 | 回合数 | 预估时间 |
|---|---|---|---|
| 快速验证 | 1 | 1 | ~2 分钟 |
| 单任务 | 1 | 32 | ~30 分钟 |
| 论文基准 | 16 | 32 (×16) | ~8 小时 |
| 全量任务 | 76 | 3 (×76) | ~12 小时 |

> 时间包含环境启动、Molmo 推理、ROCKET-2 推理和环境步进。
> 瓶颈通常在 Minecraft 环境步进（~20 FPS）和 Molmo 首帧推理（~3-5 秒/次）。
