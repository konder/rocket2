<!--
 * @Date: 2025-03-21 10:25:54
 * @LastEditors: caishaofei-mus1 1744260356@qq.com
 * @LastEditTime: 2025-03-21 10:32:18
 * @FilePath: /ROCKET2-OSS/README.md
-->
<h2 align="center"> ROCKET-2: Steering Visuomotor Policy via Cross-View Goal Alignment </h2>

<div align="center">

[`Shaofei Cai`](https://phython96.github.io/) | [`Zhancun Mu`](https://zhancunmu.owlstown.net/) | [`Anji Liu`](https://liuanji.github.io/) | [`Yitao Liang`](https://scholar.google.com/citations?user=KVzR1XEAAAAJ&hl=zh-CN&oi=ao)

All authors are affiliated with Team **[`CraftJarvis`](https://craftjarvis.github.io/)**. 

[![Project](https://img.shields.io/badge/Project-ROCKET--2-blue)](https://craftjarvis.github.io/ROCKET-2/)
[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2503.02505-red)](https://arxiv.org/abs/2503.02505)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Paper-yellow)](https://huggingface.co/papers/2503.02505)
[![BibTeX](https://img.shields.io/badge/BibTeX-Citation-blueviolet)](#citig_rocket)
[![OpenBayes Demo](https://img.shields.io/static/v1?label=Demo&message=OpenBayes%E8%B4%9D%E5%BC%8F%E8%AE%A1%E7%AE%97&color=green)](https://openbayes.com/console/public/tutorials/cHZ3teJXaQU)

<p align="center">
  <img src="images/teaser2.png" width="100%" />
</p>

<b> ROCKET-2 successfully inflicted damage on the Ender Dragon for the first time and spontaneously developed the ability to build bridges. (end-to-end training, directly control mouse & keyboard)</b>

<p align="center">
  <img src="images/comp_4x4_grid.gif" alt="Full Width GIF" width="100%" />
</p>

<b> ROCKET-2 was pre-trained only on Minecraft, yet generalizes zero-shot to other 3D games. </b>

<p align="center">
  <img src="images/output_grid.gif" alt="Full Width GIF" width="100%" />
</p>
</div>

## Latest updates
- **03/21/2025 -- We have released the codebase for ROCKET-2!**

## Docker

**Standard build (with GPU):**
```sh
docker build -t rocket2 .
docker run -it -p 7860:7860 --platform=linux/amd64 --gpus all rocket2:latest
```

**Dev build (Apple Silicon / CPU):**
```sh
# Build dev image
docker build --platform linux/amd64 -f Dockerfile.dev -t rocket2-dev .

# Run with local source mounted for development
docker run -it --rm -p 7860:7860 --platform linux/amd64 \
  -v $(pwd):/app/ROCKET-2 \
  rocket2-dev
```

After launching, open http://127.0.0.1:7860 in your browser.

## Usage Guide

![](images/demo.png)

The Gradio interface has 5 tabs: **Tutorial**, **Customize Environment**, **Launch Rocket**, **Specify Goal**, and **Record Video**. Follow the steps below to run ROCKET-2:

### Step 1: Customize Environment

1. Switch to the **Customize Environment** tab.
2. A YAML editor is pre-filled with default config (spawn position, initial inventory, etc.).
3. You can also pick a preset config from the dropdown and click **Load**.
4. **Important: Click the "Set" button to apply the config.** Without this, the environment will not be configured.

### Step 2: Reset Environment & Load Model

1. Switch to the **Launch Rocket** tab.
2. In the **Setting Panel** sub-tab, select a model checkpoint:
   - `hf:phython96/ROCKET-2-1.5x-17w` (recommended)
   - `hf:phython96/ROCKET-2-1x-22w`
3. Optionally adjust the **Classifier-Free** guidance coefficient (default: 1.5).
4. Click **Reset** — this loads the model and initializes the Minecraft environment. Wait for the game screen to appear.

### Step 3: Specify Goal

1. Switch to the **Specify Goal** tab.
2. Select a cross-view image as the goal target. Three methods available:
   - **Upload**: Upload an image directly.
   - **Gallery**: Click a preset image from the gallery at the bottom.
   - **History Observations**: Use the slider to pick from past observations.
3. Click on the image to add SAM-2 point prompts:
   - **Pos** (positive): Mark the target object.
   - **Neg** (negative): Mark areas to exclude.
4. Select the **Interaction Type**: `Approach`, `Hunt`, `Mine`, `Interact`, `Craft`, or `None`.
5. Click **Segment** — SAM-2 will segment the target object and show a colored mask overlay.

### Step 4: Launch ROCKET-2

1. Go back to the **Launch Rocket** tab.
2. In the **Control Panel**, set the number of **Steps** (default: 30).
3. Click **Go** — ROCKET-2 will autonomously control the agent to achieve the goal.
4. The display shows:
   - The current game view with the cross-view goal overlaid (top-right corner).
   - A **Visibility** indicator showing how visible the target is.
   - Current step count and memory length.

### Step 5: Record Video (Optional)

1. Switch to the **Record Video** tab.
2. Click **Make Video** to generate a replay video.
3. Click **Download** to save the video file.

### Tips

| Action | Description |
|---|---|
| **Clear Memory** | If ROCKET-2 gets stuck, click this to reset the agent's memory. |
| **Update goal frequently** | Smaller temporal gap between current view and goal image leads to better performance. |
| **Commands Panel** | Send Minecraft cheat commands (e.g., `/setblock ~0 ~0 ~4 minecraft:diamond_block`). |
| **SAM-2 model size** | Choose from large/base/small/tiny. Base is the default balanced option. |

### Troubleshooting

- **SSL / download errors on Apple Silicon**: The dev Docker image runs under x86 emulation, which can cause intermittent SSL failures when downloading model weights. The code includes automatic retry logic. If it persists, rebuild the Docker image (which pre-downloads all weights) or set `HF_ENDPOINT=https://hf-mirror.com` for a HuggingFace mirror.
- **"Please set environment config first"**: You must click **Set** in the Customize Environment tab before clicking Reset.
- **No GPU detected**: If running without GPU, the dev image defaults to CPU mode (`ROCKET_DEVICE=cpu`). Performance will be slower but functional.

## Citing ROCKET-2
If you use ROCKET-2 in your research, please use the following BibTeX entry. 

```
@article{cai2025rocket,
  title={ROCKET-2: Steering Visuomotor Policy via Cross-View Goal Alignment},
  author={Cai, Shaofei and Mu, Zhancun and Liu, Anji and Liang, Yitao},
  journal={arXiv preprint arXiv:2503.02505},
  year={2025}
}
```
