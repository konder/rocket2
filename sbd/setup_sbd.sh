#!/bin/bash
set -e

echo "=== Setting up SkillDiscovery SBD ==="

cd "$(dirname "$0")/sbd_lib"

mkdir -p weights models

echo "[1/4] Downloading VPT foundation model..."
if [ ! -f models/foundation-model-3x.model ]; then
    wget -q --show-progress https://openaipublic.blob.core.windows.net/minecraft-rl/models/foundation-model-3x.model -P models
else
    echo "  Already exists, skipping."
fi

echo "[2/4] Downloading VPT BC weights..."
if [ ! -f weights/bc-early-game-3x.weights ]; then
    wget -q --show-progress https://openaipublic.blob.core.windows.net/minecraft-rl/models/bc-early-game-3x.weights -P weights
else
    echo "  Already exists, skipping."
fi

echo "[3/4] Downloading IDM model..."
if [ ! -f models/4x_idm.model ]; then
    wget -q --show-progress https://openaipublic.blob.core.windows.net/minecraft-rl/idm/4x_idm.model -P models
else
    echo "  Already exists, skipping."
fi

echo "[4/4] Downloading IDM weights..."
if [ ! -f weights/4x_idm.weights ]; then
    wget -q --show-progress https://openaipublic.blob.core.windows.net/minecraft-rl/idm/4x_idm.weights -P weights
else
    echo "  Already exists, skipping."
fi

echo ""
echo "=== All model weights downloaded ==="
echo "  VPT model: sbd_lib/models/foundation-model-3x.model"
echo "  VPT weights: sbd_lib/weights/bc-early-game-3x.weights"
echo "  IDM model: sbd_lib/models/4x_idm.model"
echo "  IDM weights: sbd_lib/weights/4x_idm.weights"
echo ""
echo "Next: python eval_sbd_vlm.py all --video-dir /path/to/videos --output-dir eval_gallery/pipeline_results"
