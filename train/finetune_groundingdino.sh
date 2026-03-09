#!/bin/bash
# Fine-tune GroundingDINO on Minecraft block data.
#
# Prerequisites:
#   1. Collect data first:
#      python benchmark/tools/collect_grounding_data.py \
#          --task-file benchmark/eval_tasks_paper.yaml \
#          --output-dir data/grounding_data/ \
#          --steps-per-task 2000
#
#   2. Clone GroundingDINO if not already present:
#      git clone https://github.com/IDEA-Research/GroundingDINO.git
#      cd GroundingDINO && pip install -e .
#
# Usage:
#   bash benchmark/tools/finetune_groundingdino.sh

set -e

DATA_DIR="data/grounding_data"
GDINO_DIR="GroundingDINO"
OUTPUT_DIR="data/grounding_data/checkpoints"

# Pretrained weights - adjust path to your local copy
WEIGHTS="${GDINO_WEIGHTS:-/Users/nanzhang/aimc/data/weights/groundingdino/groundingdino_swint_ogc.pth}"
CONFIG="${GDINO_CONFIG:-${GDINO_DIR}/groundingdino/config/GroundingDINO_SwinT_OGC.py}"

mkdir -p "$OUTPUT_DIR"

echo "=== GroundingDINO Minecraft Fine-tune ==="
echo "Data dir  : $DATA_DIR"
echo "Weights   : $WEIGHTS"
echo "Config    : $CONFIG"
echo "Output    : $OUTPUT_DIR"
echo ""

python benchmark/tools/run_finetune_groundingdino.py \
    --coco-json   "${DATA_DIR}/annotations_coco.json" \
    --images-dir  "${DATA_DIR}/images" \
    --gdino-config  "$CONFIG" \
    --gdino-weights "$WEIGHTS" \
    --output-dir  "$OUTPUT_DIR" \
    --epochs      10 \
    --batch-size  4 \
    --lr          1e-5

echo ""
echo "Done. Fine-tuned weights saved to: $OUTPUT_DIR"
echo ""
echo "To evaluate with the fine-tuned model:"
echo "  python benchmark/benchmark_eval.py \\"
echo "      --task-file benchmark/eval_tasks_paper.yaml \\"
echo "      --goal-backend groundingdino \\"
echo "      --gdino-weights ${OUTPUT_DIR}/checkpoint_best.pth \\"
echo "      --gdino-config ${CONFIG} \\"
echo "      --episodes 3"
