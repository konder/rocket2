# Benchmark tools

Helper scripts for **comparing and judging detection backends**. They are **not** used by the main evaluation script `benchmark/benchmark_eval.py`, which uses **GroundingDino** as the official goal generator.

## Workflow

1. **capture_first_frames.py** — Capture the first frame of each paper task (requires MineStudio env).
2. **compare_detectors.py** — Run multiple backends (GroundingDINO, Molmo+SAM2, DINO+SAM2, Sa2VA, Qwen-VL API) on those frames and write `detection_results.json`.
3. **judge_detections.py** — Use a VLM to pick the best detection per task from `detection_results.json`.

## Usage (from repo root)

```bash
# 1. Capture first frames
python benchmark/tools/capture_first_frames.py \
    --task-file benchmark/eval_tasks_paper.yaml \
    --output-dir benchmark/first_frames/

# 2. Compare detectors (e.g. DINO vs Qwen VL API)
python benchmark/tools/compare_detectors.py \
    --task-file benchmark/eval_tasks_paper.yaml \
    --image-dir benchmark/first_frames/ \
    --output-dir benchmark/detection_compare/ \
    --backends groundingdino qwen_api \
    --qwen-api-base http://localhost:8000

# 3. Judge which detection is best per task
python benchmark/tools/judge_detections.py \
    --task-file benchmark/eval_tasks_paper.yaml \
    --image-dir benchmark/first_frames/ \
    --results-json benchmark/detection_compare/detection_results.json \
    --output-dir benchmark/judge_results/ \
    --api-base http://localhost:8000
```

## Main evaluation (GroundingDino)

Run the official benchmark with GroundingDino (default):

```bash
python benchmark/benchmark_eval.py \
    --task-file benchmark/eval_tasks_paper.yaml \
    --episodes 3 \
    --output-dir benchmark/results/
```

Optionally use `--goal-backend molmo` and `--molmo-model ...` for Molmo+SAM2, or `--molmo-mock` for debugging without a detector.
