"""
Collect (image, block_name, bbox) training data for GroundingDINO fine-tuning.

Strategy:
  - Use VPT policy to drive the agent — it naturally mines blocks, navigates,
    and performs meaningful actions without manual camera control.
  - Keep a rolling frame buffer (last BUFFER_SIZE frames).
  - When mine_block count increases, save frames from SAVE_OFFSET steps before
    the event (skipping the breaking-animation frame right before the event).
  - Label = mined block name; bbox = fixed center square (accuracy not critical).

Output: COCO-format JSON + images, ready for GroundingDINO fine-tuning.

Usage:
    # Collect from specific tasks:
    python collect_grounding_data.py \\
        --task-file eval/eval_tasks_paper.yaml \\
        --output-dir data/grounding_data/ \\
        --steps-per-task 2000 \\
        --tasks mine_coal mine_iron mine_emerald

    # Collect from all tasks:
    python collect_grounding_data.py \\
        --task-file eval/eval_tasks_paper.yaml \\
        --output-dir data/grounding_data/ \\
        --steps-per-task 3000
"""

import os
import sys
import json
import copy
import yaml
import argparse
import cv2
import numpy as np
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import load_callbacks_from_config, PrevActionCallback
from minestudio.models import VPTPolicy

# Default frame offsets before mine event to save.
# -1 is the breaking-animation frame (avoid); -3 ~ -6 show the intact block clearly.
DEFAULT_SAVE_OFFSETS = [-6, -5, -4, -3]


# --------------------------------------------------------------------------- #
# Block name normalization
# --------------------------------------------------------------------------- #

BLOCK_NAME_MAP = {
    "coal_ore":               "coal ore",
    "deepslate_coal_ore":     "coal ore",
    "iron_ore":               "iron ore",
    "deepslate_iron_ore":     "iron ore",
    "gold_ore":               "gold ore",
    "deepslate_gold_ore":     "gold ore",
    "diamond_ore":            "diamond ore",
    "deepslate_diamond_ore":  "diamond ore",
    "emerald_ore":            "emerald ore",
    "deepslate_emerald_ore":  "emerald ore",
    "lapis_ore":              "lapis ore",
    "deepslate_lapis_ore":    "lapis ore",
    "redstone_ore":           "redstone ore",
    "deepslate_redstone_ore": "redstone ore",
    "copper_ore":             "copper ore",
    "deepslate_copper_ore":   "copper ore",
    "stone":                  "stone",
    "cobblestone":            "cobblestone",
    "dirt":                   "dirt",
    "grass_block":            "grass block",
    "sand":                   "sand",
    "gravel":                 "gravel",
    "oak_log":                "oak log",
    "birch_log":              "birch log",
    "spruce_log":             "spruce log",
    "jungle_log":             "jungle log",
    "acacia_log":             "acacia log",
    "dark_oak_log":           "dark oak log",
}


def normalize_block_name(raw: str) -> str:
    """'minecraft:coal_ore' → 'coal ore'"""
    name = raw.split(":")[-1].lower()
    return BLOCK_NAME_MAP.get(name, name.replace("_", " "))


# --------------------------------------------------------------------------- #
# Fixed center bbox (block is always near crosshair center)
# --------------------------------------------------------------------------- #

CENTER_BOX_FRACTION = 0.25   # bbox side = 25% of image size


def center_bbox(img_w: int, img_h: int) -> tuple:
    side_w = int(img_w * CENTER_BOX_FRACTION)
    side_h = int(img_h * CENTER_BOX_FRACTION)
    cx, cy = img_w // 2, img_h // 2
    return (cx - side_w // 2, cy - side_h // 2,
            cx + side_w // 2, cy + side_h // 2)


# --------------------------------------------------------------------------- #
# mine_block delta detection
# Works with both flat dicts and nested {"minecraft": {"coal_ore": count}} dicts.
# --------------------------------------------------------------------------- #

def _flatten_mine_block(mb) -> dict:
    """Flatten mine_block dict to {raw_name: count}."""
    flat = {}
    if not isinstance(mb, dict):
        return flat
    for k, v in mb.items():
        if isinstance(v, dict):
            for blk, cnt in v.items():
                flat[blk] = int(cnt) if cnt is not None else 0
        else:
            flat[k] = int(v) if v is not None else 0
    return flat


def detect_new_mines(cur_mb, prev_mb_flat: dict) -> list:
    """Return list of raw block names that were newly mined."""
    cur_flat = _flatten_mine_block(cur_mb)
    newly = []
    for raw_name, cnt in cur_flat.items():
        if cnt > prev_mb_flat.get(raw_name, 0):
            newly.append(raw_name)
    return newly, cur_flat


# --------------------------------------------------------------------------- #
# Collection logic
# --------------------------------------------------------------------------- #

def collect_from_task(
    task: dict,
    env_conf_dir: str,
    output_images_dir: str,
    steps: int,
    vpt_model: VPTPolicy,
    device: str,
    save_offsets: list = None,
    max_samples: int = None,
) -> list:
    """
    Run one task with a VPT agent, collect mine_block-triggered annotations.

    Stopping conditions (whichever comes first):
      1. Ran `steps` steps.
      2. Collected `max_samples` annotation frames (if set).
      3. A step raises an exception.

    Returns list of annotation dicts.
    """
    if save_offsets is None:
        save_offsets = DEFAULT_SAVE_OFFSETS
    conf_path = os.path.join(env_conf_dir, task["env_conf"])
    task_name = task["name"]
    task_target = task.get("text", "")

    try:
        with open(conf_path) as f:
            env_conf = yaml.safe_load(f)
    except Exception as e:
        print(f"  [SKIP] Cannot load config {conf_path}: {e}")
        return []

    callbacks = load_callbacks_from_config(env_conf)
    callbacks.append(PrevActionCallback())

    try:
        # VPT was trained on 128x128 images — must match obs_size
        # action_type='agent': VPT outputs (buttons, camera) index format
        sim = MinecraftSim(callbacks=callbacks, action_type='agent', obs_size=(128, 128))
        obs, info = sim.reset()
    except Exception as e:
        print(f"  [ERROR] Env init failed: {e}")
        return []

    # initial_state(None) returns unbatched state for input_shape="*"
    # get_action with input_shape="*" adds the batch dim internally via unsqueeze(0)
    state = vpt_model.initial_state(batch_size=None)

    # Rolling frame buffer sized to hold the furthest-back offset needed
    buffer_size = max(abs(o) for o in save_offsets) + 2
    frame_buffer = deque(maxlen=buffer_size)

    annotations = []
    prev_mb_flat = _flatten_mine_block(info.get("mine_block", {}))

    # Warmup — let the world load before collecting
    print(f"  Warming up (40 steps)...")
    for _ in range(40):
        action, state = vpt_model.get_action(
            {"image": obs["image"]}, state, input_shape="*"
        )
        obs, _, _, _, info = sim.step(action)
        frame_buffer.append(info.get("pov", obs["image"]).copy())

    prev_mb_flat = _flatten_mine_block(info.get("mine_block", {}))
    stop_reason = f"reached {steps} steps"
    if max_samples:
        print(f"  Collecting up to {steps} steps / {max_samples} samples "
              f"for task '{task_name}' (target: {task_target})")
    else:
        print(f"  Collecting {steps} steps for task '{task_name}' (target: {task_target})")

    for step_i in range(steps):
        # VPT decides the action
        try:
            action, state = vpt_model.get_action(
                {"image": obs["image"]}, state, input_shape="*"
            )
            obs, _, _, _, info = sim.step(action)
        except Exception as e:
            import traceback
            print(f"  [WARN] Step {step_i} failed: {e}")
            traceback.print_exc()
            break

        # Buffer the current frame
        frame_buffer.append(info.get("pov", obs["image"]).copy())

        # Detect mining events
        cur_mb = info.get("mine_block", {})
        newly_mined, cur_mb_flat = detect_new_mines(cur_mb, prev_mb_flat)

        for raw_name in newly_mined:
            block_label = normalize_block_name(raw_name)

            # Save several frames before the event (skip breaking-animation frame)
            buf_list = list(frame_buffer)   # oldest → newest
            for offset in save_offsets:
                idx = len(buf_list) + offset   # offset is negative
                if idx < 0 or idx >= len(buf_list):
                    continue

                frame = buf_list[idx]
                h, w = frame.shape[:2]
                bbox  = center_bbox(w, h)

                img_filename = f"{task_name}_step{step_i:05d}_off{offset}_{raw_name}.png"
                img_path = os.path.join(output_images_dir, img_filename)
                
                # Check cv2.imwrite return value (it doesn't throw exceptions on failure)
                try:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    success = cv2.imwrite(img_path, frame_bgr)
                    if not success:
                        print(f"    [ERROR] cv2.imwrite failed for {img_filename}")
                        print(f"           Path: {img_path}")
                        print(f"           Frame: {frame.shape}, {frame.dtype}")
                        continue  # Skip this annotation if image write failed
                except Exception as e:
                    print(f"    [ERROR] Failed to save {img_filename}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

                ann = {
                    "image_path":  img_filename,
                    "block_key":   raw_name,
                    "label":       block_label,
                    "task":        task_name,
                    "task_target": task_target,
                    "bbox_xyxy":   list(bbox),
                    "img_w":       w,
                    "img_h":       h,
                    "step":        step_i,
                    "offset":      offset,
                }
                annotations.append(ann)

            print(f"    [+] step={step_i} block='{block_label}' "
                  f"→ saved {len(save_offsets)} frames "
                  f"(total so far: {len(annotations)})")

        prev_mb_flat = cur_mb_flat

        # Stopping condition 2: enough samples collected
        if max_samples and len(annotations) >= max_samples:
            stop_reason = f"reached {max_samples} samples"
            break

    print(f"  Stopped: {stop_reason}. Collected {len(annotations)} annotations.")
    sim.close()
    return annotations


# --------------------------------------------------------------------------- #
# COCO JSON export
# --------------------------------------------------------------------------- #

MINECRAFT_CATEGORIES = [
    {"id": i + 1, "name": name, "supercategory": "block"}
    for i, name in enumerate(sorted(set(BLOCK_NAME_MAP.values())))
]
CAT_NAME_TO_ID = {c["name"]: c["id"] for c in MINECRAFT_CATEGORIES}


def to_coco(annotations: list) -> dict:
    images, coco_anns = [], []
    img_id_map = {}
    ann_id = 1

    for ann in annotations:
        fname = ann["image_path"]
        if fname not in img_id_map:
            img_id = len(images) + 1
            img_id_map[fname] = img_id
            images.append({
                "id":        img_id,
                "file_name": fname,
                "width":     ann["img_w"],
                "height":    ann["img_h"],
            })
        img_id = img_id_map[fname]

        x1, y1, x2, y2 = ann["bbox_xyxy"]
        w, h = x2 - x1, y2 - y1
        cat_id = CAT_NAME_TO_ID.get(ann["label"], 1)

        coco_anns.append({
            "id":          ann_id,
            "image_id":    img_id,
            "category_id": cat_id,
            "bbox":        [x1, y1, w, h],
            "area":        w * h,
            "iscrowd":     0,
            "caption":     ann["label"],
        })
        ann_id += 1

    return {
        "info":        {"description": "Minecraft GroundingDINO training data"},
        "categories":  MINECRAFT_CATEGORIES,
        "images":      images,
        "annotations": coco_anns,
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Collect Minecraft grounding data via VPT")
    parser.add_argument("--task-file",      required=True, help="eval_tasks_paper.yaml")
    parser.add_argument("--output-dir",     default="data/grounding_data/")
    parser.add_argument("--steps-per-task", type=int, default=2000)
    parser.add_argument("--tasks",          nargs="*", default=None,
                        help="Task names to collect (default: all)")
    parser.add_argument("--vpt-model",      default="CraftJarvis/MineStudio_VPT.rl_from_early_game_2x",
                        help="HuggingFace repo or local path for VPT weights")
    parser.add_argument("--save-offsets",   nargs="+", type=int, default=DEFAULT_SAVE_OFFSETS,
                        help="Frame offsets before mine event to save (default: -6 -5 -4 -3). "
                             "Use negative integers. -1 is the breaking-animation frame (avoid).")
    parser.add_argument("--max-samples",    type=int, default=None,
                        help="Stop a task early once this many frames are collected (default: run full --steps-per-task)")
    args = parser.parse_args()

    # Validate offsets are all negative
    for o in args.save_offsets:
        if o >= 0:
            raise ValueError(f"--save-offsets must be negative integers, got {o}")

    with open(args.task_file) as f:
        config = yaml.safe_load(f)

    env_conf_dir = config.get("env_conf_dir", "../env_conf")
    if not os.path.isabs(env_conf_dir):
        env_conf_dir = os.path.normpath(
            os.path.join(os.path.dirname(args.task_file), env_conf_dir)
        )

    tasks = config["tasks"]
    if args.tasks:
        tasks = [t for t in tasks if t["name"] in args.tasks]

    images_dir = os.path.join(args.output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Load VPT once and reuse across tasks
    import torch
    from minestudio.models import load_vpt_policy
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading VPT model from '{args.vpt_model}' on {device} ...")
    # Use load_vpt_policy for compatibility with MineStudio 1.1.2+
    vpt_model = load_vpt_policy(model_path=None)  # Loads from HuggingFace
    vpt_model = vpt_model.to(device).eval()
    print("VPT model loaded.\n")

    all_annotations = []

    for task in tasks:
        print(f"\n{'='*60}")
        print(f"Task: {task['name']}")
        print(f"{'='*60}")
        anns = collect_from_task(
            task=task,
            env_conf_dir=env_conf_dir,
            output_images_dir=images_dir,
            steps=args.steps_per_task,
            vpt_model=vpt_model,
            device=device,
            save_offsets=args.save_offsets,
            max_samples=args.max_samples,
        )
        all_annotations.extend(anns)
        print(f"  Collected {len(anns)} samples from '{task['name']}'")

    # Save raw annotations
    raw_path = os.path.join(args.output_dir, "annotations_raw.json")
    with open(raw_path, "w") as f:
        json.dump(all_annotations, f, indent=2)
    print(f"\nRaw annotations: {raw_path} ({len(all_annotations)} total)")

    # Save COCO format
    coco = to_coco(all_annotations)
    coco_path = os.path.join(args.output_dir, "annotations_coco.json")
    with open(coco_path, "w") as f:
        json.dump(coco, f, indent=2)
    print(f"COCO annotations: {coco_path}")
    print(f"  Images: {len(coco['images'])}, Annotations: {len(coco['annotations'])}")
    print(f"\nNext: bash finetune_groundingdino.sh")


if __name__ == "__main__":
    main()
