"""
Capture the first frame of each task in eval_tasks_paper.yaml.

Usage:
    python benchmark/capture_first_frames.py \
        --task-file benchmark/eval_tasks_paper.yaml \
        --output-dir benchmark/first_frames/ \
        --tasks mine_coal collect_wood hunt_cowonly
    
    # Capture all tasks:
    python benchmark/capture_first_frames.py \
        --task-file benchmark/eval_tasks_paper.yaml \
        --output-dir benchmark/first_frames/
"""

import os
import sys
import time
import yaml
import argparse
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import load_callbacks_from_config, PrevActionCallback


def main():
    parser = argparse.ArgumentParser(description="Capture first frame per task")
    parser.add_argument("--task-file", required=True)
    parser.add_argument("--output-dir", default="benchmark/first_frames/")
    parser.add_argument("--tasks", nargs="*", default=None,
                        help="Specific task names (default: all)")
    parser.add_argument("--warmup-steps", type=int, default=30)
    args = parser.parse_args()

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

    os.makedirs(args.output_dir, exist_ok=True)

    for task in tasks:
        name = task["name"]
        conf_path = os.path.join(env_conf_dir, task["env_conf"])
        print(f"\n{'='*60}")
        print(f"  Task: {name}")
        print(f"  Prompt: {task.get('text', 'N/A')}")
        print(f"  Config: {conf_path}")
        print(f"{'='*60}")

        try:
            with open(conf_path) as f:
                env_conf = yaml.safe_load(f)

            callbacks = load_callbacks_from_config(env_conf)
            callbacks.append(PrevActionCallback())
            env = MinecraftSim(callbacks=callbacks)

            obs, info = env.reset()
            for _ in range(args.warmup_steps):
                time.sleep(0.05)
                obs, _, _, _, info = env.step(env.noop_action())

            frame = info["pov"]
            out_path = os.path.join(args.output_dir, f"{name}.png")
            cv2.imwrite(out_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            print(f"  Saved: {out_path} ({frame.shape[1]}x{frame.shape[0]})")

            env.close()
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    print(f"\nDone. Frames saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
