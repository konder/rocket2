"""
ROCKET-2 Benchmark Evaluation Script

Runs automated evaluation of ROCKET-2 across multiple Minecraft tasks.
Supports two task sources:
  - Paper benchmark: env_conf/ YAML files (16 tasks from ROCKET-2 paper)
  - Full benchmark: MineStudio built-in simple tasks (76 tasks)

Usage:
    # macOS local debug (mock Molmo, 1 episode per task)
    python benchmark/benchmark_eval.py \\
        --task-file benchmark/eval_tasks_paper.yaml \\
        --molmo-mock \\
        --episodes 1 \\
        --output-dir benchmark/results/

    # Linux server (real Molmo, 32 episodes, parallel)
    python benchmark/benchmark_eval.py \\
        --task-file benchmark/eval_tasks_paper.yaml \\
        --molmo-model allenai/Molmo-7B-D-0924 \\
        --sam-path ./MineStudio/minestudio/utils/realtime_sam/checkpoints \\
        --episodes 32 \\
        --output-dir benchmark/results/
"""

import os
import sys
import re
import cv2
import json
import time
import yaml
import argparse
import traceback
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import (
    load_callbacks_from_config,
    PrevActionCallback,
)

from benchmark.goal_generator import (
    MockGoalGenerator,
    GoalGeneratorBase,
    MolmoGoalGenerator,
    GroundingDinoGoalGenerator,
)
from benchmark.rocket2_agent import Rocket2Agent


def load_task_file(task_file: str) -> Dict:
    with open(task_file, "r") as f:
        return yaml.safe_load(f)


def create_env_from_paper_task(
    task: Dict, env_conf_dir: str
) -> MinecraftSim:
    """Create MinecraftSim from a paper benchmark task (env_conf YAML)."""
    conf_path = os.path.join(env_conf_dir, task["env_conf"])
    with open(conf_path, "r") as f:
        env_conf = yaml.safe_load(f)
    callbacks = load_callbacks_from_config(env_conf)
    callbacks.append(PrevActionCallback())
    return MinecraftSim(callbacks=callbacks)


def create_env_from_full_task(task: Dict) -> MinecraftSim:
    """Create MinecraftSim from a MineStudio built-in task."""
    from minestudio.benchmark import prepare_task_configs

    task_configs = prepare_task_configs("simple")
    task_name = task["name"]
    if task_name not in task_configs:
        raise ValueError(f"Task '{task_name}' not found in MineStudio simple configs")
    config_file = task_configs[task_name]
    callbacks = load_callbacks_from_config(config_file)
    callbacks.append(PrevActionCallback())
    return MinecraftSim(callbacks=callbacks)


def check_success(info_history: List[Dict], criteria: Optional[Dict]) -> bool:
    """
    Check episode success based on info events.

    Scans all info frames (not just the last one) so that transient events
    that might not persist in the final frame are still captured.
    """
    if criteria is None:
        return False
    if not info_history:
        return False

    key = criteria["key"]
    regex = criteria["regex"]
    num = criteria["num"]

    best = 0
    for info in info_history:
        if key not in info:
            continue
        events = info[key]
        if isinstance(events, dict):
            total = sum(float(c) for name, c in events.items() if re.match(regex, name))
            best = max(best, total)
    return best >= num


def check_success_single(info: Dict, criteria: Optional[Dict]) -> bool:
    """Fast single-frame success check for early termination."""
    if criteria is None:
        return False
    key = criteria["key"]
    if key not in info:
        return False
    events = info[key]
    if not isinstance(events, dict):
        return False
    total = sum(
        float(c) for name, c in events.items() if re.match(criteria["regex"], name)
    )
    return total >= criteria["num"]


def warmup_env(env: MinecraftSim, steps: int = 30) -> tuple:
    """Run noop actions to let the environment stabilize."""
    obs, info = env.reset()
    for _ in range(steps):
        time.sleep(0.05)
        noop = env.noop_action()
        obs, reward, terminated, truncated, info = env.step(noop)
    return obs, info


def _build_goal_thumbnail(
    goal_image: np.ndarray,
    mask: np.ndarray,
    point: tuple,
    interaction_type: str = "none",
    scale: float = 0.3,
) -> np.ndarray:
    """Build a thumbnail using launch.py-like mask visualization style."""
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (255, 255, 255), (0, 0, 0), (128, 128, 128),
        (128, 0, 0), (128, 128, 0), (0, 128, 0),
        (128, 0, 128), (0, 128, 128), (0, 0, 128),
    ]
    segment_mapping = {
        "hunt": 0, "use": 3, "mine": 2, "interact": 3,
        "craft": 4, "switch": 5, "approach": 6, "none": -1
    }

    vis = goal_image.copy()
    key = interaction_type.lower()
    color_idx = segment_mapping.get(key, -1)
    rgb_color = colors[color_idx] if color_idx >= 0 else (0, 255, 0)
    bgr_color = np.array(rgb_color, dtype=np.uint8).reshape(1, 1, 3)[:, :, ::-1]

    mask_overlay = (mask[..., None] * bgr_color).astype(np.uint8)
    vis = cv2.addWeighted(vis, 1.0, mask_overlay, 0.5, 0.0)

    binary_mask = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (255, 0, 0), 2)

    cv2.circle(vis, point, 5, (255, 255, 255), -1)
    vis = cv2.copyMakeBorder(vis, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    h, w = vis.shape[:2]
    vis = cv2.resize(vis, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    return vis


def _overlay_thumbnail(frame: np.ndarray, thumbnail: np.ndarray) -> np.ndarray:
    """Paste thumbnail in the top-right corner of the frame."""
    out = frame.copy()
    th, tw = thumbnail.shape[:2]
    sx = out.shape[1] - 5 - tw
    sy = 5
    if sy + th <= out.shape[0] and sx + tw <= out.shape[1] and sx >= 0:
        out[sy:sy + th, sx:sx + tw] = thumbnail
    return out


def run_single_episode(
    agent: Rocket2Agent,
    env: MinecraftSim,
    goal_gen: GoalGeneratorBase,
    task: Dict,
    max_steps: int,
    save_video: bool = False,
    video_path: Optional[str] = None,
    retry_interval: int = 20,
) -> Dict[str, Any]:
    """
    Run a single evaluation episode with explore-then-execute strategy.

    If Molmo cannot find the target in the current frame, the agent enters
    exploration mode (obj_id=-1, mask=0). Every `retry_interval` steps,
    the goal generator retries on the latest frame until the target is found.

    Returns a result dict with success, num_steps, and timing info.
    """
    agent.reset()

    obs, info = warmup_env(env)
    first_frame = info["pov"]

    text_prompt = task.get("text", "Point to the target object")
    interaction_type = task.get("interaction_type", "none")

    point, mask = goal_gen.generate(first_frame, text_prompt)
    goal_found = point is not None

    if goal_found:
        agent.set_goal(first_frame, mask, interaction_type)
        print(f"    Goal found at {point}")
    else:
        agent.set_goal(first_frame, mask, "none")
        print(f"    Target not visible, entering exploration mode")

    goal_thumb = None
    if save_video:
        goal_thumb = _build_goal_thumbnail(
            first_frame, mask, point or (0, 0), interaction_type=interaction_type
        )

    info_history = [info]
    frames = []
    if save_video:
        frames.append(_overlay_thumbnail(first_frame, goal_thumb))
    t0 = time.time()

    criteria = task.get("success_criteria")
    early_done = False

    for step in range(max_steps):
        try:
            # Retry goal detection periodically during exploration
            if not goal_found and step > 0 and step % retry_interval == 0:
                current_frame = info["pov"]
                point, mask = goal_gen.generate(current_frame, text_prompt)
                if point is not None:
                    goal_found = True
                    agent.set_goal(current_frame, mask, interaction_type)
                    if save_video:
                        goal_thumb = _build_goal_thumbnail(
                            current_frame, mask, point, interaction_type=interaction_type
                        )
                    print(f"    Goal found at step {step} at {point}")

            action = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            info_history.append(info)
            if save_video:
                frame = _overlay_thumbnail(info["pov"], goal_thumb)
                mode = "EXPLORE" if not goal_found else ""
                step_label = f"Step {step+1}/{max_steps} {mode}".strip()
                cv2.putText(
                    frame, step_label, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                    cv2.LINE_AA,
                )
                frames.append(frame)
            if criteria and check_success_single(info, criteria):
                early_done = True
                break
        except Exception:
            traceback.print_exc()
            break

    elapsed = time.time() - t0
    success = check_success(info_history, criteria)

    if save_video and video_path and frames:
        _save_video(frames, video_path)

    actual_steps = len(info_history) - 1
    result = {
        "success": success,
        "num_steps": actual_steps,
        "max_steps": max_steps,
        "early_done": early_done,
        "goal_found": goal_found,
        "elapsed_sec": round(elapsed, 2),
        "fps": round(actual_steps / max(elapsed, 0.01), 1),
        "goal_point": list(point) if point else None,
        "interaction_type": interaction_type,
    }
    if video_path and save_video:
        result["video_path"] = video_path
    return result


def _save_video(frames: List[np.ndarray], path: str):
    """Save episode frames as an MP4 video."""
    try:
        import av

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with av.open(path, mode="w", format="mp4") as container:
            stream = container.add_stream("h264", rate=20)
            stream.width = frames[0].shape[1]
            stream.height = frames[0].shape[0]
            stream.pix_fmt = "yuv420p"
            for frame_np in frames:
                frame = av.VideoFrame.from_ndarray(frame_np, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        print(f"    Video saved: {path}")
    except ImportError:
        print("[WARN] av not installed, skipping video save")


def run_task_evaluation(
    task: Dict,
    agent: Rocket2Agent,
    goal_gen: GoalGeneratorBase,
    env_conf_dir: Optional[str],
    num_episodes: int,
    output_dir: str,
    save_video: bool = False,
    retry_interval: int = 20,
) -> Dict[str, Any]:
    """Run all episodes for a single task and aggregate results."""
    task_name = task["name"]
    max_steps = task.get("max_steps", 600)
    is_paper_task = "env_conf" in task

    print(f"\n{'='*60}")
    print(f"  Task: {task_name} ({task.get('interaction_type', '?')})")
    print(f"  Episodes: {num_episodes}, Max steps: {max_steps}")
    print(f"{'='*60}")

    episode_results = []
    for ep in range(num_episodes):
        print(f"  Episode {ep+1}/{num_episodes} ...", end=" ", flush=True)

        try:
            if is_paper_task:
                env = create_env_from_paper_task(task, env_conf_dir)
            else:
                env = create_env_from_full_task(task)
        except Exception:
            traceback.print_exc()
            episode_results.append({"success": False, "error": "env_creation_failed"})
            print("FAIL (env)")
            continue

        try:
            video_path = None
            if save_video:
                video_path = os.path.join(
                    output_dir, "videos", f"{task_name}_ep{ep}.mp4"
                )

            result = run_single_episode(
                agent, env, goal_gen, task, max_steps,
                save_video=save_video, video_path=video_path,
                retry_interval=retry_interval,
            )
            episode_results.append(result)
            status = "OK" if result["success"] else "FAIL"
            steps_info = f"{result['num_steps']}/{result['max_steps']} steps"
            if result.get("early_done"):
                steps_info += " (early)"
            print(f"{status} ({steps_info}, {result['elapsed_sec']}s, {result['fps']} fps)")

        except Exception:
            traceback.print_exc()
            episode_results.append({"success": False, "error": "episode_failed"})
            print("FAIL (exception)")

        finally:
            try:
                env.close()
            except Exception:
                pass

    num_success = sum(1 for r in episode_results if r.get("success", False))
    success_rate = num_success / max(len(episode_results), 1)

    task_summary = {
        "task_name": task_name,
        "interaction_type": task.get("interaction_type", "unknown"),
        "num_episodes": len(episode_results),
        "num_success": num_success,
        "success_rate": round(success_rate, 4),
        "episodes": episode_results,
    }

    print(f"  Result: {num_success}/{len(episode_results)} "
          f"({success_rate*100:.1f}%)")
    return task_summary


def aggregate_results(task_results: List[Dict]) -> Dict:
    """Compute per-interaction-type and overall statistics."""
    by_type: Dict[str, List] = {}
    for tr in task_results:
        itype = tr.get("interaction_type", "unknown")
        by_type.setdefault(itype, []).append(tr)

    type_summary = {}
    for itype, tasks in sorted(by_type.items()):
        total_ep = sum(t["num_episodes"] for t in tasks)
        total_success = sum(t["num_success"] for t in tasks)
        rate = total_success / max(total_ep, 1)
        type_summary[itype] = {
            "num_tasks": len(tasks),
            "total_episodes": total_ep,
            "total_success": total_success,
            "success_rate": round(rate, 4),
            "tasks": [t["task_name"] for t in tasks],
        }

    overall_ep = sum(t["num_episodes"] for t in task_results)
    overall_success = sum(t["num_success"] for t in task_results)

    return {
        "overall": {
            "num_tasks": len(task_results),
            "total_episodes": overall_ep,
            "total_success": overall_success,
            "success_rate": round(overall_success / max(overall_ep, 1), 4),
        },
        "by_interaction_type": type_summary,
    }


def print_summary(summary: Dict):
    """Print a formatted summary table."""
    print(f"\n{'='*70}")
    print("  BENCHMARK RESULTS SUMMARY")
    print(f"{'='*70}")

    overall = summary["overall"]
    print(f"\n  Overall: {overall['total_success']}/{overall['total_episodes']} "
          f"({overall['success_rate']*100:.1f}%) across {overall['num_tasks']} tasks\n")

    print(f"  {'Type':<15} {'Tasks':>5} {'Episodes':>8} {'Success':>8} {'Rate':>8}")
    print(f"  {'-'*15} {'-'*5} {'-'*8} {'-'*8} {'-'*8}")

    for itype, data in sorted(summary["by_interaction_type"].items()):
        print(f"  {itype:<15} {data['num_tasks']:>5} {data['total_episodes']:>8} "
              f"{data['total_success']:>8} {data['success_rate']*100:>7.1f}%")

    print(f"\n{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description="ROCKET-2 Benchmark Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--task-file", required=True,
        help="Path to task YAML (eval_tasks_paper.yaml or eval_tasks_full.yaml)",
    )
    parser.add_argument(
        "--ckpt", default="hf:phython96/ROCKET-2-1.5x-17w",
        help="ROCKET-2 checkpoint (HuggingFace hf:repo/name or local path)",
    )
    parser.add_argument("--cfg-coef", type=float, default=1.5,
                        help="Classifier-free guidance coefficient")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Override episodes per task (default: from YAML)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override max steps per episode (default: from YAML)")
    parser.add_argument("--tasks", nargs="*", default=None,
                        help="Run only specific tasks by name (default: all)")

    goal_group = parser.add_argument_group("Goal generation")
    goal_group.add_argument(
        "--goal-backend",
        default="molmo",
        choices=["molmo", "groundingdino"],
        help="Goal point backend: molmo or groundingdino (default: molmo)",
    )
    goal_group.add_argument("--molmo-mock", action="store_true",
                            help="Use mock goal generator (center point, no Molmo)")
    goal_group.add_argument("--molmo-model", default="allenai/Molmo-7B-D-0924",
                            help="Molmo model ID for goal generation")
    goal_group.add_argument(
        "--gdino-config", default=None,
        help="GroundingDINO config .py path (optional if HF auto-download works)",
    )
    goal_group.add_argument(
        "--gdino-weights", default=None,
        help="GroundingDINO weights .pth path (optional if HF auto-download works)",
    )
    goal_group.add_argument(
        "--gdino-hf-repo", default="ShilongLiu/GroundingDINO",
        help="HF repo ID to auto-download GroundingDINO assets",
    )
    goal_group.add_argument("--gdino-box-threshold", type=float, default=0.25,
                            help="GroundingDINO box threshold (default: 0.25)")
    goal_group.add_argument("--gdino-text-threshold", type=float, default=0.20,
                            help="GroundingDINO text threshold (default: 0.20)")
    goal_group.add_argument(
        "--sam-path",
        default="./MineStudio/minestudio/utils/realtime_sam/checkpoints",
        help="Path to SAM-2 checkpoints",
    )
    goal_group.add_argument("--sam-variant", default="base",
                            choices=["large", "base", "small", "tiny"])
    goal_group.add_argument("--retry-interval", type=int, default=20,
                            help="Steps between Molmo retries during exploration (default: 20)")

    out_group = parser.add_argument_group("Output")
    out_group.add_argument("--output-dir", default="benchmark/results/",
                           help="Directory for results JSON and videos")
    out_group.add_argument("--save-video", action="store_true",
                           help="Save episode videos (slow, large files)")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    task_config = load_task_file(args.task_file)
    tasks = task_config["tasks"]

    if args.tasks:
        tasks = [t for t in tasks if t["name"] in args.tasks]
        if not tasks:
            print(f"ERROR: No tasks matched {args.tasks}")
            sys.exit(1)

    env_conf_dir = task_config.get("env_conf_dir")
    if env_conf_dir and not os.path.isabs(env_conf_dir):
        env_conf_dir = os.path.normpath(
            os.path.join(os.path.dirname(args.task_file), env_conf_dir)
        )

    print("Initializing ROCKET-2 agent ...")
    agent = Rocket2Agent(
        ckpt_path=args.ckpt,
        cfg_coef=args.cfg_coef,
    )

    if args.molmo_mock:
        print("Using MockGoalGenerator (no Molmo)")
        goal_gen = MockGoalGenerator()
    else:
        if args.goal_backend == "groundingdino":
            print("Using GroundingDinoGoalGenerator")
            goal_gen = GroundingDinoGoalGenerator(
                sam_path=args.sam_path,
                sam_variant=args.sam_variant,
                gdino_config=args.gdino_config,
                gdino_weights=args.gdino_weights,
                gdino_hf_repo=args.gdino_hf_repo,
                box_threshold=args.gdino_box_threshold,
                text_threshold=args.gdino_text_threshold,
            )
        else:
            print(f"Using MolmoGoalGenerator ({args.molmo_model})")
            goal_gen = MolmoGoalGenerator(
                molmo_model_id=args.molmo_model,
                sam_path=args.sam_path,
                sam_variant=args.sam_variant,
            )

    default_episodes = task_config.get("default_episodes", 3)
    task_results = []

    for task in tasks:
        num_episodes = args.episodes or task.get("episodes", default_episodes)
        if args.max_steps:
            task = {**task, "max_steps": args.max_steps}

        result = run_task_evaluation(
            task=task,
            agent=agent,
            goal_gen=goal_gen,
            env_conf_dir=env_conf_dir,
            num_episodes=num_episodes,
            output_dir=args.output_dir,
            save_video=args.save_video,
            retry_interval=args.retry_interval,
        )
        task_results.append(result)

    summary = aggregate_results(task_results)
    print_summary(summary)

    output = {
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "task_file": args.task_file,
            "ckpt": args.ckpt,
            "cfg_coef": args.cfg_coef,
            "molmo_mock": args.molmo_mock,
            "goal_backend": args.goal_backend,
            "molmo_model": args.molmo_model if not args.molmo_mock else None,
            "gdino_config": args.gdino_config,
            "gdino_weights": args.gdino_weights,
            "gdino_hf_repo": args.gdino_hf_repo,
        },
        "summary": summary,
        "task_results": task_results,
    }
    result_path = os.path.join(
        args.output_dir,
        f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    with open(result_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {result_path}")


if __name__ == "__main__":
    main()
