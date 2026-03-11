#!/usr/bin/env python3
"""
Retry wrapper for collect_grounding_data.py

Minecraft simulator has a known initialization issue:
  [ERROR] Env init failed: a bytes-like object is required, not 'NoneType'

This script wraps the data collection with automatic retry.
"""

import subprocess
import sys
import time
import argparse
from pathlib import Path


def run_with_retry(command: list, max_retries: int = 5, retry_delay: int = 10) -> bool:
    """
    Run command with automatic retry on failure.
    
    Returns:
        True if succeeded, False if all retries failed
    """
    for attempt in range(1, max_retries + 1):
        print(f"\n{'='*60}")
        print(f"Attempt {attempt}/{max_retries}")
        print(f"{'='*60}\n")
        
        result = subprocess.run(command, capture_output=False)
        
        if result.returncode == 0:
            print(f"\n✅ Success on attempt {attempt}")
            return True
        
        print(f"\n❌ Attempt {attempt} failed")
        
        if attempt < max_retries:
            print(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
    
    print(f"\n❌ All {max_retries} attempts failed")
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Run data collection with retry on Minecraft init errors"
    )
    parser.add_argument("--max-retries", type=int, default=5, help="Max retry attempts")
    parser.add_argument("--retry-delay", type=int, default=10, help="Seconds between retries")
    parser.add_argument("--task-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps-per-task", type=int, default=1000)
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--save-offsets", nargs="+", type=int, default=[-20, -15, -10])
    parser.add_argument("--max-samples", type=int, default=100)
    
    args = parser.parse_args()
    
    # Build command
    cmd = [
        sys.executable,
        "train/collect_grounding_data.py",
        "--task-file", args.task_file,
        "--output-dir", args.output_dir,
        "--steps-per-task", str(args.steps_per_task),
        "--save-offsets", *[str(o) for o in args.save_offsets],
        "--max-samples", str(args.max_samples),
    ]
    
    if args.tasks:
        cmd.extend(["--tasks", *args.tasks])
    
    print("Command:", " ".join(cmd))
    
    success = run_with_retry(cmd, args.max_retries, args.retry_delay)
    
    if success:
        print("\n" + "="*60)
        print("Data collection completed successfully!")
        print("="*60)
        sys.exit(0)
    else:
        print("\n" + "="*60)
        print("Data collection failed after all retries.")
        print("="*60)
        sys.exit(1)


if __name__ == "__main__":
    main()