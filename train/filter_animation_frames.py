#!/usr/bin/env python3
"""
Filter animation frames from training data.

Uses frame differencing to detect animation effects:
- High frame difference = animation/movement
- Low frame difference = stable frame

Usage:
    python filter_animation_frames.py --data-dir grounding_data_local_v3
"""

import cv2
import numpy as np
import json
import argparse
from pathlib import Path
from collections import defaultdict


def detect_animation_by_diff(frame1, frame2, threshold=10.0):
    """
    Detect if there's animation between two frames.
    
    Returns:
        (is_animated, diff_mean, center_diff)
    """
    # 全图差分
    diff = cv2.absdiff(frame1, frame2)
    diff_mean = diff.mean()
    
    # 中心区域差分（准心位置）
    h, w = frame1.shape[:2]
    center1 = frame1[h//3:2*h//3, w//3:2*w//3]
    center2 = frame2[h//3:2*h//3, w//3:2*w//3]
    center_diff = cv2.absdiff(center1, center2).mean()
    
    is_animated = center_diff > threshold
    
    return is_animated, diff_mean, center_diff


def analyze_data_quality(data_dir):
    """分析训练数据中的动画帧"""
    data_dir = Path(data_dir)
    images_dir = data_dir / "images"
    
    # 加载标注
    with open(data_dir / "annotations_coco.json") as f:
        coco = json.load(f)
    
    id_to_img = {img["id"]: img for img in coco["images"]}
    
    # 按场景分组（step相同的是一个场景）
    scenes = defaultdict(list)
    for img_info in coco["images"]:
        name = img_info["file_name"]
        # 提取 step 部分：collect_wood_step00017
        parts = name.split("_")
        # 找到 step 部分
        step_idx = None
        for i, p in enumerate(parts):
            if p.startswith("step"):
                step_idx = i
                break
        if step_idx:
            scene = "_".join(parts[:step_idx+1])  # collect_wood_step00017
        else:
            scene = "_".join(parts[:2])
        offset = int(name.split("_off")[1].split("_")[0])
        scenes[scene].append((offset, img_info))
    
    print("=== 动画帧检测分析 ===\n")
    
    filtered_images = []
    animation_frames = []
    
    for scene, files in sorted(scenes.items()):
        print(f"\n场景: {scene}")
        files.sort(key=lambda x: x[0])
        
        # 读取图片
        frames = {}
        for offset, img_info in files:
            img_path = images_dir / img_info["file_name"]
            img = cv2.imread(str(img_path))
            if img is not None:
                frames[offset] = (img, img_info)
        
        # 检测每帧是否有动画
        offsets = sorted(frames.keys())
        scene_stable = []
        scene_animated = []
        
        for i, off in enumerate(offsets):
            frame, img_info = frames[off]
            
            # 与下一帧对比（如果存在）
            if i < len(offsets) - 1:
                next_off = offsets[i + 1]
                next_frame, _ = frames[next_off]
                is_anim, diff_mean, center_diff = detect_animation_by_diff(frame, next_frame)
            else:
                # 最后一帧，与前帧对比
                prev_off = offsets[i - 1]
                prev_frame, _ = frames[prev_off]
                is_anim, diff_mean, center_diff = detect_animation_by_diff(frame, prev_frame)
            
            status = "⚠️ 动画" if is_anim else "✅ 稳定"
            print(f"  off{off}: 中心差分={center_diff:.1f} {status}")
            
            if is_anim:
                animation_frames.append(img_info)
                scene_animated.append(off)
            else:
                filtered_images.append(img_info)
                scene_stable.append(off)
        
        print(f"  本场景稳定: {scene_stable}, 动画: {scene_animated}")
    
    # 统计
    total = len(coco["images"])
    stable = len(filtered_images)
    animated = len(animation_frames)
    
    print("="*50)
    print(f"总图片数: {total}")
    print(f"稳定帧: {stable} ({100*stable/total:.1f}%)")
    print(f"动画帧: {animated} ({100*animated/total:.1f}%)")
    print()
    
    return filtered_images, animation_frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--threshold", type=float, default=10.0)
    args = parser.parse_args()
    
    analyze_data_quality(args.data_dir)


if __name__ == "__main__":
    main()