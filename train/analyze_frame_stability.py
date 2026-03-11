#!/usr/bin/env python3
"""
改进的帧稳定性分析

问题：相邻帧比较可能误判（两个动画帧之间差分可能很小）
解决：对更大范围的帧做全局分析，找出最稳定的帧

方法：
1. 收集大范围的帧（20-30帧）
2. 计算所有帧之间的差异矩阵
3. 找出与其他帧差异最小的帧（最稳定）
4. 过滤掉差异大的帧（动画帧）
"""

import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict


def analyze_global_stability(frames: dict) -> dict:
    """
    对一组帧做全局稳定性分析
    
    Returns:
        每个offset的稳定性得分（与其他帧的平均差异）
    """
    offsets = sorted(frames.keys())
    n = len(offsets)
    
    if n < 2:
        return {offsets[0]: 0} if n == 1 else {}
    
    # 计算差异矩阵
    diff_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            f1, f2 = frames[offsets[i]], frames[offsets[j]]
            diff = cv2.absdiff(f1, f2).mean()
            diff_matrix[i, j] = diff
            diff_matrix[j, i] = diff
    
    # 计算每个帧的稳定性得分（与其他帧的平均差异）
    stability_scores = {}
    for i, offset in enumerate(offsets):
        avg_diff = diff_matrix[i].mean()
        stability_scores[offset] = avg_diff
    
    return stability_scores


def find_stable_frames(frames: dict, threshold_factor: float = 0.7) -> tuple:
    """
    找出稳定帧
    
    Args:
        frames: {offset: image}
        threshold_factor: 稳定性阈值因子（越小越严格）
    
    Returns:
        (stable_offsets, animation_offsets, stability_scores)
    """
    scores = analyze_global_stability(frames)
    
    if not scores:
        return [], [], {}
    
    # 找出最稳定的帧（差异最小）
    min_score = min(scores.values())
    max_score = max(scores.values())
    threshold = min_score + (max_score - min_score) * threshold_factor
    
    stable = [off for off, score in scores.items() if score <= threshold]
    animation = [off for off, score in scores.items() if score > threshold]
    
    return stable, animation, scores


def main():
    data_dir = Path("/Users/nanzhang/rocket2/grounding_data_local_v5")
    images_dir = data_dir / "images"
    
    print("="*60)
    print("改进的帧稳定性分析")
    print("="*60)
    
    # 按场景分组
    scenes = defaultdict(list)
    for img_file in sorted(images_dir.glob("*.png")):
        name = img_file.stem
        parts = name.split("_")
        step_idx = None
        for i, p in enumerate(parts):
            if p.startswith("step"):
                step_idx = i
                break
        if step_idx:
            scene = "_".join(parts[:step_idx+1])
        else:
            scene = "_".join(parts[:2])
        offset = int(name.split("_off")[1].split("_")[0])
        scenes[scene].append((offset, img_file))
    
    all_stable = 0
    all_animation = 0
    
    for scene, files in sorted(scenes.items())[:5]:
        print(f"\n场景: {scene}")
        
        # 读取帧
        frames = {}
        for offset, img_file in files:
            img = cv2.imread(str(img_file))
            if img is not None:
                frames[offset] = img
        
        # 全局稳定性分析
        stable, animation, scores = find_stable_frames(frames)
        
        print(f"  稳定性得分:")
        for off in sorted(scores.keys()):
            marker = "✅ 稳定" if off in stable else "⚠️ 动画"
            print(f"    off{off:3d}: {scores[off]:6.1f} {marker}")
        
        all_stable += len(stable)
        all_animation += len(animation)
    
    print("\n" + "="*60)
    print(f"总计: 稳定帧 {all_stable}, 动画帧 {all_animation}")
    print("="*60)
    
    print("\n建议的新 save_offsets 配置：")
    print("  --save-offsets -30 -25 -20 -15 -10 -5")
    print("  原因：更大范围，可以找到真正的稳定帧")


if __name__ == "__main__":
    main()