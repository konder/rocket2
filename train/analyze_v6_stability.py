#!/usr/bin/env python3
"""分析 v6 数据的帧稳定性"""

import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict


def main():
    data_dir = Path("/Users/nanzhang/rocket2/grounding_data_local_v6")
    images_dir = data_dir / "images"
    
    print("="*60)
    print("v6 数据帧稳定性分析（更大范围：-30 到 -5）")
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
    
    # 统计每个offset被选为稳定的次数
    offset_stable_count = defaultdict(int)
    offset_total_count = defaultdict(int)
    
    for scene, files in sorted(scenes.items()):
        # 读取帧
        frames = {}
        for offset, img_file in files:
            img = cv2.imread(str(img_file))
            if img is not None:
                frames[offset] = img
                offset_total_count[offset] += 1
        
        # 计算全局稳定性
        offsets = sorted(frames.keys())
        if len(offsets) < 2:
            continue
        
        # 计算差异矩阵
        n = len(offsets)
        diff_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                diff = cv2.absdiff(frames[offsets[i]], frames[offsets[j]]).mean()
                diff_matrix[i, j] = diff
                diff_matrix[j, i] = diff
        
        # 找最稳定的帧
        scores = {}
        for i, off in enumerate(offsets):
            scores[off] = diff_matrix[i].mean()
        
        # 选择稳定性最好的帧
        best_offset = min(scores, key=scores.get)
        offset_stable_count[best_offset] += 1
    
    print("\n=== 每个offset被选为最稳定的次数 ===")
    for offset in sorted(offset_stable_count.keys()):
        stable = offset_stable_count[offset]
        total = offset_total_count[offset]
        pct = 100 * stable / len(scenes) if len(scenes) > 0 else 0
        print(f"  off={offset:3d}: {stable:3d}次 ({pct:5.1f}%)")
    
    print("\n=== 结论 ===")
    best_offset = max(offset_stable_count, key=offset_stable_count.get)
    print(f"最常被选为稳定的offset: {best_offset}")
    print(f"建议使用: --save-offsets {best_offset}")


if __name__ == "__main__":
    main()
