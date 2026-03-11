#!/usr/bin/env python3
"""
训练数据清洗脚本：动画帧过滤

功能：
1. 读取一个事件保存的所有帧（不同offset）
2. 全局稳定性分析
3. 过滤动画帧，保留最稳定的帧
4. 输出清洗后的数据

Usage:
    python clean_animation_frames.py --data-dir grounding_data_local_v6 --output-dir grounding_data_local_v6_cleaned
"""

import cv2
import numpy as np
import json
import shutil
import argparse
from pathlib import Path
from collections import defaultdict


def compute_stability_scores(frames: dict) -> dict:
    """
    计算每帧的全局稳定性得分
    
    方法：计算每帧与其他所有帧的平均差异
    得分越低 = 越稳定
    """
    offsets = sorted(frames.keys())
    n = len(offsets)
    
    if n < 2:
        return {offsets[0]: 0} if n == 1 else {}
    
    # 计算差异矩阵
    diff_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            diff = cv2.absdiff(frames[offsets[i]], frames[offsets[j]]).mean()
            diff_matrix[i, j] = diff
            diff_matrix[j, i] = diff
    
    # 计算每个帧的稳定性得分
    scores = {}
    for i, offset in enumerate(offsets):
        scores[offset] = diff_matrix[i].mean()
    
    return scores


def select_stable_frames(frames: dict, method: str = "best_n", n: int = 2, threshold_factor: float = 0.7) -> list:
    """
    选择稳定帧
    
    Args:
        frames: {offset: image}
        method: 选择方法
            - "best_n": 选择稳定性最好的n帧
            - "threshold": 选择稳定性得分低于阈值的帧
        n: 选择帧数（method="best_n"时）
        threshold_factor: 阈值因子（method="threshold"时）
    
    Returns:
        选择的offset列表
    """
    scores = compute_stability_scores(frames)
    
    if not scores:
        return []
    
    if method == "best_n":
        # 选择稳定性最好的n帧
        sorted_offsets = sorted(scores.keys(), key=lambda x: scores[x])
        return sorted_offsets[:n]
    
    elif method == "threshold":
        # 选择稳定性得分低于阈值的帧
        min_score = min(scores.values())
        max_score = max(scores.values())
        threshold = min_score + (max_score - min_score) * threshold_factor
        return [off for off, score in scores.items() if score <= threshold]
    
    return []


def clean_data(data_dir: str, output_dir: str, method: str = "best_n", n: int = 2):
    """
    清洗数据：过滤动画帧
    
    Args:
        data_dir: 原始数据目录
        output_dir: 输出目录
        method: 选择方法
        n: 每个事件保留的帧数
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    images_dir = data_dir / "images"
    output_images_dir = output_dir / "images"
    output_images_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载标注
    with open(data_dir / "annotations_coco.json") as f:
        coco = json.load(f)
    
    id_to_img = {img["id"]: img for img in coco["images"]}
    
    # 按事件分组
    events = defaultdict(list)
    for img_info in coco["images"]:
        name = img_info["file_name"]
        parts = name.split("_")
        # 找到step部分
        step_idx = None
        for i, p in enumerate(parts):
            if p.startswith("step"):
                step_idx = i
                break
        if step_idx:
            event_key = "_".join(parts[:step_idx+1])  # collect_wood_step00017
        else:
            event_key = "_".join(parts[:2])
        offset = int(name.split("_off")[1].split("_")[0])
        events[event_key].append((offset, img_info))
    
    print("="*60)
    print("动画帧过滤清洗")
    print("="*60)
    print(f"原始数据: {len(coco['images'])} 张图片, {len(events)} 个事件")
    print(f"过滤方法: {method}, 每事件保留 {n} 帧")
    print()
    
    # 处理每个事件
    cleaned_images = []
    cleaned_annotations = []
    stats = {"total": 0, "kept": 0, "filtered": 0}
    
    for event_key, files in sorted(events.items()):
        # 读取帧
        frames = {}
        for offset, img_info in files:
            img_path = images_dir / img_info["file_name"]
            img = cv2.imread(str(img_path))
            if img is not None:
                frames[offset] = (img, img_info)
        
        if not frames:
            continue
        
        # 计算稳定性得分
        scores = compute_stability_scores({off: f[0] for off, f in frames.items()})
        
        # 选择稳定帧
        selected_offsets = select_stable_frames({off: f[0] for off, f in frames.items()}, method=method, n=n)
        
        # 输出分析结果
        print(f"\n事件: {event_key}")
        print(f"  原始帧数: {len(frames)}")
        print(f"  稳定性得分:")
        for off in sorted(scores.keys()):
            marker = "✅ 保留" if off in selected_offsets else "❌ 过滤"
            print(f"    off={off:3d}: {scores[off]:6.1f} {marker}")
        
        # 复制保留的帧
        for offset in selected_offsets:
            img, img_info = frames[offset]
            
            # 复制图片
            src_path = images_dir / img_info["file_name"]
            dst_path = output_images_dir / img_info["file_name"]
            shutil.copy(src_path, dst_path)
            
            # 更新ID
            new_id = len(cleaned_images) + 1
            img_info_copy = img_info.copy()
            img_info_copy["id"] = new_id
            cleaned_images.append(img_info_copy)
            
            # 更新标注
            for ann in coco["annotations"]:
                if ann["image_id"] == img_info["id"]:
                    ann_copy = ann.copy()
                    ann_copy["image_id"] = new_id
                    ann_copy["id"] = len(cleaned_annotations) + 1
                    cleaned_annotations.append(ann_copy)
                    break
            
            stats["kept"] += 1
        
        stats["filtered"] += len(frames) - len(selected_offsets)
        stats["total"] += len(frames)
    
    # 保存清洗后的标注
    cleaned_coco = {
        "info": coco.get("info", {}),
        "categories": coco["categories"],
        "images": cleaned_images,
        "annotations": cleaned_annotations
    }
    
    with open(output_dir / "annotations_coco.json", "w") as f:
        json.dump(cleaned_coco, f, indent=2)
    
    # 统计
    print("\n" + "="*60)
    print("清洗结果")
    print("="*60)
    print(f"原始图片: {stats['total']}")
    print(f"保留图片: {stats['kept']} ({100*stats['kept']/stats['total']:.1f}%)")
    print(f"过滤图片: {stats['filtered']} ({100*stats['filtered']/stats['total']:.1f}%)")
    print(f"输出目录: {output_dir}")
    
    return cleaned_coco


def analyze_event(data_dir: str, event_key: str):
    """
    分析单个事件的所有帧，用于验证清洗效果
    """
    data_dir = Path(data_dir)
    images_dir = data_dir / "images"
    
    # 加载标注
    with open(data_dir / "annotations_coco.json") as f:
        coco = json.load(f)
    
    id_to_img = {img["id"]: img for img in coco["images"]}
    
    # 找到该事件的所有帧
    frames = {}
    for img_info in coco["images"]:
        name = img_info["file_name"]
        if event_key in name:
            offset = int(name.split("_off")[1].split("_")[0])
            img_path = images_dir / img_info["file_name"]
            img = cv2.imread(str(img_path))
            if img is not None:
                frames[offset] = (img, img_info)
    
    if not frames:
        print(f"未找到事件: {event_key}")
        return
    
    print(f"\n事件分析: {event_key}")
    print(f"帧数: {len(frames)}")
    
    # 计算稳定性
    scores = compute_stability_scores({off: f[0] for off, f in frames.items()})
    
    print("\n稳定性分析:")
    sorted_offsets = sorted(scores.keys(), key=lambda x: scores[x])
    for i, off in enumerate(sorted_offsets):
        rank = i + 1
        score = scores[off]
        img_info = frames[off][1]
        label = ""
        for ann in coco["annotations"]:
            if ann["image_id"] == img_info["id"]:
                for cat in coco["categories"]:
                    if cat["id"] == ann["category_id"]:
                        label = cat["name"]
                break
        print(f"  #{rank}: off={off:3d}, 得分={score:6.1f}, 标签={label}")
    
    # 显示最稳定的帧
    best_offset = sorted_offsets[0]
    print(f"\n最稳定的帧: off={best_offset}")
    
    return scores


def main():
    parser = argparse.ArgumentParser(description="训练数据清洗：动画帧过滤")
    parser.add_argument("--data-dir", required=True, help="原始数据目录")
    parser.add_argument("--output-dir", default=None, help="输出目录")
    parser.add_argument("--method", default="best_n", choices=["best_n", "threshold"], help="选择方法")
    parser.add_argument("--n", type=int, default=2, help="每个事件保留的帧数")
    parser.add_argument("--analyze-event", type=str, default=None, help="分析单个事件")
    args = parser.parse_args()
    
    if args.analyze_event:
        analyze_event(args.data_dir, args.analyze_event)
    elif args.output_dir:
        clean_data(args.data_dir, args.output_dir, method=args.method, n=args.n)
    else:
        parser.error("--output-dir is required when not using --analyze-event")


if __name__ == "__main__":
    main()