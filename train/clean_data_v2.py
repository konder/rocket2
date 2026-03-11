#!/usr/bin/env python3
"""
训练数据清洗脚本：动画帧过滤 + SAM bbox过滤

过滤流程：
1. 动画帧过滤（全局稳定性分析）
2. SAM bbox过滤（太大或太小的框）

Usage:
    python clean_data_v2.py --data-dir grounding_data_local_v8 --output-dir grounding_data_local_v8_cleaned
"""

import cv2
import numpy as np
import json
import shutil
import argparse
from pathlib import Path
from collections import defaultdict


def compute_stability_scores(frames: dict) -> dict:
    """计算每帧的全局稳定性得分"""
    offsets = sorted(frames.keys())
    n = len(offsets)
    
    if n < 2:
        return {offsets[0]: 0} if n == 1 else {}
    
    diff_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            diff = cv2.absdiff(frames[offsets[i]], frames[offsets[j]]).mean()
            diff_matrix[i, j] = diff
            diff_matrix[j, i] = diff
    
    scores = {}
    for i, offset in enumerate(offsets):
        scores[offset] = diff_matrix[i].mean()
    
    return scores


def check_bbox_quality(bbox: list, img_w: int, img_h: int, 
                        min_area_ratio: float = 0.01, 
                        max_area_ratio: float = 0.5,
                        min_aspect: float = 0.2,
                        max_aspect: float = 5.0) -> tuple:
    """
    检查bbox质量
    
    Returns:
        (is_valid, reason)
    """
    x, y, w, h = bbox
    area = w * h
    img_area = img_w * img_h
    area_ratio = area / img_area
    
    # 检查面积
    if area_ratio < min_area_ratio:
        return False, f"bbox太小 ({area_ratio:.1%} < {min_area_ratio:.1%})"
    if area_ratio > max_area_ratio:
        return False, f"bbox太大 ({area_ratio:.1%} > {max_area_ratio:.1%})"
    
    # 检查宽高比
    aspect = w / h if h > 0 else 0
    if aspect < min_aspect or aspect > max_aspect:
        return False, f"宽高比异常 ({aspect:.2f})"
    
    # 检查边界
    if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
        return False, "bbox超出图像边界"
    
    return True, "OK"


def clean_data(data_dir: str, output_dir: str, 
               keep_best_n: int = 3,
               min_area_ratio: float = 0.01,
               max_area_ratio: float = 0.5):
    """
    清洗数据
    
    Args:
        data_dir: 原始数据目录
        output_dir: 输出目录
        keep_best_n: 每个事件保留的最佳帧数
        min_area_ratio: bbox最小面积比例
        max_area_ratio: bbox最大面积比例
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    images_dir = data_dir / "images"
    output_images_dir = output_dir / "images"
    output_images_dir.mkdir(parents=True, exist_ok=True)
    
    with open(data_dir / "annotations_coco.json") as f:
        coco = json.load(f)
    
    id_to_img = {img["id"]: img for img in coco["images"]}
    id_to_ann = {ann["image_id"]: ann for ann in coco["annotations"]}
    id_to_cat = {cat["id"]: cat["name"] for cat in coco["categories"]}
    
    # 按事件分组
    events = defaultdict(list)
    for img_info in coco["images"]:
        name = img_info["file_name"]
        parts = name.split("_")
        step_idx = None
        for i, p in enumerate(parts):
            if p.startswith("step"):
                step_idx = i
                break
        if step_idx:
            event_key = "_".join(parts[:step_idx+1])
        else:
            event_key = "_".join(parts[:2])
        offset = int(name.split("_off")[1].split("_")[0])
        events[event_key].append((offset, img_info))
    
    print("="*60)
    print("数据清洗（动画帧 + SAM bbox过滤）")
    print("="*60)
    print(f"原始数据: {len(coco['images'])} 张图片, {len(events)} 个事件")
    print(f"配置: 每事件保留 {keep_best_n} 帧")
    print(f"bbox过滤: 面积比例 {min_area_ratio:.1%} ~ {max_area_ratio:.1%}")
    print()
    
    cleaned_images = []
    cleaned_annotations = []
    stats = {
        "total": 0, 
        "animation_filtered": 0, 
        "bbox_filtered": 0,
        "kept": 0
    }
    
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
        
        # Step 1: 计算稳定性得分
        scores = compute_stability_scores({off: f[0] for off, f in frames.items()})
        
        # Step 2: 按稳定性排序
        sorted_offsets = sorted(scores.keys(), key=lambda x: scores[x])
        
        # Step 3: 选择最佳帧并检查bbox质量
        selected = []
        for offset in sorted_offsets:
            img, img_info = frames[offset]
            ann = id_to_ann.get(img_info["id"])
            
            if ann:
                # 检查bbox质量
                is_valid, reason = check_bbox_quality(
                    ann["bbox"], 
                    img_info["width"], 
                    img_info["height"],
                    min_area_ratio=min_area_ratio,
                    max_area_ratio=max_area_ratio
                )
                
                if is_valid:
                    selected.append((offset, img_info, ann, scores[offset]))
                    if len(selected) >= keep_best_n:
                        break
                else:
                    stats["bbox_filtered"] += 1
        
        # 输出分析结果
        print(f"\n事件: {event_key}")
        print(f"  原始帧数: {len(frames)}")
        print(f"  稳定性排序:")
        for i, off in enumerate(sorted_offsets[:5]):
            marker = "✅" if any(s[0] == off for s in selected) else "❌"
            score = scores[off]
            ann = id_to_ann.get(frames[off][1]["id"])
            if ann:
                w, h = ann["bbox"][2], ann["bbox"][3]
                area_ratio = w * h / (img_info["width"] * img_info["height"])
                print(f"    #{i+1}: off={off:3d}, 得分={score:5.1f}, bbox面积={area_ratio:.1%} {marker}")
        
        print(f"  保留: {len(selected)} 帧")
        
        # 复制保留的帧
        for offset, img_info, ann, score in selected:
            src_path = images_dir / img_info["file_name"]
            dst_path = output_images_dir / img_info["file_name"]
            shutil.copy(src_path, dst_path)
            
            new_id = len(cleaned_images) + 1
            img_info_copy = img_info.copy()
            img_info_copy["id"] = new_id
            cleaned_images.append(img_info_copy)
            
            ann_copy = ann.copy()
            ann_copy["image_id"] = new_id
            ann_copy["id"] = len(cleaned_annotations) + 1
            cleaned_annotations.append(ann_copy)
            
            stats["kept"] += 1
        
        stats["animation_filtered"] += len(frames) - len(selected) - stats["bbox_filtered"]
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
    print(f"动画帧过滤: {stats['animation_filtered']}")
    print(f"bbox过滤: {stats['bbox_filtered']}")
    print(f"保留图片: {stats['kept']} ({100*stats['kept']/stats['total']:.1f}%)")
    print(f"输出目录: {output_dir}")
    
    return cleaned_coco


def main():
    parser = argparse.ArgumentParser(description="数据清洗：动画帧 + SAM bbox过滤")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--keep-best-n", type=int, default=3)
    parser.add_argument("--min-area-ratio", type=float, default=0.01)
    parser.add_argument("--max-area-ratio", type=float, default=0.5)
    args = parser.parse_args()
    
    clean_data(
        args.data_dir, 
        args.output_dir,
        keep_best_n=args.keep_best_n,
        min_area_ratio=args.min_area_ratio,
        max_area_ratio=args.max_area_ratio
    )


if __name__ == "__main__":
    main()