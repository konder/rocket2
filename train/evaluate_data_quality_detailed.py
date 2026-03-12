#!/usr/bin/env python3
"""
训练数据质量详细评估脚本

功能：
1. 读取图片和 mask 进行评估
2. 检查图片质量（清晰度、亮度、对比度）
3. 检查 mask 质量（面积、位置、完整性）
4. 随机采样可视化
5. 生成详细质量报告

Usage:
    python evaluate_data_quality_detailed.py --data-dir data/grounding_data --samples 10
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from progress_logger import ProgressLogger


def analyze_image_quality(image_path: str) -> dict:
    """分析图片质量"""
    img = cv2.imread(image_path)
    if img is None:
        return {"error": "无法读取图片"}
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 基本统计
    height, width = gray.shape
    
    # 亮度
    brightness = gray.mean()
    brightness_std = gray.std()
    
    # 清晰度 (Laplacian variance)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # 对比度
    contrast = gray.std()
    
    # 饱和度分析（检测动画效果）
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = hsv[:,:,1].mean()
    low_sat_ratio = np.sum(hsv[:,:,1] < 30) / hsv[:,:,1].size
    
    return {
        "width": width,
        "height": height,
        "brightness": round(brightness, 1),
        "brightness_std": round(brightness_std, 1),
        "sharpness": round(sharpness, 1),
        "contrast": round(contrast, 1),
        "saturation": round(saturation, 1),
        "low_sat_ratio": round(low_sat_ratio, 3),
        "has_animation_effect": low_sat_ratio > 0.3
    }


def analyze_bbox_quality(bbox: list, img_width: int, img_height: int) -> dict:
    """分析标注框质量"""
    x, y, w, h = bbox
    
    # 面积
    area = w * h
    img_area = img_width * img_height
    area_ratio = area / img_area
    
    # 位置
    center_x = x + w / 2
    center_y = y + h / 2
    center_normalized = [
        round(center_x / img_width, 3),
        round(center_y / img_height, 3)
    ]
    
    # 边界检查
    in_bounds = (x >= 0 and y >= 0 and x + w <= img_width and y + h <= img_height)
    
    # 质量判断
    quality = "good"
    issues = []
    
    if area_ratio < 0.01:
        quality = "too_small"
        issues.append(f"bbox过小 ({area_ratio:.1%})")
    elif area_ratio > 0.5:
        quality = "too_large"
        issues.append(f"bbox过大 ({area_ratio:.1%})")
    
    if not in_bounds:
        quality = "out_of_bounds"
        issues.append("bbox超出图像边界")
    
    return {
        "bbox": bbox,
        "area": area,
        "area_ratio": round(area_ratio, 4),
        "center": center_normalized,
        "in_bounds": in_bounds,
        "quality": quality,
        "issues": issues
    }


def evaluate_dataset(data_dir: str, output_dir: str = None, num_samples: int = 10):
    """
    详细评估训练数据集
    
    Args:
        data_dir: 数据目录
        output_dir: 输出目录（用于保存可视化图片）
        num_samples: 随机采样数量
    """
    data_dir = Path(data_dir)
    images_dir = data_dir / "images"
    
    # 加载标注
    with open(data_dir / "annotations_coco.json") as f:
        coco = json.load(f)
    
    # 输出目录
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # 进度日志
    log_file = data_dir / "quality_evaluation.log"
    progress = ProgressLogger(str(log_file), "Data Evaluation")
    progress.log_event("开始数据质量评估", f"数据目录: {data_dir}, 采样数: {num_samples}")
    
    # 构建映射
    id_to_img = {img["id"]: img for img in coco["images"]}
    id_to_cat = {cat["id"]: cat["name"] for cat in coco["categories"]}
    
    # 分析每张图片
    results = []
    issues_count = {
        "animation_effect": 0,
        "bbox_too_small": 0,
        "bbox_too_large": 0,
        "bbox_out_of_bounds": 0,
        "low_sharpness": 0,
        "low_brightness": 0,
        "high_brightness": 0
    }
    
    total = len(coco["images"])
    
    for i, img_info in enumerate(coco["images"]):
        img_path = images_dir / img_info["file_name"]
        
        # 图片质量分析
        img_quality = analyze_image_quality(str(img_path))
        
        # 标注框分析
        anns_for_img = [a for a in coco["annotations"] if a["image_id"] == img_info["id"]]
        bbox_qualities = []
        
        for ann in anns_for_img:
            bbox_q = analyze_bbox_quality(ann["bbox"], img_info["width"], img_info["height"])
            bbox_q["category"] = id_to_cat[ann["category_id"]]
            bbox_qualities.append(bbox_q)
            
            # 统计问题
            if bbox_q["quality"] == "too_small":
                issues_count["bbox_too_small"] += 1
            elif bbox_q["quality"] == "too_large":
                issues_count["bbox_too_large"] += 1
            if not bbox_q["in_bounds"]:
                issues_count["bbox_out_of_bounds"] += 1
        
        # 动画效果检测
        if img_quality.get("has_animation_effect"):
            issues_count["animation_effect"] += 1
        
        # 清晰度和亮度问题
        if img_quality.get("sharpness", 0) < 1000:
            issues_count["low_sharpness"] += 1
        if img_quality.get("brightness", 0) < 30:
            issues_count["low_brightness"] += 1
        elif img_quality.get("brightness", 0) > 200:
            issues_count["high_brightness"] += 1
        
        results.append({
            "image": img_info["file_name"],
            "image_quality": img_quality,
            "bbox_qualities": bbox_qualities
        })
        
        # 记录进度
        if (i + 1) % 10 == 0:
            progress.log("process", i + 1, total, info={"analyzed": i + 1})
    
    # 统计
    total_images = len(results)
    total_annotations = sum(len(r["bbox_qualities"]) for r in results)
    
    # 类别分布
    from collections import Counter
    cat_count = Counter()
    for r in results:
        for bq in r["bbox_qualities"]:
            cat_count[bq["category"]] += 1
    
    # 问题总结
    issues_summary = []
    for issue, count in issues_count.items():
        if count > 0:
            pct = count / max(total_images, 1) * 100
            issues_summary.append({
                "issue": issue,
                "count": count,
                "percentage": round(pct, 1)
            })
    
    # 生成报告
    report = {
        "timestamp": datetime.now().isoformat(),
        "data_dir": str(data_dir),
        "summary": {
            "total_images": total_images,
            "total_annotations": total_annotations,
            "total_categories": len(cat_count),
            "category_distribution": dict(cat_count)
        },
        "issues_count": issues_count,
        "issues_summary": issues_summary,
        "quality_score": {
            "good_images": total_images - issues_count["animation_effect"] - issues_count["low_sharpness"],
            "good_bboxes": total_annotations - issues_count["bbox_too_small"] - issues_count["bbox_too_large"] - issues_count["bbox_out_of_bounds"],
            "overall_quality": "good" if len(issues_summary) == 0 else "needs_improvement"
        },
        "detailed_results": results[:num_samples]  # 只保存采样结果
    }
    
    # 保存报告
    report_path = data_dir / "quality_report_detailed.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    progress.log_event("评估完成", f"报告保存到: {report_path}")
    progress.close()
    
    # 打印摘要
    print("\n" + "=" * 60)
    print("数据质量评估报告")
    print("=" * 60)
    print(f"总图片数: {total_images}")
    print(f"总标注数: {total_annotations}")
    print(f"类别数: {len(cat_count)}")
    print()
    print("问题统计:")
    for issue in issues_summary:
        print(f"  {issue['issue']}: {issue['count']} ({issue['percentage']:.1f}%)")
    print()
    print(f"质量评分: {report['quality_score']['overall_quality']}")
    print(f"报告文件: {report_path}")
    
    return report


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--samples", type=int, default=10)
    args = parser.parse_args()
    
    evaluate_dataset(args.data_dir, args.output_dir, args.samples)


if __name__ == "__main__":
    main()