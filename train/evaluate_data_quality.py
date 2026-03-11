#!/usr/bin/env python3
"""
训练数据质量自动评估脚本

按照以下标准评估：
1. 图片中不能出现任何游戏画面外的标记框等信息，只保留原始的游戏画面
2. mask文件中的标注bbox应该贴合目标block的边框，面积应该小于完整画面的2/3
3. 应该至少每个block包含一张图片
4. 标注目标block应该亮度合适，清晰

Usage:
    python evaluate_data_quality.py --data-dir /path/to/data
"""

import json
import cv2
import numpy as np
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime


class DataQualityEvaluator:
    """训练数据质量评估器"""
    
    # 质量标准阈值
    BBOX_MAX_AREA_RATIO = 2/3  # 标注框最大面积比例
    MIN_BRIGHTNESS = 30        # 最小亮度
    MAX_BRIGHTNESS = 200       # 最大亮度
    MIN_SHARPNESS = 1000       # 最小清晰度
    EDGE_MARKER_THRESHOLD = 0.01  # 边缘标记检测阈值
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.coco_json = self.data_dir / "annotations_coco.json"
        self.images_dir = self.data_dir / "images"
        
        with open(self.coco_json) as f:
            self.coco = json.load(f)
        
        self.id_to_cat = {cat["id"]: cat["name"] for cat in self.coco["categories"]}
        self.id_to_img = {img["id"]: img for img in self.coco["images"]}
        self.issues = defaultdict(list)
        self.stats = {}
    
    def evaluate_all(self) -> dict:
        """执行所有质量检查"""
        print("=" * 60)
        print("训练数据质量自动评估")
        print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        # 逐个评估
        self._check_external_markers()
        self._check_bbox_quality()
        self._check_category_coverage()
        self._check_image_quality()
        self._check_destruction_animation()
        
        # 生成总结
        self._generate_summary()
        
        return self._get_report()
    
    def _check_external_markers(self):
        """标准1：检查游戏画面外标记"""
        print("\n【标准1】游戏画面外无标记框")
        
        marker_issues = 0
        for img_info in self.coco["images"]:
            img_path = self.images_dir / img_info["file_name"]
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            H, W = img.shape[:2]
            border = 5
            
            # 检查边缘是否有异常颜色（绿色标注残留）
            edges = [
                img[:border, :],   # top
                img[-border:, :],  # bottom
                img[:, :border],   # left
                img[:, -border:],  # right
            ]
            
            for edge in edges:
                # 检测绿色像素 (G > 200, R < 50, B < 50)
                green_mask = (edge[:,:,1] > 200) & (edge[:,:,0] < 50) & (edge[:,:,2] < 50)
                green_ratio = np.sum(green_mask) / edge.size * 3
                
                if green_ratio > self.EDGE_MARKER_THRESHOLD:
                    self.issues["边缘标记残留"].append(img_info["file_name"])
                    marker_issues += 1
                    break
        
        if marker_issues == 0:
            print("  ✅ 所有图片无外部标记")
        else:
            print(f"  ❌ 有 {marker_issues} 张图片边缘有标记残留")
        
        self.stats["external_markers"] = marker_issues
    
    def _check_bbox_quality(self):
        """标准2：检查标注框质量"""
        print("\n【标准2】标注框贴合目标，面积<2/3画面")
        
        area_issues = 0
        fit_issues = 0
        
        for ann in self.coco["annotations"]:
            img_info = self.id_to_img[ann["image_id"]]
            img_path = self.images_dir / img_info["file_name"]
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            H, W = img.shape[:2]
            x, y, w, h = ann["bbox"]
            
            # 检查面积
            bbox_area = w * h
            img_area = W * H
            area_ratio = bbox_area / img_area
            
            if area_ratio > self.BBOX_MAX_AREA_RATIO:
                self.issues["标注框过大"].append({
                    "file": img_info["file_name"],
                    "ratio": f"{area_ratio:.1%}"
                })
                area_issues += 1
            
            # 检查贴合度（边缘检测）
            roi = img[int(y):int(y+h), int(x):int(x+w)]
            if roi.size > 0:
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(roi_gray, 50, 150)
                edge_ratio = np.sum(edges > 0) / (w * h)
                
                if edge_ratio < 0.005:
                    self.issues["标注不贴合"].append(img_info["file_name"])
                    fit_issues += 1
        
        if area_issues == 0:
            print("  ✅ 所有标注框面积合理")
        else:
            print(f"  ❌ 有 {area_issues} 张图片标注框过大")
        
        if fit_issues == 0:
            print("  ✅ 标注框贴合目标")
        else:
            print(f"  ⚠️ 有 {fit_issues} 张图片标注可能不贴合")
        
        self.stats["bbox_area_issues"] = area_issues
        self.stats["bbox_fit_issues"] = fit_issues
    
    def _check_category_coverage(self):
        """标准3：检查类别覆盖"""
        print("\n【标准3】每个block至少一张图片")
        
        cat_count = Counter(ann["category_id"] for ann in self.coco["annotations"])
        all_cats = set(self.id_to_cat.keys())
        present_cats = set(cat_count.keys())
        missing_cats = all_cats - present_cats
        
        print(f"  定义类别数: {len(all_cats)}")
        print(f"  有数据类别数: {len(present_cats)}")
        
        if missing_cats:
            missing_names = [self.id_to_cat[cid] for cid in missing_cats]
            print(f"  ❌ 缺少 {len(missing_cats)} 类block数据")
            print(f"     缺失: {', '.join(missing_names[:5])}{'...' if len(missing_names) > 5 else ''}")
        else:
            print("  ✅ 所有block类别都有数据")
        
        self.stats["total_categories"] = len(all_cats)
        self.stats["present_categories"] = len(present_cats)
        self.stats["missing_categories"] = list(missing_cats)
    
    def _check_image_quality(self):
        """标准4：检查图像质量"""
        print("\n【标准4】目标block清晰")
        
        dark_issues = 0
        bright_issues = 0
        blur_issues = 0
        
        for img_info in self.coco["images"]:
            img_path = self.images_dir / img_info["file_name"]
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            brightness = gray.mean()
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            if brightness < self.MIN_BRIGHTNESS:
                self.issues["图像过暗"].append(img_info["file_name"])
                dark_issues += 1
            elif brightness > self.MAX_BRIGHTNESS:
                self.issues["图像过亮"].append(img_info["file_name"])
                bright_issues += 1
            
            if sharpness < self.MIN_SHARPNESS:
                self.issues["图像模糊"].append(img_info["file_name"])
                blur_issues += 1
        
        if dark_issues == 0 and bright_issues == 0:
            print("  ✅ 图像亮度合格")
        else:
            if dark_issues:
                print(f"  ❌ 有 {dark_issues} 张图片过暗")
            if bright_issues:
                print(f"  ❌ 有 {bright_issues} 张图片过亮")
        
        if blur_issues == 0:
            print("  ✅ 图像清晰度合格")
        else:
            print(f"  ❌ 有 {blur_issues} 张图片模糊")
        
        self.stats["dark_images"] = dark_issues
        self.stats["bright_images"] = bright_issues
        self.stats["blur_images"] = blur_issues
    
    def _check_destruction_animation(self):
        """标准4扩展：检查破坏动画效果"""
        print("\n【标准4扩展】无破坏动画污染")
        
        animation_issues = 0
        
        for ann in self.coco["annotations"]:
            img_info = self.id_to_img[ann["image_id"]]
            img_path = self.images_dir / img_info["file_name"]
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            # Minecraft破坏动画会有渐变叠加效果
            # 检查HSV饱和度分布，破坏动画会有异常
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            saturation = hsv[:,:,1]
            
            # 检查是否有大面积低饱和度区域（可能是破坏动画）
            low_sat_ratio = np.sum(saturation < 30) / saturation.size
            
            if low_sat_ratio > 0.3:
                self.issues["可能有破坏动画"].append(img_info["file_name"])
                animation_issues += 1
        
        if animation_issues == 0:
            print("  ✅ 未检测到破坏动画")
        else:
            print(f"  ⚠️ 有 {animation_issues} 张图片可能有破坏动画效果")
        
        self.stats["animation_issues"] = animation_issues
    
    def _generate_summary(self):
        """生成总结"""
        print("\n" + "=" * 60)
        
        total = len(self.coco["annotations"])
        valid = total - len(set(
            f for issues in self.issues.values() 
            for f in (issues if isinstance(issues[0], str) else [i["file"] for i in issues])
        ))
        
        print(f"总结: {valid}/{total} 图片通过所有标准")
        
        # 问题汇总
        if self.issues:
            print("\n问题汇总:")
            for issue_type, files in self.issues.items():
                print(f"  - {issue_type}: {len(files)} 张")
        else:
            print("\n✅ 所有图片通过质量检查")
        
        print("=" * 60)
        
        self.stats["total_images"] = len(self.coco["images"])
        self.stats["total_annotations"] = total
        self.stats["valid_count"] = valid
        self.stats["pass_rate"] = f"{100 * valid / total:.1f}%"
    
    def _get_report(self) -> dict:
        """生成报告字典"""
        return {
            "timestamp": datetime.now().isoformat(),
            "data_dir": str(self.data_dir),
            "statistics": self.stats,
            "issues": {k: v for k, v in self.issues.items()},
            "category_distribution": {
                self.id_to_cat[k]: v 
                for k, v in Counter(
                    ann["category_id"] for ann in self.coco["annotations"]
                ).items()
            }
        }
    
    def save_report(self, output_path: str = None):
        """保存报告"""
        if output_path is None:
            output_path = self.data_dir / "quality_report.json"
        
        report = self._get_report()
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n报告已保存: {output_path}")
        return report


def main():
    parser = argparse.ArgumentParser(description="训练数据质量自动评估")
    parser.add_argument(
        "--data-dir", 
        default="/Users/nanzhang/rocket2/grounding_data_local_v3",
        help="数据目录路径"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="报告输出路径"
    )
    args = parser.parse_args()
    
    evaluator = DataQualityEvaluator(args.data_dir)
    evaluator.evaluate_all()
    evaluator.save_report(args.output)


if __name__ == "__main__":
    main()