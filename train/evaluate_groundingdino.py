#!/usr/bin/env python3
"""
Run inference with GroundingDINO model (pretrained or fine-tuned) on Minecraft data.
Evaluate detection performance and save results.
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
import cv2
import numpy as np
from PIL import Image


def load_groundingdino_model(config_path, weights_path=None, device="mps"):
    """Load GroundingDINO model."""
    from groundingdino.models import build_model
    from groundingdino.util.slconfig import SLConfig
    from groundingdino.util.misc import clean_state_dict
    
    # Load config
    args = SLConfig.fromfile(config_path)
    args.device = device
    
    # Build model
    model = build_model(args)
    
    # Load weights
    if weights_path and os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location="cpu")
        # Handle both wrapped and unwrapped checkpoints
        if "model" in checkpoint:
            model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        else:
            model.load_state_dict(clean_state_dict(checkpoint), strict=False)
    else:
        # Use default pretrained weights
        hf_cache = Path.home() / ".cache/huggingface/hub/models--ShilongLiu--GroundingDINO/snapshots"
        snapshot_dir = list(hf_cache.glob("*"))[0] if hf_cache.exists() else None
        if snapshot_dir:
            weights_path = snapshot_dir / "groundingdino_swint_ogc.pth"
            checkpoint = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        else:
            raise FileNotFoundError("No weights found")
    
    model = model.to(device)
    model.eval()
    return model


def run_inference(model, image_path, text_prompt, box_threshold=0.35, text_threshold=0.25, device="mps"):
    """Run inference on a single image."""
    from groundingdino.util.inference import predict, load_image
    
    # Load and preprocess image
    image_source, image_tensor = load_image(image_path)
    
    boxes, logits, phrases = predict(
        model=model,
        image=image_tensor,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=device,
    )
    
    return boxes, logits, phrases


def evaluate_on_dataset(model, coco_json, images_dir, config, device="mps"):
    """Evaluate model on COCO-format dataset."""
    
    with open(coco_json, 'r') as f:
        coco = json.load(f)
    
    # Build mappings
    id_to_img = {img["id"]: img for img in coco["images"]}
    id_to_cat = {cat["id"]: cat["name"] for cat in coco["categories"]}
    
    # Group annotations by image
    from collections import defaultdict
    img_to_anns = defaultdict(list)
    for ann in coco["annotations"]:
        img_to_anns[ann["image_id"]].append(ann)
    
    results = []
    total_gt = 0
    total_pred = 0
    
    for img_id, anns in img_to_anns.items():
        img_info = id_to_img[img_id]
        img_path = os.path.join(images_dir, img_info["file_name"])
        
        # Get unique categories for this image
        categories = list(set(id_to_cat[ann["category_id"]] for ann in anns))
        prompt = " . ".join(categories) + " ."
        
        # Run inference
        boxes, logits, phrases = run_inference(model, img_path, prompt, device=device)
        
        if boxes is None:
            continue
        
        num_pred = len(boxes) if boxes is not None else 0
        num_gt = len(anns)
        
        results.append({
            "image": img_info["file_name"],
            "prompt": prompt,
            "gt_count": num_gt,
            "pred_count": num_pred,
            "categories": categories,
        })
        
        total_gt += num_gt
        total_pred += num_pred
    
    summary = {
        "total_images": len(results),
        "total_gt_boxes": total_gt,
        "total_pred_boxes": total_pred,
        "avg_pred_per_image": total_pred / max(len(results), 1),
        "precision_estimate": total_gt / max(total_pred, 1) if total_pred > 0 else 0,
    }
    
    return results, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-json", required=True, help="COCO format annotations")
    parser.add_argument("--images-dir", required=True, help="Images directory")
    parser.add_argument("--config", default="open_groundingdino/tools/GroundingDINO_SwinT_OGC.py")
    parser.add_argument("--weights", default=None, help="Fine-tuned weights (optional)")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--name", default="groundingdino", help="Model name for results")
    args = parser.parse_args()
    
    # Resolve config path
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path.home() / "rocket2" / args.config
    
    print(f"Loading model from: {config_path}")
    if args.weights:
        print(f"Using weights: {args.weights}")
    
    device = args.device if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    
    model = load_groundingdino_model(str(config_path), args.weights, device)
    
    print(f"Evaluating on: {args.coco_json}")
    results, summary = evaluate_on_dataset(model, args.coco_json, args.images_dir, args, device)
    
    output_data = {
        "model": f"{args.name} ({'fine-tuned' if args.weights else 'pretrained'})",
        "results": results,
        "summary": summary,
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n=== Results ===")
    print(f"Total images: {summary['total_images']}")
    print(f"Total GT boxes: {summary['total_gt_boxes']}")
    print(f"Total predicted: {summary['total_pred_boxes']}")
    print(f"Avg predictions/image: {summary['avg_pred_per_image']:.1f}")
    print(f"Precision estimate (GT/Pred): {summary['precision_estimate']:.2%}")
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()