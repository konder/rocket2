#!/usr/bin/env python3
"""Convert COCO format to ODVG format for open_groundingdino training."""

import json
import argparse
from pathlib import Path


def coco_to_xyxy(bbox):
    """Convert COCO bbox [x, y, width, height] to [x1, y1, x2, y2]."""
    x, y, w, h = bbox
    return [round(x, 2), round(y, 2), round(x + w, 2), round(y + h, 2)]


def convert_coco_to_odvg(input_json, output_jsonl, output_labelmap):
    """Convert COCO format annotations to ODVG JSONL format."""
    
    with open(input_json, 'r') as f:
        coco = json.load(f)
    
    # Build mappings
    id_to_img = {img["id"]: img for img in coco["images"]}
    id_to_cat = {cat["id"]: cat["name"] for cat in coco["categories"]}
    
    # Group annotations by image
    from collections import defaultdict
    img_to_anns = defaultdict(list)
    for ann in coco["annotations"]:
        img_to_anns[ann["image_id"]].append(ann)
    
    # Build label map (0-indexed)
    categories = sorted(coco["categories"], key=lambda x: x["id"])
    label_map = {}
    cat_id_to_label = {}
    for idx, cat in enumerate(categories):
        label_map[str(idx)] = cat["name"]
        cat_id_to_label[cat["id"]] = idx
    
    # Convert to ODVG format
    odvg_records = []
    for img_id, anns in img_to_anns.items():
        img_info = id_to_img[img_id]
        
        instances = []
        for ann in anns:
            bbox_xyxy = coco_to_xyxy(ann["bbox"])
            label = cat_id_to_label[ann["category_id"]]
            category = id_to_cat[ann["category_id"]]
            instances.append({
                "bbox": bbox_xyxy,
                "label": label,
                "category": category
            })
        
        odvg_records.append({
            "filename": img_info["file_name"],
            "height": img_info["height"],
            "width": img_info["width"],
            "detection": {
                "instances": instances
            }
        })
    
    # Write ODVG JSONL
    with open(output_jsonl, 'w') as f:
        for record in odvg_records:
            f.write(json.dumps(record) + '\n')
    
    # Write label map
    with open(output_labelmap, 'w') as f:
        json.dump(label_map, f, indent=2)
    
    print(f"Converted {len(odvg_records)} images")
    print(f"ODVG annotations: {output_jsonl}")
    print(f"Label map: {output_labelmap}")
    print(f"Categories: {len(label_map)}")
    
    return label_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input COCO JSON")
    parser.add_argument("--output", required=True, help="Output ODVG JSONL")
    parser.add_argument("--labelmap", required=True, help="Output label map JSON")
    args = parser.parse_args()
    
    convert_coco_to_odvg(args.input, args.output, args.labelmap)


if __name__ == "__main__":
    main()