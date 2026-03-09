"""
Fine-tune GroundingDINO on Minecraft block data (COCO format).

Freezes the text encoder (BERT) and backbone; only fine-tunes the
detection head and cross-modal fusion layers. This is the recommended
approach when the domain shift is primarily visual (Minecraft pixel art
style) rather than vocabulary.

Usage (called by finetune_groundingdino.sh):
    python run_finetune_groundingdino.py \\
        --coco-json   data/grounding_data/annotations_coco.json \\
        --images-dir  data/grounding_data/images \\
        --gdino-config  GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \\
        --gdino-weights /path/to/groundingdino_swint_ogc.pth \\
        --output-dir  data/grounding_data/checkpoints \\
        --epochs 10 --batch-size 4 --lr 1e-5
"""

import os
import sys
import json
import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import cv2
import numpy as np
from PIL import Image


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

class MinecraftGroundingDataset(Dataset):
    """
    Loads COCO-format annotations produced by collect_grounding_data.py.
    Returns (image_tensor, caption_str, boxes_tensor, labels_tensor).
    """

    def __init__(self, coco_json: str, images_dir: str, transform=None):
        with open(coco_json) as f:
            coco = json.load(f)

        self.images_dir = images_dir
        self.transform  = transform

        # Build id → image meta
        id_to_img = {img["id"]: img for img in coco["images"]}
        # id → category name
        id_to_cat = {cat["id"]: cat["name"] for cat in coco["categories"]}

        # Group annotations by image
        from collections import defaultdict
        img_to_anns = defaultdict(list)
        for ann in coco["annotations"]:
            img_to_anns[ann["image_id"]].append(ann)

        self.samples = []
        for img_id, anns in img_to_anns.items():
            img_meta = id_to_img[img_id]
            boxes   = []
            captions = []
            for ann in anns:
                x, y, w, h = ann["bbox"]
                # Normalize to [0,1]
                W, H = img_meta["width"], img_meta["height"]
                cx = (x + w / 2) / W
                cy = (y + h / 2) / H
                bw = w / W
                bh = h / H
                boxes.append([cx, cy, bw, bh])
                captions.append(id_to_cat[ann["category_id"]])

            self.samples.append({
                "file_name": img_meta["file_name"],
                "boxes":     boxes,      # list of [cx,cy,w,h] normalized
                "captions":  captions,   # list of str
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = os.path.join(self.images_dir, sample["file_name"])

        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        pil = Image.fromarray(img)
        if self.transform:
            img_tensor, _ = self.transform(pil, None)
        else:
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        boxes   = torch.tensor(sample["boxes"],   dtype=torch.float32)
        caption = " . ".join(set(sample["captions"])) + " ."

        return img_tensor, caption, boxes


def collate_fn(batch):
    images, captions, boxes = zip(*batch)
    return list(images), list(captions), list(boxes)


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #

def build_transform():
    from groundingdino.datasets import transforms as T
    return T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def freeze_modules(model):
    """Freeze backbone and text encoder; train only fusion + detection head."""
    frozen_prefixes = ["backbone", "bert"]
    for name, param in model.named_parameters():
        if any(name.startswith(p) for p in frozen_prefixes):
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    from groundingdino.util.inference import load_model
    model = load_model(args.gdino_config, args.gdino_weights)
    model = model.to(device)
    model.train()
    freeze_modules(model)

    transform = build_transform()
    dataset   = MinecraftGroundingDataset(args.coco_json, args.images_dir, transform)
    loader    = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
    )
    print(f"Dataset: {len(dataset)} samples, {len(loader)} batches/epoch")

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * len(loader))

    os.makedirs(args.output_dir, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for batch_i, (images, captions, gt_boxes) in enumerate(loader):
            images = [img.to(device) for img in images]

            # GroundingDINO forward in training mode returns a loss dict
            try:
                outputs = model(images, captions=captions)

                # Build targets for the loss
                targets = []
                for boxes in gt_boxes:
                    targets.append({
                        "boxes":  boxes.to(device),
                        "labels": torch.zeros(len(boxes), dtype=torch.long, device=device),
                    })

                # Compute loss using the model's built-in criterion if available
                if hasattr(model, "criterion") and model.criterion is not None:
                    loss_dict = model.criterion(outputs, targets)
                    loss = sum(loss_dict.values())
                else:
                    # Fallback: simple box regression loss on highest-confidence box
                    pred_boxes  = outputs["pred_boxes"]     # [B, num_queries, 4]
                    pred_logits = outputs["pred_logits"]    # [B, num_queries, C]
                    loss = torch.tensor(0.0, device=device, requires_grad=True)
                    for b_idx, gt in enumerate(gt_boxes):
                        gt = gt.to(device)
                        if gt.numel() == 0:
                            continue
                        # Match pred box closest to each gt box (simple L1)
                        pb = pred_boxes[b_idx]   # [num_queries, 4]
                        for gt_box in gt:
                            dists = torch.abs(pb - gt_box.unsqueeze(0)).sum(-1)
                            best  = dists.argmin()
                            loss  = loss + torch.abs(pb[best] - gt_box).sum()

            except Exception as e:
                print(f"  [WARN] batch {batch_i} loss error: {e}")
                continue

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

            if batch_i % 10 == 0:
                print(f"  Epoch {epoch+1}/{args.epochs}  "
                      f"Batch {batch_i}/{len(loader)}  "
                      f"Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / max(len(loader), 1)
        print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")

        ckpt_path = os.path.join(args.output_dir, f"checkpoint_epoch{epoch+1}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"  Saved: {ckpt_path}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(args.output_dir, "checkpoint_best.pth")
            torch.save(model.state_dict(), best_path)
            print(f"  Best model updated: {best_path}")

    print(f"\nFine-tuning complete. Best loss: {best_loss:.4f}")
    print(f"Best checkpoint: {os.path.join(args.output_dir, 'checkpoint_best.pth')}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-json",      required=True)
    parser.add_argument("--images-dir",     required=True)
    parser.add_argument("--gdino-config",   required=True)
    parser.add_argument("--gdino-weights",  required=True)
    parser.add_argument("--output-dir",     default="data/grounding_data/checkpoints")
    parser.add_argument("--epochs",         type=int,   default=10)
    parser.add_argument("--batch-size",     type=int,   default=4)
    parser.add_argument("--lr",             type=float, default=1e-5)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
