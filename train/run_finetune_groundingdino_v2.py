"""
Fine-tune GroundingDINO on Minecraft block data (COCO format) - V2

Improved version with proper Hungarian matching and focal loss.
Based on Open-GroundingDino implementation.
"""

import os
import sys
import json
import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import cv2
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment


# --------------------------------------------------------------------------- #
# Box utilities
# --------------------------------------------------------------------------- #

def box_cxcywh_to_xyxy(boxes):
    """Convert boxes from [cx, cy, w, h] to [x1, y1, x2, y2] format"""
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/
    boxes1: [N, 4] in xyxy format
    boxes2: [M, 4] in xyxy format
    Returns: [N, M] GIoU matrix
    """
    # Ensure boxes are in xyxy format
    assert boxes1.shape[-1] == 4
    assert boxes2.shape[-1] == 4
    
    # Area of boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # Intersection
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  # [N, M, 2]
    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]
    
    # Union
    union = area1[:, None] + area2[None, :] - inter
    
    # IoU
    iou = inter / (union + 1e-7)
    
    # Smallest enclosing box
    lt_enc = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    rb_enc = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh_enc = (rb_enc - lt_enc).clamp(min=0)
    area_enc = wh_enc[:, :, 0] * wh_enc[:, :, 1]
    
    # GIoU
    giou = iou - (area_enc - union) / (area_enc + 1e-7)
    
    return giou


# --------------------------------------------------------------------------- #
# Hungarian Matcher
# --------------------------------------------------------------------------- #

class HungarianMatcher(nn.Module):
    """Hungarian matcher for DETR-style object detection"""
    
    def __init__(self, cost_class=2.0, cost_bbox=5.0, cost_giou=2.0, focal_alpha=0.25):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.focal_alpha = focal_alpha
    
    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        outputs: dict with 'pred_logits' [B, N, C] and 'pred_boxes' [B, N, 4]
        targets: list of dicts with 'labels' [M] and 'boxes' [M, 4]
        Returns: list of (pred_idx, target_idx) tuples for each batch
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]
        
        # Flatten predictions
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  # [B*N, C]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [B*N, 4]
        
        # Concatenate targets
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        
        if len(tgt_bbox) == 0:
            return [(torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)) for _ in range(bs)]
        
        # Classification cost (focal loss style)
        alpha = self.focal_alpha
        gamma = 2.0
        neg_cost = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost[:, tgt_ids] - neg_cost[:, tgt_ids]
        
        # Bbox L1 cost
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        
        # GIoU cost
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox),
            box_cxcywh_to_xyxy(tgt_bbox)
        )
        
        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()
        C[torch.isnan(C)] = 0.0
        C[torch.isinf(C)] = 0.0
        
        # Hungarian assignment
        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


# --------------------------------------------------------------------------- #
# SetCriterion (Loss)
# --------------------------------------------------------------------------- #

class SetCriterion(nn.Module):
    """Loss computation for GroundingDINO"""
    
    def __init__(self, matcher, weight_dict, focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
    
    def loss_labels(self, outputs, targets, indices, num_boxes):
        """Focal loss for contrastive classification"""
        # For GroundingDINO, we use a simplified approach:
        # Matched boxes should have high confidence, unmatched should have low
        pred_logits = outputs['pred_logits']  # [B, N, 256]
        
        # Create target: matched boxes = 1, unmatched = 0
        # For simplicity, we'll use max confidence across text dimension
        bs, nq = pred_logits.shape[:2]
        
        # Get matched indices
        target_classes = torch.zeros(bs, nq, dtype=torch.float, device=pred_logits.device)
        
        for i, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) > 0:
                target_classes[i, src_idx] = 1.0
        
        # Get max confidence per query (simplified for contrastive)
        pred_scores = pred_logits.sigmoid().max(dim=-1)[0]  # [B, N]
        
        # Binary cross entropy
        alpha = self.focal_alpha
        gamma = self.focal_gamma
        
        ce_loss = F.binary_cross_entropy_with_logits(pred_scores, target_classes, reduction='none')
        
        # Focal weighting
        p_t = pred_scores * target_classes + (1 - pred_scores) * (1 - target_classes)
        loss = ce_loss * ((1 - p_t) ** gamma)
        
        if alpha >= 0:
            alpha_t = alpha * target_classes + (1 - alpha) * (1 - target_classes)
            loss = alpha_t * loss
        
        loss = loss.sum() / num_boxes
        
        return {'loss_ce': loss}
    
    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """L1 and GIoU loss for bounding boxes"""
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][J] for t, (_, J) in zip(targets, indices)], dim=0)
        
        # L1 loss
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        
        # GIoU loss
        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes)
        ))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        
        return losses
    
    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
    
    def forward(self, outputs, targets):
        """Compute all losses"""
        # Match predictions to targets
        indices = self.matcher(outputs, targets)
        
        # Number of boxes for normalization
        num_boxes = sum(len(t["boxes"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        num_boxes = max(num_boxes.item(), 1)
        
        # Compute losses
        losses = {}
        losses.update(self.loss_labels(outputs, targets, indices, num_boxes))
        losses.update(self.loss_boxes(outputs, targets, indices, num_boxes))
        
        # Weight losses
        final_loss = sum(losses[k] * self.weight_dict.get(k, 1.0) for k in losses)
        
        return final_loss, losses


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

class MinecraftGroundingDataset(Dataset):
    """COCO-format dataset for Minecraft block detection"""
    
    def __init__(self, coco_json, images_dir, transform=None):
        with open(coco_json) as f:
            coco = json.load(f)
        
        self.images_dir = images_dir
        self.transform = transform
        
        id_to_img = {img["id"]: img for img in coco["images"]}
        id_to_cat = {cat["id"]: cat["name"] for cat in coco["categories"]}
        
        from collections import defaultdict
        img_to_anns = defaultdict(list)
        for ann in coco["annotations"]:
            img_to_anns[ann["image_id"]].append(ann)
        
        self.samples = []
        for img_id, anns in img_to_anns.items():
            img_meta = id_to_img[img_id]
            boxes = []
            labels = []
            for ann in anns:
                x, y, w, h = ann["bbox"]
                W, H = img_meta["width"], img_meta["height"]
                cx = (x + w / 2) / W
                cy = (y + h / 2) / H
                bw = w / W
                bh = h / H
                boxes.append([cx, cy, bw, bh])
                labels.append(ann["category_id"])
            
            self.samples.append({
                "file_name": img_meta["file_name"],
                "boxes": boxes,
                "labels": labels,
                "width": img_meta["width"],
                "height": img_meta["height"],
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
        
        boxes = torch.tensor(sample["boxes"], dtype=torch.float32)
        labels = torch.tensor(sample["labels"], dtype=torch.long)
        
        return img_tensor, boxes, labels, sample["file_name"]


def collate_fn(batch):
    images, boxes, labels, filenames = zip(*batch)
    return list(images), list(boxes), list(labels), list(filenames)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def build_transform():
    import groundingdino.datasets.transforms as T
    return T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def freeze_modules(model):
    """Freeze backbone and text encoder"""
    frozen_prefixes = ["backbone", "bert"]
    for name, param in model.named_parameters():
        if any(name.startswith(p) for p in frozen_prefixes):
            param.requires_grad = False
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")


def train(args):
    # Force CPU for training due to MPS compatibility issues with grid_sampler_2d_backward
    # MPS doesn't support this operation needed for backward pass
    device = torch.device("cpu")
    print(f"Device: CPU (MPS has compatibility issues with grid_sampler_2d_backward)")
    
    # Load model
    from groundingdino.util.inference import load_model
    model = load_model(args.gdino_config, args.gdino_weights)
    model = model.to(device)
    model.train()
    freeze_modules(model)
    
    # Build criterion
    matcher = HungarianMatcher(
        cost_class=args.cost_class,
        cost_bbox=args.cost_bbox,
        cost_giou=args.cost_giou,
    )
    weight_dict = {
        'loss_ce': args.weight_ce,
        'loss_bbox': args.weight_bbox,
        'loss_giou': args.weight_giou,
    }
    criterion = SetCriterion(matcher, weight_dict)
    
    # Dataset
    transform = build_transform()
    dataset = MinecraftGroundingDataset(args.coco_json, args.images_dir, transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # CPU mode
        collate_fn=collate_fn,
    )
    print(f"Dataset: {len(dataset)} samples, {len(loader)} batches/epoch")
    
    # Optimizer
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * len(loader))
    
    # Training loop
    os.makedirs(args.output_dir, exist_ok=True)
    best_loss = float("inf")
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_losses = {}
        
        for batch_i, (images, gt_boxes, gt_labels, filenames) in enumerate(loader):
            images = [img.to(device) for img in images]
            
            # Convert to NestedTensor
            from groundingdino.util.misc import nested_tensor_from_tensor_list
            samples = nested_tensor_from_tensor_list(images)
            
            # Forward pass
            captions = ["block ."] * len(images)  # Simple caption for all blocks
            outputs = model(samples, captions=captions)
            
            # Build targets
            targets = []
            for boxes, labels in zip(gt_boxes, gt_labels):
                targets.append({
                    "boxes": boxes.to(device),
                    "labels": labels.to(device),
                })
            
            # Compute loss
            loss, loss_dict = criterion(outputs, targets)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            for k, v in loss_dict.items():
                epoch_losses[k] = epoch_losses.get(k, 0) + v.item()
            
            if batch_i % 10 == 0:
                loss_str = "  ".join([f"{k}: {v.item():.4f}" for k, v in loss_dict.items()])
                print(f"  Epoch {epoch+1}/{args.epochs}  Batch {batch_i}/{len(loader)}  Loss: {loss.item():.4f}  [{loss_str}]")
        
        avg_loss = epoch_loss / max(len(loader), 1)
        print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")
        
        # Save checkpoint
        ckpt_path = os.path.join(args.output_dir, f"checkpoint_epoch{epoch+1}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"  Saved: {ckpt_path}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(args.output_dir, "checkpoint_best.pth")
            torch.save(model.state_dict(), best_path)
            print(f"  Best model updated: {best_path}")
    
    print(f"\nFine-tuning complete. Best loss: {best_loss:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-json", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--gdino-config", required=True)
    parser.add_argument("--gdino-weights", required=True)
    parser.add_argument("--output-dir", default="data/grounding_data/checkpoints")
    parser.add_argument("--log-file", default=None, help="Path to log file (default: output_dir/training.log)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    # Loss weights
    parser.add_argument("--cost-class", type=float, default=2.0)
    parser.add_argument("--cost-bbox", type=float, default=5.0)
    parser.add_argument("--cost-giou", type=float, default=2.0)
    parser.add_argument("--weight-ce", type=float, default=2.0)
    parser.add_argument("--weight-bbox", type=float, default=5.0)
    parser.add_argument("--weight-giou", type=float, default=2.0)
    args = parser.parse_args()
    
    # Set up logging
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = args.log_file or os.path.join(args.output_dir, "training.log")
    
    # Redirect stdout to both console and file
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Replace print with logger
    import builtins
    original_print = builtins.print
    def logged_print(*args, **kwargs):
        original_print(*args, **kwargs)
        logger.info(' '.join(str(a) for a in args))
    builtins.print = logged_print
    
    print(f"Log file: {log_file}")
    
    train(args)


if __name__ == "__main__":
    main()