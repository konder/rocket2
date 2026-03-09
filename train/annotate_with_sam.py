"""
Post-process collected grounding data: use SAM-2 with center-point prompt
to generate a precise segmentation mask for each image, then update the
bounding boxes in annotations_coco.json.

The mined block is always near the crosshair center, so the center point
is a reliable SAM prompt.

Usage:
    python annotate_with_sam.py \\
        --data-dir data/grounding_data/ \\
        --sam-path ./MineStudio/minestudio/utils/realtime_sam/checkpoints \\
        --sam-variant base

    # Preview masks without updating JSON:
    python annotate_with_sam.py \\
        --data-dir data/grounding_data/ \\
        --sam-path ./MineStudio/minestudio/utils/realtime_sam/checkpoints \\
        --preview --preview-dir data/grounding_data/previews/
"""

import os
import sys
import json
import argparse
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# --------------------------------------------------------------------------- #
# SAM-2 wrapper (single image, center-point prompt)
# --------------------------------------------------------------------------- #

class CenterPointSAM:
    """
    Uses SAM2ImagePredictor for single-image segmentation.
    Prompt: image center point (positive label=1).
    multimask_output=True → returns 3 candidates, picks highest IoU score.
    Falls back to a circular region if the best mask is too small.
    """

    CKPT_MAP = {
        "large": ("sam2_hiera_large.pt",      "sam2_hiera_l.yaml"),
        "base":  ("sam2_hiera_base_plus.pt",  "sam2_hiera_b+.yaml"),
        "small": ("sam2_hiera_small.pt",      "sam2_hiera_s.yaml"),
        "tiny":  ("sam2_hiera_tiny.pt",       "sam2_hiera_t.yaml"),
    }
    MIN_MASK_PIXELS = 30     # below this → fallback
    FALLBACK_RADIUS = 0.08   # fraction of min(H, W)

    def __init__(self, sam_path: str, variant: str = "base", device: str = None):
        import torch
        import hydra
        from hydra import compose, initialize_config_dir
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        ckpt_file, cfg_file = self.CKPT_MAP[variant]
        ckpt_full = os.path.abspath(os.path.join(sam_path, ckpt_file))

        if not os.path.exists(ckpt_full):
            raise FileNotFoundError(
                f"SAM-2 checkpoint not found: {ckpt_full}\n"
                f"Download it to {sam_path} or pass --sam-path."
            )

        # sam2_configs lives next to the checkpoints dir: <sam_path>/../sam2/sam2_configs
        sam2_configs_dir = os.path.abspath(
            os.path.join(sam_path, "..", "sam2", "sam2_configs")
        )
        if not os.path.isdir(sam2_configs_dir):
            raise FileNotFoundError(f"sam2_configs not found: {sam2_configs_dir}")

        # build_sam2() calls compose() without initialising Hydra first.
        # initialize_config_dir() accepts an absolute path, avoiding CWD issues.
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        initialize_config_dir(config_dir=sam2_configs_dir, version_base="1.3")

        hydra_overrides = [
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        ]
        cfg = compose(config_name=cfg_file, overrides=hydra_overrides)
        OmegaConf.resolve(cfg)
        sam_model = instantiate(cfg.model, _recursive_=True)

        # Load checkpoint weights
        sd = torch.load(ckpt_full, map_location="cpu")["model"]
        missing, unexpected = sam_model.load_state_dict(sd)
        if missing:
            raise RuntimeError(f"Missing keys: {missing}")
        if unexpected:
            raise RuntimeError(f"Unexpected keys: {unexpected}")

        sam_model = sam_model.to(device).eval()
        self.predictor = SAM2ImagePredictor(sam_model)
        print(f"[SAM] Loaded SAM2ImagePredictor ({variant}) from {ckpt_full} on {device}")

    def segment(self, image_rgb: np.ndarray):
        """
        Segment the object at image center using SAM2ImagePredictor.

        :param image_rgb: uint8 RGB image (H, W, 3)
        :returns: (mask, bbox_xyxy, used_fallback)
            mask:         (H, W) float32 binary mask
            bbox_xyxy:    (x1, y1, x2, y2) int
            used_fallback: True if best mask was too small → used circle fallback
        """
        import torch
        h, w = image_rgb.shape[:2]
        cx, cy = w // 2, h // 2

        self.predictor.set_image(image_rgb)

        with torch.inference_mode():
            masks, scores, _ = self.predictor.predict(
                point_coords=np.array([[cx, cy]]),
                point_labels=np.array([1]),
                multimask_output=True,
            )

        # masks: (3, H, W) float32 binary.
        # SAM2 scores are unreliable on Minecraft pixel-art style:
        # the highest-scored mask is often the smallest (few pixels).
        # Strategy: pick the LARGEST mask that still covers the center point.
        # Fall back to highest-score mask if none cover the center.
        masks = masks.astype(np.float32)
        center_covering = [
            (masks[i].sum(), i)
            for i in range(len(masks))
            if masks[i][cy, cx] > 0.5
        ]
        if center_covering:
            center_covering.sort(key=lambda x: -x[0])   # largest first
            mask = masks[center_covering[0][1]]
        else:
            mask = masks[int(np.argmax(scores))]

        if mask.sum() < self.MIN_MASK_PIXELS:
            radius = int(min(h, w) * self.FALLBACK_RADIUS)
            mask = np.zeros((h, w), dtype=np.float32)
            cv2.circle(mask, (cx, cy), radius, 1.0, -1)
            used_fallback = True
        else:
            used_fallback = False

        bbox = _mask_to_bbox(mask)
        return mask, bbox, used_fallback


def _mask_to_bbox(mask: np.ndarray):
    """Convert binary mask to (x1, y1, x2, y2). Returns center square if mask empty."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        h, w = mask.shape
        s = min(h, w) // 8
        cx, cy = w // 2, h // 2
        return (cx - s, cy - s, cx + s, cy + s)
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


# --------------------------------------------------------------------------- #
# Visualisation helper
# --------------------------------------------------------------------------- #

def draw_preview(image_rgb: np.ndarray, mask: np.ndarray, bbox, label: str,
                 used_fallback: bool) -> np.ndarray:
    """Overlay mask + bbox + label onto image; return BGR for cv2.imwrite."""
    vis = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR).copy()
    h, w = vis.shape[:2]

    # Green mask overlay
    overlay = vis.copy()
    overlay[mask > 0] = (0, 200, 80)
    vis = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)

    # Bbox
    x1, y1, x2, y2 = bbox
    color = (0, 100, 255) if used_fallback else (0, 255, 0)
    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

    # Center crosshair
    cx, cy = w // 2, h // 2
    cv2.drawMarker(vis, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 12, 2)

    # Label
    tag = f"{label}" + (" [fallback]" if used_fallback else "")
    cv2.putText(vis, tag, (x1, max(y1 - 6, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return vis


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Annotate grounding images with SAM-2 center point")
    parser.add_argument("--data-dir",    required=True,
                        help="Directory containing images/ and annotations_coco.json")
    parser.add_argument("--sam-path",    required=True,
                        help="Path to SAM-2 checkpoint directory")
    parser.add_argument("--sam-variant", default="base",
                        choices=["large", "base", "small", "tiny"])
    parser.add_argument("--preview",     action="store_true",
                        help="Save overlay images to --preview-dir (does not update JSON)")
    parser.add_argument("--preview-dir", default=None,
                        help="Where to save preview images (default: <data-dir>/previews/)")
    parser.add_argument("--overwrite",   action="store_true",
                        help="Overwrite existing sam_bbox_xyxy in annotations")
    args = parser.parse_args()

    images_dir   = os.path.join(args.data_dir, "images")
    coco_path    = os.path.join(args.data_dir, "annotations_coco.json")
    raw_path     = os.path.join(args.data_dir, "annotations_raw.json")

    if not os.path.exists(coco_path):
        print(f"ERROR: {coco_path} not found. Run collect_grounding_data.py first.")
        sys.exit(1)

    with open(coco_path) as f:
        coco = json.load(f)

    raw_anns = None
    if os.path.exists(raw_path):
        with open(raw_path) as f:
            raw_anns = json.load(f)

    if args.preview:
        preview_dir = args.preview_dir or os.path.join(args.data_dir, "previews")
        os.makedirs(preview_dir, exist_ok=True)

    # Build id → category name
    id_to_cat = {c["id"]: c["name"] for c in coco["categories"]}
    id_to_img  = {img["id"]: img for img in coco["images"]}

    # Load SAM
    sam = CenterPointSAM(args.sam_path, args.sam_variant)

    total = len(coco["annotations"])
    fallback_count = 0
    skip_count = 0

    print(f"\nProcessing {total} annotations...")

    for i, ann in enumerate(coco["annotations"]):
        # Skip if already annotated and not --overwrite
        if "sam_bbox_xyxy" in ann and not args.overwrite:
            skip_count += 1
            continue

        img_meta = id_to_img[ann["image_id"]]
        img_path = os.path.join(images_dir, img_meta["file_name"])
        label    = id_to_cat.get(ann["category_id"], "unknown")

        if not os.path.exists(img_path):
            print(f"  [SKIP] Missing image: {img_path}")
            continue

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"  [SKIP] Cannot read: {img_path}")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        try:
            mask, bbox, used_fallback = sam.segment(img_rgb)
        except Exception as e:
            print(f"  [WARN] SAM failed on {img_meta['file_name']}: {e}")
            continue

        if used_fallback:
            fallback_count += 1

        # Update annotation with SAM bbox (keep original center bbox as fallback reference)
        ann["sam_bbox_xyxy"] = list(bbox)
        x1, y1, x2, y2 = bbox
        ann["bbox"] = [x1, y1, x2 - x1, y2 - y1]   # COCO format [x,y,w,h]
        ann["area"] = (x2 - x1) * (y2 - y1)
        ann["sam_fallback"] = used_fallback

        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"  [{i+1}/{total}] {img_meta['file_name']} "
                  f"label='{label}' bbox={bbox} fallback={used_fallback}")

        if args.preview:
            vis = draw_preview(img_rgb, mask, bbox, label, used_fallback)
            preview_name = img_meta["file_name"].replace(".png", "_sam.png")
            cv2.imwrite(os.path.join(preview_dir, preview_name), vis)

    # Save updated COCO
    if not args.preview:
        with open(coco_path, "w") as f:
            json.dump(coco, f, indent=2)
        print(f"\nUpdated: {coco_path}")

        # Also update raw annotations if available
        if raw_anns is not None:
            ann_by_fname = {}
            for ann in coco["annotations"]:
                fname = id_to_img[ann["image_id"]]["file_name"]
                ann_by_fname[fname] = ann

            for raw in raw_anns:
                fname = raw.get("image_path", "")
                if fname in ann_by_fname:
                    a = ann_by_fname[fname]
                    if "sam_bbox_xyxy" in a:
                        raw["sam_bbox_xyxy"] = a["sam_bbox_xyxy"]
                        raw["sam_fallback"]  = a.get("sam_fallback", False)

            with open(raw_path, "w") as f:
                json.dump(raw_anns, f, indent=2)
            print(f"Updated: {raw_path}")

    print(f"\nDone.")
    print(f"  Total annotations : {total}")
    print(f"  Skipped (existing): {skip_count}")
    print(f"  SAM fallback used : {fallback_count}")
    if args.preview:
        print(f"  Previews saved to : {preview_dir}")
        print(f"  (JSON not modified in preview mode)")


if __name__ == "__main__":
    main()
