"""
Compare goal detection across multiple backends:
  - groundingdino : GroundingDINO direct bbox
  - molmo         : Molmo pointing + SAM2 segmentation → bbox
  - gdino_sam2    : GroundingDINO bbox → SAM2 box-prompt refine → bbox
  - sa2va         : R-Sa2VA (SAM2 + VLM) text-prompted segmentation → bbox

Usage (all four):
    python benchmark/compare_detectors.py \
        --task-file benchmark/eval_tasks_paper.yaml \
        --image-dir benchmark/first_frames/ \
        --output-dir benchmark/detection_compare/ \
        --backends groundingdino molmo gdino_sam2 sa2va \
        --sam-path ./MineStudio/minestudio/utils/realtime_sam/checkpoints \
        --device cuda

Usage (DINO vs DINO+SAM2 only):
    python benchmark/compare_detectors.py \
        --task-file benchmark/eval_tasks_paper.yaml \
        --image-dir benchmark/first_frames/ \
        --output-dir benchmark/detection_compare/ \
        --backends groundingdino gdino_sam2 \
        --sam-path ./MineStudio/minestudio/utils/realtime_sam/checkpoints

Sa2VA requires:
    pip install "transformers @ git+https://github.com/huggingface/transformers.git@main"
"""

import os
import sys
import json
import re
import argparse
import yaml
import cv2
import numpy as np
import torch
from typing import List, Dict, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

COLORS = {
    "groundingdino": (0, 200, 0),  # green (BGR)
    "molmo": (0, 0, 255),          # red
    "gdino_sam2": (0, 255, 255),   # yellow
    "sa2va": (255, 100, 0),        # blue
}

DISPLAY_NAMES = {
    "groundingdino": "GroundingDINO",
    "molmo": "Molmo+SAM2",
    "gdino_sam2": "DINO+SAM2",
    "sa2va": "R-Sa2VA",
}


def get_device(requested: Optional[str] = None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ---------------------------------------------------------------------------
# Molmo backend
# ---------------------------------------------------------------------------
class MolmoDetector:
    def __init__(self, model_id: str, device: str,
                 sam_path: Optional[str] = None, sam_variant: str = "base"):
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.device = device
        if device == "cpu":
            dtype = torch.float32
        elif device == "cuda" and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
        self.dtype = dtype

        print(f"[Molmo] Loading {model_id} ...")
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype=dtype,
        ).to(device).eval()

        if dtype in (torch.float16, torch.bfloat16):
            target = dtype
            self.model.model.vision_backbone.to(torch.float32)

            def _hook(module, inp, out):
                if isinstance(out, torch.Tensor):
                    return out.to(target)
                if isinstance(out, (tuple, list)):
                    return type(out)(o.to(target) if isinstance(o, torch.Tensor) else o for o in out)
                return out

            self.model.model.vision_backbone.register_forward_hook(_hook)

        self.sam_predictor = None
        if sam_path:
            self._load_sam(sam_path, sam_variant)

        print(f"[Molmo] Ready on {device} (dtype={dtype}, SAM2={'yes' if self.sam_predictor else 'no'})")

    def _load_sam(self, sam_path: str, variant: str):
        from sam2.build_sam import build_sam2_camera_predictor

        ckpt_mapping = {
            "large": ("sam2_hiera_large.pt", "sam2_hiera_l.yaml"),
            "base": ("sam2_hiera_base_plus.pt", "sam2_hiera_b+.yaml"),
            "small": ("sam2_hiera_small.pt", "sam2_hiera_s.yaml"),
            "tiny": ("sam2_hiera_tiny.pt", "sam2_hiera_t.yaml"),
        }
        ckpt_file, cfg_file = ckpt_mapping.get(variant, ckpt_mapping["base"])
        ckpt_full = os.path.join(sam_path, ckpt_file)
        self.sam_predictor = build_sam2_camera_predictor(
            cfg_file, ckpt_full, device=self.device,
        )
        print(f"[Molmo] SAM-2 ({variant}) loaded from {ckpt_full}")

    def _sam_point_to_bbox(self, image_rgb: np.ndarray,
                           point: Tuple[int, int]) -> Optional[List[float]]:
        """Use SAM2 to segment around the point and derive a tight bbox."""
        if self.sam_predictor is None:
            return None

        self.sam_predictor.load_first_frame(image_rgb)
        _, _, out_mask_logits = self.sam_predictor.add_new_prompt(
            frame_idx=0, obj_id=0,
            points=[list(point)], labels=[1],
        )
        mask = (out_mask_logits[0, 0] > 0.0).cpu().numpy().astype(np.uint8)

        if mask.sum() < 10:
            return None

        ys, xs = np.where(mask > 0)
        return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]

    def detect(self, image: np.ndarray, prompt: str) -> List[Dict]:
        from PIL import Image as PILImage
        from transformers import GenerationConfig

        pil = PILImage.fromarray(image)
        inputs = self.processor.process(images=[pil], text=prompt)
        inputs = {
            k: v.to(self.device).unsqueeze(0) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
        max_id = self.model.config.vocab_size - 1
        if inputs["input_ids"].max() > max_id:
            inputs["input_ids"] = inputs["input_ids"].clamp(max=max_id)

        gen_cfg = GenerationConfig(
            max_new_tokens=64, use_cache=True, do_sample=False,
            stop_strings=["<|endoftext|>"], repetition_penalty=1.2,
        )
        with torch.inference_mode():
            out = self.model.generate_from_batch(
                inputs, generation_config=gen_cfg,
                tokenizer=self.processor.tokenizer,
            )
        input_len = inputs["input_ids"].shape[1]
        text = self.processor.tokenizer.decode(out[0, input_len:], skip_special_tokens=True)
        print(f"  [Molmo] raw: {text[:200]}")

        h, w = image.shape[:2]
        point = self._parse_point(text, w, h)
        if point is None:
            return []
        x, y = point

        sam_bbox = self._sam_point_to_bbox(image, point)
        if sam_bbox is not None:
            bbox = sam_bbox
            label = "molmo+sam2"
            print(f"  [Molmo] SAM2 bbox: ({bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f})")
        else:
            r = 30
            bbox = [max(0, x - r), max(0, y - r), min(w - 1, x + r), min(h - 1, y + r)]
            label = "molmo_point"
            if self.sam_predictor is not None:
                print(f"  [Molmo] SAM2 mask too small, using point fallback")

        return [{
            "label": label,
            "score": -1.0,
            "bbox": bbox,
            "point": (x, y),
            "raw_text": text.strip(),
        }]

    @staticmethod
    def _parse_point(text: str, w: int, h: int):
        m = re.search(r"x\d*\s*=\s*([\d.]+).*?y\d*\s*=\s*([\d.]+)", text, re.I)
        if m:
            return (int(np.clip(float(m.group(1)) / 100.0 * w, 0, w - 1)),
                    int(np.clip(float(m.group(2)) / 100.0 * h, 0, h - 1)))
        m = re.search(r'<point\s+x="([\d.]+)"\s+y="([\d.]+)"', text)
        if m:
            return (int(np.clip(float(m.group(1)) / 100.0 * w, 0, w - 1)),
                    int(np.clip(float(m.group(2)) / 100.0 * h, 0, h - 1)))
        return None


# ---------------------------------------------------------------------------
# GroundingDINO backend
# ---------------------------------------------------------------------------
class GDinoDetector:
    HF_REPO = "ShilongLiu/GroundingDINO"

    def __init__(self, config_path: Optional[str], weights_path: Optional[str],
                 box_thr: float, text_thr: float, device: str):
        from groundingdino.util.inference import load_model
        self.device = device
        self.box_thr = box_thr
        self.text_thr = text_thr

        if not weights_path or not os.path.exists(weights_path):
            weights_path = self._hf_download([
                "groundingdino_swint_ogc.pth",
                "weights/groundingdino_swint_ogc.pth",
            ])
        if not config_path or not os.path.exists(config_path):
            config_path = self._hf_download([
                "GroundingDINO_SwinT_OGC.py",
                "groundingdino/config/GroundingDINO_SwinT_OGC.py",
            ])

        print(f"[GDINO] config  = {config_path}")
        print(f"[GDINO] weights = {weights_path}")
        self.model = load_model(config_path, weights_path).to(device).eval()
        print(f"[GDINO] Ready on {device}")

    @classmethod
    def _hf_download(cls, filenames: list) -> str:
        from huggingface_hub import hf_hub_download
        for name in filenames:
            try:
                return hf_hub_download(repo_id=cls.HF_REPO, filename=name)
            except Exception:
                continue
        raise RuntimeError(
            f"Could not download {filenames} from {cls.HF_REPO}. "
            f"Pass --gdino-weights / --gdino-config explicitly."
        )

    def detect(self, image: np.ndarray, prompt: str) -> List[Dict]:
        from PIL import Image as PILImage
        from groundingdino.datasets import transforms as T
        from groundingdino.util.inference import predict

        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        pil = PILImage.fromarray(image)
        tensor, _ = transform(pil, None)
        caption = prompt.strip()
        if not caption.endswith("."):
            caption += "."

        boxes, logits, phrases = predict(
            model=self.model, image=tensor, caption=caption,
            box_threshold=self.box_thr, text_threshold=self.text_thr,
            device=self.device,
        )
        if boxes is None or len(boxes) == 0:
            print(f"  [GDINO] no detection")
            return []

        h, w = image.shape[:2]
        results = []
        for i in range(len(logits)):
            box = boxes[i].tolist()
            if max(box) <= 1.5:
                cx, cy, bw, bh = box
                x1 = np.clip((cx - bw / 2) * w, 0, w - 1)
                y1 = np.clip((cy - bh / 2) * h, 0, h - 1)
                x2 = np.clip((cx + bw / 2) * w, 0, w - 1)
                y2 = np.clip((cy + bh / 2) * h, 0, h - 1)
            else:
                x1, y1, x2, y2 = box
            score = float(logits[i])
            phrase = phrases[i] if phrases else "?"
            results.append({
                "label": phrase,
                "score": score,
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "point": (int((x1 + x2) / 2), int((y1 + y2) / 2)),
            })
            print(f"  [GDINO] [{i}] '{phrase}' score={score:.3f} "
                  f"bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")
        return results


# ---------------------------------------------------------------------------
# GroundingDINO + SAM2:  DINO bbox → SAM2 box prompt → mask → refined bbox
# ---------------------------------------------------------------------------
class GDinoSAM2Detector:
    """DINO finds bbox, SAM2 refines it via box prompt segmentation."""

    def __init__(self, gdino: GDinoDetector,
                 sam_path: str, sam_variant: str, device: str):
        from sam2.build_sam import build_sam2_camera_predictor

        self.gdino = gdino
        self.device = device

        ckpt_mapping = {
            "large": ("sam2_hiera_large.pt", "sam2_hiera_l.yaml"),
            "base": ("sam2_hiera_base_plus.pt", "sam2_hiera_b+.yaml"),
            "small": ("sam2_hiera_small.pt", "sam2_hiera_s.yaml"),
            "tiny": ("sam2_hiera_tiny.pt", "sam2_hiera_t.yaml"),
        }
        ckpt_file, cfg_file = ckpt_mapping.get(sam_variant, ckpt_mapping["base"])
        ckpt_full = os.path.join(sam_path, ckpt_file)
        self.sam_predictor = build_sam2_camera_predictor(
            cfg_file, ckpt_full, device=device,
        )
        print(f"[DINO+SAM2] SAM-2 ({sam_variant}) loaded from {ckpt_full}")

    def detect(self, image: np.ndarray, prompt: str) -> List[Dict]:
        dino_dets = self.gdino.detect(image, prompt)
        if not dino_dets:
            return []

        h, w = image.shape[:2]
        results = []

        for i, det in enumerate(dino_dets):
            x1, y1, x2, y2 = det["bbox"]
            score = det["score"]
            label = det["label"]

            self.sam_predictor.load_first_frame(image)
            # Box prompt: two corner points (top-left, bottom-right) with
            # labels [2, 3] — SAM2's internal representation of a box prompt.
            _, _, out_mask_logits = self.sam_predictor.add_new_prompt(
                frame_idx=0, obj_id=0,
                points=[[int(x1), int(y1)], [int(x2), int(y2)]],
                labels=[2, 3],
            )
            mask = (out_mask_logits[0, 0] > 0.0).cpu().numpy().astype(np.uint8)

            if mask.sum() >= 10:
                ys, xs = np.where(mask > 0)
                rx1, ry1 = float(xs.min()), float(ys.min())
                rx2, ry2 = float(xs.max()), float(ys.max())
                print(f"  [DINO+SAM2] [{i}] SAM2 refined: "
                      f"({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}) → "
                      f"({rx1:.0f},{ry1:.0f},{rx2:.0f},{ry2:.0f})")
            else:
                rx1, ry1, rx2, ry2 = x1, y1, x2, y2
                print(f"  [DINO+SAM2] [{i}] SAM2 mask too small, keeping DINO bbox")

            results.append({
                "label": label,
                "score": score,
                "bbox": [rx1, ry1, rx2, ry2],
                "point": (int((rx1 + rx2) / 2), int((ry1 + ry2) / 2)),
            })

        return results


# ---------------------------------------------------------------------------
# R-Sa2VA: SAM2 + VLM unified model — text prompt → segmentation mask → bbox
# https://github.com/bytedance/Sa2VA
# ---------------------------------------------------------------------------
class Sa2VADetector:
    """R-Sa2VA uses a unified SAM2+VLM architecture to segment objects from text."""

    def __init__(self, model_id: str, device: str):
        from transformers import AutoModelForCausalLM, AutoProcessor
        self.device = device
        self.model_id = model_id

        print(f"[Sa2VA] Loading {model_id} ...")

        # Sa2VA's SAM2 init calls torch.linspace().item() which fails on
        # meta tensors created by transformers' init_empty_weights() context.
        # Patch torch.linspace to always produce CPU tensors during loading.
        _orig_linspace = torch.linspace
        def _cpu_linspace(*args, **kwargs):
            if "device" not in kwargs or str(kwargs.get("device")) == "meta":
                kwargs["device"] = "cpu"
            return _orig_linspace(*args, **kwargs)
        torch.linspace = _cpu_linspace

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            ).to(device).eval()
        finally:
            torch.linspace = _orig_linspace

        self.processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True,
        )
        self.tokenizer = None
        print(f"[Sa2VA] Ready on {device}")

    @staticmethod
    def _build_prompt(prompt: str) -> str:
        obj_name = prompt.strip()
        for prefix in ["Point to the", "Point to"]:
            if obj_name.lower().startswith(prefix.lower()):
                obj_name = obj_name[len(prefix):].strip()
                break
        return f"<image>Please segment the {obj_name}."

    def detect(self, image: np.ndarray, prompt: str) -> List[Dict]:
        from PIL import Image as PILImage

        pil = PILImage.fromarray(image)
        h, w = image.shape[:2]
        text = self._build_prompt(prompt)

        print(f"  [Sa2VA] prompt: {text}")
        result = self.model.predict_forward(
            image=pil,
            text=text,
            tokenizer=self.tokenizer,
            processor=self.processor,
        )

        prediction = result.get("prediction", "")
        print(f"  [Sa2VA] prediction: {prediction[:200]}")

        results = []
        if "[SEG]" in prediction and "prediction_masks" in result:
            masks = result["prediction_masks"]
            for seg_idx, mask_set in enumerate(masks):
                if isinstance(mask_set, list):
                    mask = mask_set[0]
                else:
                    mask = mask_set

                if isinstance(mask, torch.Tensor):
                    mask = mask.cpu().numpy()
                mask = mask.astype(np.uint8)

                if mask.shape[:2] != (h, w):
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

                if mask.sum() < 10:
                    print(f"  [Sa2VA] seg[{seg_idx}] mask too small, skipping")
                    continue

                ys, xs = np.where(mask > 0)
                x1, y1 = float(xs.min()), float(ys.min())
                x2, y2 = float(xs.max()), float(ys.max())

                results.append({
                    "label": "sa2va_seg",
                    "score": -1.0,
                    "bbox": [x1, y1, x2, y2],
                    "point": (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                })
                print(f"  [Sa2VA] seg[{seg_idx}] bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")
        else:
            print(f"  [Sa2VA] no [SEG] token in output")

        return results


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def draw_detections(image: np.ndarray, backend: str, detections: List[Dict],
                    top_k: int = 1) -> np.ndarray:
    color = COLORS.get(backend, (128, 128, 128))
    name = DISPLAY_NAMES.get(backend, backend)
    vis = image.copy()

    items = detections[:top_k] if top_k > 0 else detections
    for i, det in enumerate(items):
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        score = det["score"]
        label = det.get("label", "")

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        if det.get("point"):
            px, py = det["point"]
            cv2.circle(vis, (px, py), 6, color, -1)
            cv2.circle(vis, (px, py), 8, (255, 255, 255), 1)

        tag = f"{name}"
        if 0 <= score < 1.0:
            tag += f" {score:.2f}"
        if label and label not in ("molmo_point", "molmo+sam2", "sa2va_seg"):
            tag += f" [{label}]"
        rank = f"#{i}" if len(items) > 1 else ""
        text = f"{tag}{rank}"

        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(y1 - 6, th + 4)
        cv2.rectangle(vis, (x1, ty - th - 4), (x1 + tw + 4, ty + 2), color, -1)
        cv2.putText(vis, text, (x1 + 2, ty - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Compare detection backends")
    parser.add_argument("--task-file", required=True)
    parser.add_argument("--image-dir", required=True,
                        help="Directory with {task_name}.png first-frame images")
    parser.add_argument("--output-dir", default="benchmark/detection_compare/")
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--backends", nargs="+",
                        choices=["groundingdino", "molmo", "gdino_sam2", "sa2va"],
                        default=["groundingdino", "molmo", "gdino_sam2", "sa2va"])
    parser.add_argument("--device", default=None)
    # Molmo + SAM2
    parser.add_argument("--molmo-model", default="allenai/Molmo-7B-D-0924")
    parser.add_argument("--sam-path", default=None,
                        help="SAM2 checkpoints directory (required for molmo and gdino_sam2). "
                             "e.g. ./MineStudio/minestudio/utils/realtime_sam/checkpoints")
    parser.add_argument("--sam-variant", default="base",
                        choices=["large", "base", "small", "tiny"])
    # GroundingDINO
    parser.add_argument("--gdino-config",
                        default="/opt/conda/envs/rocket2/lib/python3.10/site-packages/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument("--gdino-weights", default=None)
    parser.add_argument("--gdino-box-threshold", type=float, default=0.25)
    parser.add_argument("--gdino-text-threshold", type=float, default=0.20)
    # Sa2VA
    parser.add_argument("--sa2va-model", default="HarborYuan/R-Sa2VA-Qwen3VL-4B-RL")
    # Display
    parser.add_argument("--top-k", type=int, default=1,
                        help="Draw top-K detections per model (0=all, default=1)")
    args = parser.parse_args()

    device = get_device(args.device)

    with open(args.task_file) as f:
        config = yaml.safe_load(f)
    tasks = config["tasks"]
    if args.tasks:
        tasks = [t for t in tasks if t["name"] in args.tasks]

    os.makedirs(args.output_dir, exist_ok=True)

    detectors = {}
    gdino = None

    if "groundingdino" in args.backends or "gdino_sam2" in args.backends:
        gdino = GDinoDetector(
            args.gdino_config, args.gdino_weights,
            args.gdino_box_threshold, args.gdino_text_threshold, device,
        )
        if "groundingdino" in args.backends:
            detectors["groundingdino"] = gdino

    if "molmo" in args.backends:
        detectors["molmo"] = MolmoDetector(
            args.molmo_model, device,
            sam_path=args.sam_path, sam_variant=args.sam_variant,
        )

    if "gdino_sam2" in args.backends:
        if not args.sam_path:
            parser.error("--sam-path is required for gdino_sam2 backend")
        detectors["gdino_sam2"] = GDinoSAM2Detector(
            gdino, args.sam_path, args.sam_variant, device,
        )

    if "sa2va" in args.backends:
        detectors["sa2va"] = Sa2VADetector(args.sa2va_model, device)

    summary = []

    for task in tasks:
        name = task["name"]
        prompt = task.get("text", "")
        img_path = os.path.join(args.image_dir, f"{name}.png")
        if not os.path.exists(img_path):
            print(f"\n[SKIP] {name}: {img_path} not found")
            continue

        print(f"\n{'='*60}")
        print(f"  Task: {name}")
        print(f"  Prompt: {prompt}")
        print(f"{'='*60}")

        bgr = cv2.imread(img_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        canvas = bgr.copy()
        task_results = {"task": name, "prompt": prompt}

        for backend, detector in detectors.items():
            print(f"\n  --- {DISPLAY_NAMES[backend]} ---")
            try:
                dets = detector.detect(rgb, prompt)
            except Exception as e:
                import traceback
                print(f"  ERROR: {e}")
                traceback.print_exc()
                dets = []

            task_results[backend] = [
                {"label": d["label"], "score": d["score"],
                 "bbox": d["bbox"], "point": d["point"]}
                for d in dets
            ]
            canvas = draw_detections(canvas, backend, dets, top_k=args.top_k)

        h, w = canvas.shape[:2]
        legend_y = h - 10
        for i, backend in enumerate(detectors):
            color = COLORS[backend]
            lx = 10 + i * 200
            cv2.rectangle(canvas, (lx, legend_y - 14), (lx + 12, legend_y), color, -1)
            cv2.putText(canvas, DISPLAY_NAMES[backend], (lx + 16, legend_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(canvas, f"Task: {name} | Prompt: {prompt}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        out_path = os.path.join(args.output_dir, f"{name}_compare.png")
        cv2.imwrite(out_path, canvas)
        print(f"\n  Saved: {out_path}")
        summary.append(task_results)

    json_path = os.path.join(args.output_dir, "detection_results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nJSON results: {json_path}")
    print(f"Done. {len(summary)} tasks processed.")


if __name__ == "__main__":
    main()
