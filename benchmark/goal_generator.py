"""
Goal generators for ROCKET-2 benchmark evaluation.

Provides automated cross-view goal specification via:
  - MockGoalGenerator: deterministic center-point (for macOS debugging)
  - MolmoGoalGenerator: Molmo-7B + SAM-2 pipeline (for Linux evaluation)
"""

import os
import cv2
import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Any


def get_device() -> str:
    env_device = os.environ.get("ROCKET_DEVICE")
    if env_device:
        return env_device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class GoalGeneratorBase(ABC):
    """Generate (point, mask) from an observation image and text instruction."""

    @abstractmethod
    def generate(
        self,
        obs_image: np.ndarray,
        text: str,
    ) -> Tuple[Tuple[int, int], np.ndarray]:
        """
        Args:
            obs_image: (H, W, 3) RGB image from the environment.
            text: natural-language instruction for Molmo, e.g. "Point to the cow".

        Returns:
            point: (x, y) pixel coordinate on obs_image.
            mask: (H, W) binary segmentation mask (float32, 0 or 1).
        """
        ...


class MockGoalGenerator(GoalGeneratorBase):
    """Returns center point and a circular mask for debugging without Molmo/SAM."""

    def __init__(self, radius_frac: float = 0.15):
        self.radius_frac = radius_frac

    def generate(
        self,
        obs_image: np.ndarray,
        text: str,
    ) -> Tuple[Tuple[int, int], np.ndarray]:
        h, w = obs_image.shape[:2]
        cx, cy = w // 2, h // 2
        mask = np.zeros((h, w), dtype=np.float32)
        radius = int(min(h, w) * self.radius_frac)
        cv2.circle(mask, (cx, cy), radius, 1.0, -1)
        return (cx, cy), mask


class MolmoGoalGenerator(GoalGeneratorBase):
    """
    Uses Molmo-7B-D to locate the target object from a text prompt,
    then SAM-2 to produce a segmentation mask.
    """

    def __init__(
        self,
        molmo_model_id: str = "allenai/Molmo-7B-D-0924",
        sam_path: str = "./MineStudio/minestudio/utils/realtime_sam/checkpoints",
        sam_variant: str = "base",
        device: Optional[str] = None,
    ):
        self.device = device or get_device()
        self._load_molmo(molmo_model_id)
        self._load_sam(sam_path, sam_variant)

    def _load_molmo(self, model_id: str):
        from transformers import AutoModelForCausalLM, AutoProcessor

        if self.device == "cpu":
            dtype = torch.float32
        elif self.device == "cuda" and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16

        print(f"[GoalGenerator] Loading Molmo from {model_id} ...")
        self.molmo_processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True
        )
        self.molmo_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(self.device).eval()

        tokenizer_vocab = len(self.molmo_processor.tokenizer)
        if self.molmo_model.config.vocab_size < tokenizer_vocab:
            print(f"[GoalGenerator] WARNING: tokenizer ({tokenizer_vocab}) > model vocab ({self.molmo_model.config.vocab_size}). "
                  f"Clear HF cache and re-download with transformers=={self.molmo_model.config.transformers_version}")

        if dtype in (torch.float16, torch.bfloat16):
            target_dtype = dtype
            self.molmo_model.model.vision_backbone.to(torch.float32)

            def _downcast_hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    return output.to(target_dtype)
                if isinstance(output, (tuple, list)):
                    return type(output)(
                        o.to(target_dtype) if isinstance(o, torch.Tensor) else o
                        for o in output
                    )
                return output

            self.molmo_model.model.vision_backbone.register_forward_hook(_downcast_hook)
            print(f"[GoalGenerator] Vision backbone float32 + downcast hook ({target_dtype}) applied")

        self._molmo_dtype = dtype

        print(f"[GoalGenerator] Molmo loaded on {self.device} (dtype={dtype})")

    def _load_sam(self, sam_path: str, variant: str):
        from sam2.build_sam import build_sam2_camera_predictor

        ckpt_mapping = {
            "large": ("sam2_hiera_large.pt", "sam2_hiera_l.yaml"),
            "base": ("sam2_hiera_base_plus.pt", "sam2_hiera_b+.yaml"),
            "small": ("sam2_hiera_small.pt", "sam2_hiera_s.yaml"),
            "tiny": ("sam2_hiera_tiny.pt", "sam2_hiera_t.yaml"),
        }
        ckpt_file, cfg_file = ckpt_mapping[variant]
        ckpt_full = os.path.join(sam_path, ckpt_file)
        self.sam_predictor = build_sam2_camera_predictor(
            cfg_file, ckpt_full, device=self.device
        )
        print(f"[GoalGenerator] SAM-2 ({variant}) loaded from {ckpt_full}")

    def _molmo_predict_point(
        self, image: np.ndarray, text: str
    ) -> Tuple[int, int]:
        """Run Molmo to get (x, y) point from text prompt on the image."""
        from PIL import Image as PILImage
        from transformers import GenerationConfig

        pil_img = PILImage.fromarray(image)
        inputs = self.molmo_processor.process(images=[pil_img], text=text)
        inputs = {
            k: v.to(self.device).unsqueeze(0) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        max_token_id = self.molmo_model.config.vocab_size - 1
        if inputs["input_ids"].max() > max_token_id:
            inputs["input_ids"] = inputs["input_ids"].clamp(max=max_token_id)

        gen_config = GenerationConfig(
            max_new_tokens=64,
            use_cache=True,
            do_sample=False,
            stop_strings=["<|endoftext|>"],
            repetition_penalty=1.2,
        )

        with torch.inference_mode():
            output = self.molmo_model.generate_from_batch(
                inputs,
                generation_config=gen_config,
                tokenizer=self.molmo_processor.tokenizer,
            )

        # Only decode the newly generated tokens (skip the input prompt)
        input_len = inputs["input_ids"].shape[1]
        generated_tokens = output[0, input_len:]
        generated_text = self.molmo_processor.tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        )
        print(f"[GoalGenerator] Molmo raw output: {generated_text[:200]}")
        return self._parse_point(generated_text, image.shape[1], image.shape[0])

    @staticmethod
    def _parse_point(text: str, img_w: int, img_h: int) -> Tuple[int, int]:
        """Parse Molmo output to extract (x, y) pixel coordinate."""
        import re

        pattern = r"x\d*\s*=\s*([\d.]+).*?y\d*\s*=\s*([\d.]+)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            x_frac = float(match.group(1)) / 100.0
            y_frac = float(match.group(2)) / 100.0
            return (
                int(np.clip(x_frac * img_w, 0, img_w - 1)),
                int(np.clip(y_frac * img_h, 0, img_h - 1)),
            )

        point_pattern = r"<point\s+x=\"([\d.]+)\"\s+y=\"([\d.]+)\""
        match = re.search(point_pattern, text)
        if match:
            x_frac = float(match.group(1)) / 100.0
            y_frac = float(match.group(2)) / 100.0
            return (
                int(np.clip(x_frac * img_w, 0, img_w - 1)),
                int(np.clip(y_frac * img_h, 0, img_h - 1)),
            )

        print(f"[GoalGenerator] WARNING: Could not parse point from: {text}")
        return None

    def generate(
        self,
        obs_image: np.ndarray,
        text: str,
    ) -> Tuple[Optional[Tuple[int, int]], np.ndarray]:
        """
        Returns (point, mask). If Molmo cannot find the target in the image,
        returns (None, zero_mask) to signal exploration mode.
        """
        point = self._molmo_predict_point(obs_image, text)
        h, w = obs_image.shape[:2]

        if point is None:
            return None, np.zeros((h, w), dtype=np.float32)

        self.sam_predictor.load_first_frame(obs_image)
        _, _, out_mask_logits = self.sam_predictor.add_new_prompt(
            frame_idx=0,
            obj_id=0,
            points=[list(point)],
            labels=[1],
        )
        mask = (out_mask_logits[0, 0] > 0.0).cpu().numpy().astype(np.float32)

        if mask.sum() < 10:
            print(f"[GoalGenerator] WARNING: SAM mask too small, using circular fallback")
            mask = np.zeros((h, w), dtype=np.float32)
            radius = int(min(h, w) * 0.1)
            cv2.circle(mask, point, radius, 1.0, -1)

        return point, mask


class GroundingDinoGoalGenerator(GoalGeneratorBase):
    """
    Uses GroundingDINO to locate target object, then SAM-2 to segment mask.

    Supports two asset modes:
      1) Local explicit paths: gdino_config + gdino_weights
      2) Best-effort auto-download from HuggingFace (if paths not provided)
    """

    def __init__(
        self,
        sam_path: str = "./MineStudio/minestudio/utils/realtime_sam/checkpoints",
        sam_variant: str = "base",
        gdino_config: Optional[str] = None,
        gdino_weights: Optional[str] = None,
        gdino_hf_repo: str = "ShilongLiu/GroundingDINO",
        box_threshold: float = 0.25,
        text_threshold: float = 0.20,
        device: Optional[str] = None,
    ):
        self.device = device or get_device()
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self._load_sam(sam_path, sam_variant)
        self._load_groundingdino(gdino_config, gdino_weights, gdino_hf_repo)

    def _load_sam(self, sam_path: str, variant: str):
        from sam2.build_sam import build_sam2_camera_predictor

        ckpt_mapping = {
            "large": ("sam2_hiera_large.pt", "sam2_hiera_l.yaml"),
            "base": ("sam2_hiera_base_plus.pt", "sam2_hiera_b+.yaml"),
            "small": ("sam2_hiera_small.pt", "sam2_hiera_s.yaml"),
            "tiny": ("sam2_hiera_tiny.pt", "sam2_hiera_t.yaml"),
        }
        ckpt_file, cfg_file = ckpt_mapping[variant]
        ckpt_full = os.path.join(sam_path, ckpt_file)
        self.sam_predictor = build_sam2_camera_predictor(
            cfg_file, ckpt_full, device=self.device
        )
        print(f"[GoalGenerator] SAM-2 ({variant}) loaded from {ckpt_full}")

    def _hf_try_download(self, repo_id: str, filenames: list) -> Optional[str]:
        try:
            from huggingface_hub import hf_hub_download
        except Exception:
            return None

        for name in filenames:
            try:
                return hf_hub_download(repo_id=repo_id, filename=name)
            except Exception:
                continue
        return None

    def _load_groundingdino(
        self,
        gdino_config: Optional[str],
        gdino_weights: Optional[str],
        gdino_hf_repo: str,
    ):
        from groundingdino.util.inference import load_model

        config_path = gdino_config
        weights_path = gdino_weights

        if not config_path or not os.path.exists(config_path):
            config_path = self._hf_try_download(
                gdino_hf_repo,
                [
                    "GroundingDINO_SwinT_OGC.py",
                    "groundingdino/config/GroundingDINO_SwinT_OGC.py",
                    "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                ],
            )

        if not weights_path or not os.path.exists(weights_path):
            weights_path = self._hf_try_download(
                gdino_hf_repo,
                [
                    "groundingdino_swint_ogc.pth",
                    "weights/groundingdino_swint_ogc.pth",
                ],
            )

        if not config_path or not os.path.exists(config_path):
            raise RuntimeError(
                "GroundingDINO config not found. Pass --gdino-config "
                "(e.g. /Users/nanzhang/aimc/data/weights/groundingdino/GroundingDINO_SwinT_OGC.py)."
            )
        if not weights_path or not os.path.exists(weights_path):
            raise RuntimeError(
                "GroundingDINO weights not found. Pass --gdino-weights "
                "(e.g. /Users/nanzhang/aimc/data/weights/groundingdino/groundingdino_swint_ogc.pth)."
            )

        print(f"[GoalGenerator] Loading GroundingDINO config: {config_path}")
        print(f"[GoalGenerator] Loading GroundingDINO weights: {weights_path}")
        self.gdino_model = load_model(config_path, weights_path).to(self.device).eval()

    @staticmethod
    def _normalize_caption(text: str) -> str:
        caption = text.strip()
        if not caption.endswith("."):
            caption += "."
        return caption

    def _gdino_predict_point(
        self, image: np.ndarray, text: str
    ) -> Tuple[Optional[Tuple[int, int]], Optional[np.ndarray], float]:
        from PIL import Image as PILImage
        from groundingdino.datasets import transforms as T
        from groundingdino.util.inference import predict

        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        pil_img = PILImage.fromarray(image)
        image_tensor, _ = transform(pil_img, None)
        caption = self._normalize_caption(text)

        boxes, logits, phrases = predict(
            model=self.gdino_model,
            image=image_tensor,
            caption=caption,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            device=self.device,
        )
        if boxes is None or len(boxes) == 0:
            print(f"[GoalGenerator] GroundingDINO no detection for: {caption}")
            return None, None, 0.0

        boxes_np = np.asarray(boxes)
        logits_np = np.asarray(logits).reshape(-1)
        best_idx = int(np.argmax(logits_np))
        best_box = boxes_np[best_idx]
        best_score = float(logits_np[best_idx])
        best_phrase = str(list(phrases)[best_idx]) if phrases is not None else "unknown"

        h, w = image.shape[:2]
        if np.max(best_box) <= 1.5:
            cx, cy, bw, bh = [float(x) for x in best_box.tolist()]
            x1 = (cx - bw / 2.0) * w
            y1 = (cy - bh / 2.0) * h
            x2 = (cx + bw / 2.0) * w
            y2 = (cy + bh / 2.0) * h
        else:
            x1, y1, x2, y2 = [float(x) for x in best_box.tolist()]

        x1 = float(np.clip(x1, 0, w - 1))
        y1 = float(np.clip(y1, 0, h - 1))
        x2 = float(np.clip(x2, 0, w - 1))
        y2 = float(np.clip(y2, 0, h - 1))
        cx = int(round((x1 + x2) / 2.0))
        cy = int(round((y1 + y2) / 2.0))
        bbox = np.array([x1, y1, x2, y2], dtype=np.float32)

        print(
            f"[GoalGenerator] GroundingDINO best: phrase='{best_phrase}', score={best_score:.3f}, "
            f"bbox=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}), point=({cx},{cy})"
        )
        return (cx, cy), bbox, best_score

    def generate(
        self,
        obs_image: np.ndarray,
        text: str,
    ) -> Tuple[Optional[Tuple[int, int]], np.ndarray]:
        point, bbox, score = self._gdino_predict_point(obs_image, text)
        h, w = obs_image.shape[:2]

        if point is None:
            return None, np.zeros((h, w), dtype=np.float32)

        self.sam_predictor.load_first_frame(obs_image)
        _, _, out_mask_logits = self.sam_predictor.add_new_prompt(
            frame_idx=0,
            obj_id=0,
            points=[list(point)],
            labels=[1],
        )
        mask = (out_mask_logits[0, 0] > 0.0).cpu().numpy().astype(np.float32)

        if mask.sum() < 10:
            print(
                f"[GoalGenerator] WARNING: SAM mask too small from DINO point "
                f"(score={score:.3f}), using bbox fallback mask"
            )
            x1, y1, x2, y2 = bbox.astype(np.int32).tolist() if bbox is not None else [0, 0, 0, 0]
            mask = np.zeros((h, w), dtype=np.float32)
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 1.0
            else:
                radius = int(min(h, w) * 0.1)
                cv2.circle(mask, point, radius, 1.0, -1)

        return point, mask
