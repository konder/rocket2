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
from typing import Tuple, Optional


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

        # CPU needs float32; MPS and CUDA can use float16 (saves ~50% memory)
        dtype = torch.float32 if self.device == "cpu" else torch.float16

        print(f"[GoalGenerator] Loading Molmo from {model_id} ...")
        self.molmo_processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True
        )
        self.molmo_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(self.device).eval()

        if self.device == "mps" and dtype == torch.float16:
            # MPS float16 causes type mismatch in vision backbone layer norms
            # (float32 intermediates vs float16 weights). Fix: compute vision
            # backbone in float32, then downcast output to float16 via hook
            # to avoid both the type mismatch and MPS's 4GB single-tensor limit.
            self.molmo_model.model.vision_backbone.to(torch.float32)

            def _downcast_hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    return output.to(torch.float16)
                if isinstance(output, (tuple, list)):
                    return type(output)(
                        o.to(torch.float16) if isinstance(o, torch.Tensor) else o
                        for o in output
                    )
                return output

            self.molmo_model.model.vision_backbone.register_forward_hook(_downcast_hook)
            print(f"[GoalGenerator] MPS: vision backbone float32 + downcast hook")

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
        return (img_w // 2, img_h // 2)

    def generate(
        self,
        obs_image: np.ndarray,
        text: str,
    ) -> Tuple[Tuple[int, int], np.ndarray]:
        point = self._molmo_predict_point(obs_image, text)
        h, w = obs_image.shape[:2]

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
