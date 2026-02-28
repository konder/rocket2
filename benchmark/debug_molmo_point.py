"""
Quick Molmo point-debug script.

Usage:
    python benchmark/debug_molmo_point.py \
        --image /path/to/frame.png \
        --text "Point to the coal ore" \
        --output benchmark/results/molmo_point_debug.png
"""

import argparse
import os
import re
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig


def get_device(device_arg: Optional[str]) -> str:
    if device_arg:
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_point(text: str, img_w: int, img_h: int) -> Optional[Tuple[int, int]]:
    # Pattern style: x0="37.4" y0="97.1"
    pattern = r'x\d*\s*=\s*"?(?P<x>[\d.]+)"?.*?y\d*\s*=\s*"?(?P<y>[\d.]+)"?'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        x_frac = float(match.group("x")) / 100.0
        y_frac = float(match.group("y")) / 100.0
        return (
            int(np.clip(x_frac * img_w, 0, img_w - 1)),
            int(np.clip(y_frac * img_h, 0, img_h - 1)),
        )

    # Pattern style: <point x="37.4" y="97.1" ...>
    point_pattern = r'<point\s+x="([\d.]+)"\s+y="([\d.]+)"'
    match = re.search(point_pattern, text, re.IGNORECASE)
    if match:
        x_frac = float(match.group(1)) / 100.0
        y_frac = float(match.group(2)) / 100.0
        return (
            int(np.clip(x_frac * img_w, 0, img_w - 1)),
            int(np.clip(y_frac * img_h, 0, img_h - 1)),
        )
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Input RGB image path")
    parser.add_argument("--text", default="Point to the coal ore", help="Molmo prompt text")
    parser.add_argument("--model-id", default="allenai/Molmo-7B-D-0924")
    parser.add_argument("--output", default="benchmark/results/molmo_point_debug.png")
    parser.add_argument("--device", default=None, help="cpu/cuda/mps (auto if omitted)")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    device = get_device(args.device)
    if device == "cpu":
        dtype = torch.float32
    elif device == "cuda" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    print(f"[MolmoDebug] device={device}, dtype={dtype}")
    print(f"[MolmoDebug] loading model: {args.model_id}")
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(device).eval()

    # Keep vision backbone in float32 for stability, downcast outputs back to target dtype.
    if dtype in (torch.float16, torch.bfloat16):
        target_dtype = dtype
        model.model.vision_backbone.to(torch.float32)

        def _downcast_hook(module, input_tensor, output_tensor):
            if isinstance(output_tensor, torch.Tensor):
                return output_tensor.to(target_dtype)
            if isinstance(output_tensor, (tuple, list)):
                return type(output_tensor)(
                    o.to(target_dtype) if isinstance(o, torch.Tensor) else o
                    for o in output_tensor
                )
            return output_tensor

        model.model.vision_backbone.register_forward_hook(_downcast_hook)

    img = Image.open(args.image).convert("RGB")
    img_np = np.array(img)
    h, w = img_np.shape[:2]

    inputs = processor.process(images=[img], text=args.text)
    inputs = {
        k: v.to(device).unsqueeze(0) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }

    # Workaround for tokenizer-vocab drift in remote Molmo assets.
    max_token_id = model.config.vocab_size - 1
    if isinstance(inputs.get("input_ids"), torch.Tensor):
        if inputs["input_ids"].max() > max_token_id:
            print(
                f"[MolmoDebug] clamp input_ids to vocab range [0, {max_token_id}] "
                f"(tokenizer/model mismatch workaround)"
            )
            inputs["input_ids"] = inputs["input_ids"].clamp(max=max_token_id)

    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        use_cache=True,
        do_sample=False,
        stop_strings=["<|endoftext|>"],
        repetition_penalty=1.2,
    )

    with torch.inference_mode():
        output = model.generate_from_batch(
            inputs,
            generation_config=gen_config,
            tokenizer=processor.tokenizer,
        )

    input_len = inputs["input_ids"].shape[1]
    generated_tokens = output[0, input_len:]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print(f"[MolmoDebug] raw output: {generated_text}")

    point = parse_point(generated_text, w, h)
    vis = img_np.copy()
    if point is not None:
        cv2.drawMarker(
            vis,
            point,
            color=(0, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=24,
            thickness=2,
            line_type=cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            f"point={point}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        print(f"[MolmoDebug] parsed point: {point}")
    else:
        cv2.putText(
            vis,
            "point=None",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        print("[MolmoDebug] parse failed: point=None")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    cv2.imwrite(args.output, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print(f"[MolmoDebug] saved: {args.output}")


if __name__ == "__main__":
    main()
