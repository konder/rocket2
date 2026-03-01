"""
Compare goal detection across Molmo, GroundingDINO, and Qwen3.5-27B.

For each task's first-frame screenshot, run all selected backends and
draw bounding boxes / points on the image with scores.

Usage:
    python benchmark/compare_detectors.py \
        --task-file benchmark/eval_tasks_paper.yaml \
        --image-dir benchmark/first_frames/ \
        --output-dir benchmark/detection_compare/ \
        --backends molmo groundingdino qwen \
        --device cuda

    # Single task, single backend:
    python benchmark/compare_detectors.py \
        --task-file benchmark/eval_tasks_paper.yaml \
        --image-dir benchmark/first_frames/ \
        --output-dir benchmark/detection_compare/ \
        --tasks mine_coal \
        --backends groundingdino

Qwen3.5-27B requires latest transformers:
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
    "molmo": (0, 0, 255),       # red (BGR)
    "groundingdino": (0, 200, 0),  # green
    "qwen": (255, 100, 0),      # blue
}

DISPLAY_NAMES = {
    "molmo": "Molmo-7B",
    "groundingdino": "GroundingDINO",
    "qwen": "Qwen3.5-27B",
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
    def __init__(self, model_id: str, device: str):
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

        print(f"[Molmo] Ready on {device} (dtype={dtype})")

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
        r = 30
        return [{
            "label": "molmo_point",
            "score": 1.0,
            "bbox": [max(0, x - r), max(0, y - r), min(w - 1, x + r), min(h - 1, y + r)],
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
# Qwen VL backend — API mode (SGLang / vLLM) or local transformers
# ---------------------------------------------------------------------------
class QwenVLDetector:
    def __init__(self, model_id: str, device: str,
                 quant: Optional[str] = None, api_base: Optional[str] = None):
        self.model_id = model_id
        self.api_base = api_base

        if api_base:
            print(f"[QwenVL] API mode → {api_base}  model={model_id}")
            self.model = None
            self.processor = None
        else:
            from transformers import AutoProcessor
            self.device = device
            dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
            if device == "cpu":
                dtype = torch.float32
            print(f"[QwenVL] Loading {model_id} locally (quant={quant or 'none'}) ...")
            self.model = self._load_model_local(model_id, dtype, quant)
            self.processor = AutoProcessor.from_pretrained(model_id)
            print(f"[QwenVL] Ready locally (dtype={dtype})")

    @staticmethod
    def _load_model_local(model_id: str, dtype, quant: Optional[str] = None):
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        model_type = getattr(config, "model_type", "")
        print(f"[QwenVL] model_type: {model_type}")

        extra = {"torch_dtype": dtype}
        if quant in ("4bit", "4"):
            from transformers import BitsAndBytesConfig
            extra["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            )
        elif quant in ("8bit", "8"):
            from transformers import BitsAndBytesConfig
            extra["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        if torch.cuda.is_available():
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            extra["max_memory"] = {0: f"{int(gpu_mem * 0.85)}GiB", "cpu": "48GiB"}

        is_vl = any(t in model_type for t in ["qwen2_5_vl", "qwen2_vl", "qwen3_vl"])
        if is_vl:
            from transformers import AutoModelForImageTextToText
            print(f"[QwenVL] Loading as AutoModelForImageTextToText ({model_type})")
            return AutoModelForImageTextToText.from_pretrained(
                model_id, device_map="auto", **extra,
            ).eval()
        elif "qwen3" in model_type:
            from transformers import AutoModelForCausalLM
            print(f"[QwenVL] Loading as AutoModelForCausalLM ({model_type})")
            return AutoModelForCausalLM.from_pretrained(
                model_id, device_map="auto", trust_remote_code=True, **extra,
            ).eval()
        else:
            from transformers import AutoModelForImageTextToText
            print(f"[QwenVL] Loading as AutoModelForImageTextToText (generic)")
            return AutoModelForImageTextToText.from_pretrained(
                model_id, device_map="auto", trust_remote_code=True, **extra,
            ).eval()

    def _build_prompt(self, prompt: str, w: int, h: int) -> str:
        obj_name = prompt.strip()
        for prefix in ["Point to the", "Point to"]:
            if obj_name.lower().startswith(prefix.lower()):
                obj_name = obj_name[len(prefix):].strip()
                break
        return (
            f'Detect {obj_name} in this image and return the bounding box coordinates. '
            f'Output format: {{"bbox_2d": [x1, y1, x2, y2], "label": "{obj_name}"}}'
        )

    def detect(self, image: np.ndarray, prompt: str) -> List[Dict]:
        h, w = image.shape[:2]
        if self.api_base:
            return self._detect_api(image, prompt, w, h)
        return self._detect_local(image, prompt, w, h)

    def _detect_api(self, image: np.ndarray, prompt: str, w: int, h: int) -> List[Dict]:
        import base64
        from openai import OpenAI

        _, buf = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        b64 = base64.b64encode(buf).decode("utf-8")

        client = OpenAI(base_url=self.api_base, api_key="EMPTY")
        text_prompt = self._build_prompt(prompt, w, h)

        resp = client.chat.completions.create(
            model=self.model_id,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": text_prompt},
                ],
            }],
            max_tokens=1024,
            temperature=0.7,
            top_p=0.8,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        raw_text = resp.choices[0].message.content or ""
        raw_text = self._strip_thinking(raw_text)
        print(f"  [QwenVL-API] raw: {raw_text[:500]}")
        return self._parse_detections(raw_text, w, h)

    def _detect_local(self, image: np.ndarray, prompt: str, w: int, h: int) -> List[Dict]:
        from PIL import Image as PILImage

        pil = PILImage.fromarray(image)
        text_prompt = self._build_prompt(prompt, w, h)
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": text_prompt},
        ]}]

        try:
            dev = self.model.device
        except Exception:
            dev = "cuda" if torch.cuda.is_available() else "cpu"

        # Method 1: apply_chat_template with images= kwarg (Qwen2.5-VL style)
        inputs = None
        for kwargs in [
            {"images": [pil], "enable_thinking": False},
            {"images": [pil]},
            {},
        ]:
            try:
                inputs = self.processor.apply_chat_template(
                    messages, add_generation_prompt=True,
                    tokenize=True, return_dict=True, return_tensors="pt",
                    **kwargs,
                )
                break
            except (TypeError, KeyError):
                continue

        # Method 2: two-step — text template then processor call
        if inputs is None:
            print(f"  [QwenVL] Falling back to two-step processing")
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = self.processor(
                text=[text], images=[pil], return_tensors="pt",
            )

        inputs = {k: v.to(dev) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        with torch.inference_mode():
            out_ids = self.model.generate(**inputs, max_new_tokens=2048)

        trimmed = [o[len(i):] for i, o in zip(inputs["input_ids"], out_ids)]
        raw_text = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        print(f"  [QwenVL] raw (before strip): {raw_text[:300]}")
        raw_text = self._strip_thinking(raw_text)
        print(f"  [QwenVL] raw (after strip):  {raw_text[:500]}")
        return self._parse_detections(raw_text, w, h)

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """Remove <think>...</think> block from Qwen3.5 output."""
        m = re.search(r'</think>\s*', text)
        if m:
            return text[m.end():]
        return text

    @staticmethod
    def _parse_detections(text: str, w: int, h: int) -> List[Dict]:
        results = []

        code_match = re.search(r'```(?:json)?\s*(.*?)```', text, re.DOTALL)
        parse_text = code_match.group(1).strip() if code_match else text

        def _extract_item(item: dict) -> Optional[Dict]:
            bbox = item.get("bbox_2d", item.get("bbox", None))
            if not bbox or len(bbox) != 4:
                return None
            x1, y1, x2, y2 = [float(v) for v in bbox]
            # Qwen VL may output coords in 0-1000 normalized range
            if all(0 <= v <= 1000 for v in [x1, y1, x2, y2]) and max(x1, y1, x2, y2) > 1.5:
                if x2 <= 1.0 and y2 <= 1.0:
                    pass  # already in pixel range [0,1]
                else:
                    x1, y1, x2, y2 = x1 / 1000 * w, y1 / 1000 * h, x2 / 1000 * w, y2 / 1000 * h
            x1 = np.clip(x1, 0, w - 1)
            y1 = np.clip(y1, 0, h - 1)
            x2 = np.clip(x2, 0, w - 1)
            y2 = np.clip(y2, 0, h - 1)
            score = float(item.get("score", item.get("confidence", 1.0)))
            label = item.get("label", "?")
            return {
                "label": label, "score": score,
                "bbox": [x1, y1, x2, y2],
                "point": (int((x1 + x2) / 2), int((y1 + y2) / 2)),
            }

        # Try JSON array
        json_match = re.search(r'\[.*\]', parse_text, re.DOTALL)
        if json_match:
            try:
                items = json.loads(json_match.group())
                if isinstance(items, list):
                    for item in items:
                        r = _extract_item(item)
                        if r:
                            results.append(r)
                    if results:
                        return results
            except json.JSONDecodeError:
                pass

        # Try single JSON object { "bbox_2d": ... }
        json_obj_match = re.search(r'\{[^{}]*"bbox_2d"[^{}]*\}', parse_text)
        if json_obj_match:
            try:
                item = json.loads(json_obj_match.group())
                r = _extract_item(item)
                if r:
                    results.append(r)
                    return results
            except json.JSONDecodeError:
                pass

        # Fallback: regex for bbox_2d values
        for m in re.finditer(
            r'"?bbox_2d"?\s*:\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]',
            parse_text
        ):
            r = _extract_item({"bbox_2d": [m.group(1), m.group(2), m.group(3), m.group(4)]})
            if r:
                results.append(r)
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
        if score < 1.0:
            tag += f" {score:.2f}"
        if label and label != "molmo_point":
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
                        choices=["molmo", "groundingdino", "qwen"],
                        default=["molmo", "groundingdino", "qwen"])
    parser.add_argument("--device", default=None)
    # Molmo
    parser.add_argument("--molmo-model", default="allenai/Molmo-7B-D-0924")
    # GroundingDINO
    parser.add_argument("--gdino-config",
                        default="/opt/conda/envs/rocket2/lib/python3.10/site-packages/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument("--gdino-weights", default=None)
    parser.add_argument("--gdino-box-threshold", type=float, default=0.25)
    parser.add_argument("--gdino-text-threshold", type=float, default=0.20)
    # Qwen
    parser.add_argument("--qwen-model", default="Qwen/Qwen3.5-27B")
    parser.add_argument("--qwen-quant", default="none", choices=["4bit", "8bit", "none"],
                        help="Quantization (only for local loading, not API mode)")
    parser.add_argument("--qwen-api", default=None,
                        help="OpenAI-compatible API base URL (e.g. http://localhost:8000/v1). "
                             "Use with SGLang/vLLM. Skips local model loading.")
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
    if "molmo" in args.backends:
        detectors["molmo"] = MolmoDetector(args.molmo_model, device)
    if "groundingdino" in args.backends:
        detectors["groundingdino"] = GDinoDetector(
            args.gdino_config, args.gdino_weights,
            args.gdino_box_threshold, args.gdino_text_threshold, device,
        )
    if "qwen" in args.backends:
        q = args.qwen_quant if args.qwen_quant != "none" else None
        detectors["qwen"] = QwenVLDetector(
            args.qwen_model, device, quant=q, api_base=args.qwen_api,
        )

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
