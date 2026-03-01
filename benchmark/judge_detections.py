"""
Use a VLM (Qwen3-VL) to judge which detection candidate is best.

Reads detection_results.json from compare_detectors.py, draws all
candidates as numbered boxes on the original image, and asks the
VLM to pick the best one for each task.

Usage (local model):
    python benchmark/judge_detections.py \
        --task-file benchmark/eval_tasks_paper.yaml \
        --image-dir benchmark/first_frames/ \
        --results-json benchmark/detection_compare/molmo_dino/detection_results.json \
        --output-dir benchmark/judge_results/ \
        --model Qwen/Qwen3-VL-8B-Instruct-FP8 \
        --device cuda

Usage (API mode via SGLang/vLLM):
    python benchmark/judge_detections.py \
        --task-file benchmark/eval_tasks_paper.yaml \
        --image-dir benchmark/first_frames/ \
        --results-json benchmark/detection_compare/molmo_dino/detection_results.json \
        --output-dir benchmark/judge_results/ \
        --model Qwen/Qwen3-VL-8B-Instruct-FP8 \
        --api-base http://localhost:8000/v1
"""

import os
import sys
import json
import re
import argparse
import cv2
import numpy as np
import torch
from typing import List, Dict, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

BOX_COLORS = [
    (0, 0, 255),    # #1 red
    (0, 200, 0),    # #2 green
    (255, 100, 0),  # #3 blue
    (0, 200, 255),  # #4 yellow
    (200, 0, 200),  # #5 magenta
    (200, 200, 0),  # #6 cyan
    (128, 0, 255),  # #7 pink
    (0, 128, 255),  # #8 orange
]

SOURCE_NAMES = {
    "molmo": "Molmo",
    "groundingdino": "DINO",
    "qwen": "Qwen",
}


def draw_candidates(image_bgr: np.ndarray, candidates: List[Dict]) -> np.ndarray:
    vis = image_bgr.copy()
    for c in candidates:
        idx = c["idx"]
        color = BOX_COLORS[idx % len(BOX_COLORS)]
        x1, y1, x2, y2 = [int(v) for v in c["bbox"]]

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)

        if c.get("point"):
            px, py = c["point"]
            cv2.circle(vis, (px, py), 7, color, -1)
            cv2.circle(vis, (px, py), 9, (255, 255, 255), 2)

        tag = f"#{idx + 1} {c['source']}"
        if c.get("score", 1.0) < 1.0:
            tag += f" {c['score']:.2f}"

        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ty = max(y1 - 8, th + 6)
        cv2.rectangle(vis, (x1, ty - th - 6), (x1 + tw + 6, ty + 4), color, -1)
        cv2.putText(vis, tag, (x1 + 3, ty - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return vis


def build_judge_prompt(task_prompt: str, candidates: List[Dict]) -> str:
    lines = [
        f"Task: {task_prompt}",
        "",
        "The image shows a Minecraft game screenshot with numbered colored boxes.",
        "Each box is a candidate detection from a different model:",
        "",
    ]
    for c in candidates:
        lines.append(
            f"  Box #{c['idx'] + 1} ({c['source']}): "
            f"bbox=({c['bbox'][0]:.0f},{c['bbox'][1]:.0f},{c['bbox'][2]:.0f},{c['bbox'][3]:.0f})"
        )
    lines += [
        "",
        "Which box best matches the task target? If none is correct, reply 0.",
        "Reply with ONLY the box number (e.g. 1, 2, or 0).",
    ]
    return "\n".join(lines)


def parse_judge_answer(text: str, num_candidates: int) -> int:
    m = re.search(r'</think>\s*', text)
    if m:
        text = text[m.end():]

    nums = re.findall(r'\b(\d+)\b', text.strip())
    if nums:
        choice = int(nums[0])
        if 0 <= choice <= num_candidates:
            return choice
    return -1


class JudgeVLM:
    def __init__(self, model_id: str, device: str, api_base: Optional[str] = None):
        self.model_id = model_id
        self.api_base = api_base

        if api_base:
            print(f"[Judge] API mode → {api_base}")
            self.model = None
            self.processor = None
        else:
            from transformers import AutoModelForImageTextToText, AutoProcessor, AutoConfig
            config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            model_type = getattr(config, "model_type", "")

            dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
            if device == "cpu":
                dtype = torch.float32

            print(f"[Judge] Loading {model_id} (model_type={model_type}) ...")
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id, device_map="auto", torch_dtype=dtype,
            ).eval()
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.device = device
            print(f"[Judge] Ready on {device}")

    def judge(self, image_rgb: np.ndarray, prompt: str) -> str:
        if self.api_base:
            return self._judge_api(image_rgb, prompt)
        return self._judge_local(image_rgb, prompt)

    def _judge_api(self, image_rgb: np.ndarray, prompt: str) -> str:
        import base64
        from openai import OpenAI

        _, buf = cv2.imencode(".png", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        b64 = base64.b64encode(buf).decode("utf-8")

        client = OpenAI(base_url=self.api_base, api_key="EMPTY")
        resp = client.chat.completions.create(
            model=self.model_id,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
            max_tokens=256,
            temperature=0.1,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return resp.choices[0].message.content or ""

    def _judge_local(self, image_rgb: np.ndarray, prompt: str) -> str:
        from PIL import Image as PILImage

        pil = PILImage.fromarray(image_rgb)
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]}]

        try:
            dev = self.model.device
        except Exception:
            dev = "cuda" if torch.cuda.is_available() else "cpu"

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

        if inputs is None:
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = self.processor(text=[text], images=[pil], return_tensors="pt")

        inputs = {k: v.to(dev) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        with torch.inference_mode():
            out_ids = self.model.generate(**inputs, max_new_tokens=256)

        trimmed = [o[len(i):] for i, o in zip(inputs["input_ids"], out_ids)]
        raw = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return raw


def main():
    parser = argparse.ArgumentParser(description="VLM judge for detection candidates")
    parser.add_argument("--task-file", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--results-json", required=True,
                        help="detection_results.json from compare_detectors.py")
    parser.add_argument("--output-dir", default="benchmark/judge_results/")
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct-FP8")
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top-k", type=int, default=2,
                        help="Use top-K detections from each backend as candidates")
    args = parser.parse_args()

    with open(args.results_json) as f:
        all_results = json.load(f)

    with open(args.task_file) as f:
        import yaml
        task_config = yaml.safe_load(f)
    task_names = {t["name"] for t in task_config["tasks"]}
    if args.tasks:
        task_names = set(args.tasks)

    os.makedirs(args.output_dir, exist_ok=True)
    judge = JudgeVLM(args.model, args.device, api_base=args.api_base)

    final_results = []

    for task_entry in all_results:
        task_name = task_entry["task"]
        if task_name not in task_names:
            continue

        prompt = task_entry["prompt"]
        img_path = os.path.join(args.image_dir, f"{task_name}.png")
        if not os.path.exists(img_path):
            print(f"[SKIP] {task_name}: image not found")
            continue

        print(f"\n{'='*60}")
        print(f"  Task: {task_name}")
        print(f"  Prompt: {prompt}")
        print(f"{'='*60}")

        candidates = []
        for backend in ["molmo", "groundingdino", "qwen"]:
            dets = task_entry.get(backend, [])
            source = SOURCE_NAMES.get(backend, backend)
            for det in dets[:args.top_k]:
                candidates.append({
                    "idx": len(candidates),
                    "source": source,
                    "backend": backend,
                    "label": det.get("label", "?"),
                    "score": det.get("score", 1.0),
                    "bbox": det["bbox"],
                    "point": det.get("point"),
                })

        if not candidates:
            print("  No candidates from any backend, skipping")
            final_results.append({
                "task": task_name, "prompt": prompt,
                "winner": None, "reason": "no_candidates",
            })
            continue

        print(f"  Candidates: {len(candidates)}")
        for c in candidates:
            print(f"    #{c['idx']+1} {c['source']}: "
                  f"bbox=({c['bbox'][0]:.0f},{c['bbox'][1]:.0f},"
                  f"{c['bbox'][2]:.0f},{c['bbox'][3]:.0f}) "
                  f"score={c['score']:.2f}")

        bgr = cv2.imread(img_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        annotated_bgr = draw_candidates(bgr, candidates)
        judge_prompt = build_judge_prompt(prompt, candidates)

        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        raw_answer = judge.judge(annotated_rgb, judge_prompt)
        print(f"  [Judge] raw: {raw_answer[:200]}")

        choice = parse_judge_answer(raw_answer, len(candidates))
        print(f"  [Judge] choice: #{choice}")

        winner = None
        if 1 <= choice <= len(candidates):
            winner = candidates[choice - 1]
            print(f"  >>> Winner: #{choice} {winner['source']} "
                  f"bbox=({winner['bbox'][0]:.0f},{winner['bbox'][1]:.0f},"
                  f"{winner['bbox'][2]:.0f},{winner['bbox'][3]:.0f})")

            cv2.rectangle(annotated_bgr, 
                          (int(winner['bbox'][0]), int(winner['bbox'][1])),
                          (int(winner['bbox'][2]), int(winner['bbox'][3])),
                          (255, 255, 255), 4)
            cv2.putText(annotated_bgr, "WINNER",
                        (int(winner['bbox'][0]), int(winner['bbox'][3]) + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        elif choice == 0:
            print(f"  >>> Judge: none correct")
        else:
            print(f"  >>> Judge: could not parse answer")

        cv2.putText(annotated_bgr, f"Task: {task_name} | {prompt}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        out_img = os.path.join(args.output_dir, f"{task_name}_judge.png")
        cv2.imwrite(out_img, annotated_bgr)
        print(f"  Saved: {out_img}")

        final_results.append({
            "task": task_name,
            "prompt": prompt,
            "num_candidates": len(candidates),
            "judge_raw": raw_answer.strip(),
            "judge_choice": choice,
            "winner": {
                "source": winner["source"],
                "backend": winner["backend"],
                "bbox": winner["bbox"],
                "point": winner["point"],
                "score": winner["score"],
            } if winner else None,
            "candidates": [
                {"idx": c["idx"] + 1, "source": c["source"],
                 "bbox": c["bbox"], "score": c["score"]}
                for c in candidates
            ],
        })

    json_path = os.path.join(args.output_dir, "judge_results.json")
    with open(json_path, "w") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    win_counts = {}
    for r in final_results:
        w = r.get("winner")
        src = w["source"] if w else "none"
        win_counts[src] = win_counts.get(src, 0) + 1
    for src, cnt in sorted(win_counts.items(), key=lambda x: -x[1]):
        print(f"  {src}: {cnt}/{len(final_results)} wins")

    print(f"\nResults: {json_path}")
    print(f"Done. {len(final_results)} tasks judged.")


if __name__ == "__main__":
    main()
