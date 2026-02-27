"""
Quick evaluation script for the SBD + VLM pipeline on raw YouTube Minecraft videos.

Prerequisites:
    # 1. Download VPT/IDM model weights (one-time):
    bash setup_sbd.sh

    # 2. Install gym3 (needed by VPT model):
    pip install gym3

Usage:
    # Run SBD (Skill Boundary Detection) on test videos
    python eval_sbd_vlm.py sbd --video-dir /path/to/videos --output-dir eval_gallery/sbd_results

    # Run VLM classification on detected segments
    python eval_sbd_vlm.py classify --sbd-dir eval_gallery/sbd_results --output-dir eval_gallery/vlm_results

    # Run SAM-2 backward tracking on classified events
    python eval_sbd_vlm.py track --vlm-dir eval_gallery/vlm_results --output-dir eval_gallery/tracking_results

    # Visualize full pipeline results
    python eval_sbd_vlm.py visualize --results-dir eval_gallery/tracking_results

    # Or run all steps at once:
    python eval_sbd_vlm.py all --video-dir /path/to/videos --output-dir eval_gallery/pipeline_results
"""

import os
import sys
import cv2
import json
import time
import torch
import pickle
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field, asdict

SBD_LIB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sbd_lib")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SkillSegment:
    video_path: str
    start_frame: int
    end_frame: int
    boundary_score: float = 0.0

@dataclass
class ClassifiedEvent:
    video_path: str
    event_frame: int
    start_frame: int
    end_frame: int
    event_type: str  # mine_block, kill_entity, use_item, craft_item, approach
    object_name: str = ""
    confidence: float = 0.0
    vlm_raw_response: str = ""

@dataclass
class TrackedEvent(ClassifiedEvent):
    mask_frames: Dict[int, np.ndarray] = field(default_factory=dict)
    point: Tuple[int, int] = (0, 0)


# ---------------------------------------------------------------------------
# Step 1: Video utilities
# ---------------------------------------------------------------------------

def extract_frames(video_path: str, start: int = 0, end: int = -1,
                   step: int = 1, resize: Tuple[int, int] = None) -> List[np.ndarray]:
    """Extract frames from video file. Returns list of RGB numpy arrays."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end < 0:
        end = total

    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for i in range(start, min(end, total)):
        ret, frame = cap.read()
        if not ret:
            break
        if (i - start) % step == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if resize:
                rgb = cv2.resize(rgb, resize)
            frames.append(rgb)
    cap.release()
    return frames


def get_video_info(video_path: str) -> Dict:
    cap = cv2.VideoCapture(video_path)
    info = {
        "path": video_path,
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    info["duration_sec"] = info["total_frames"] / max(info["fps"], 1)
    cap.release()
    return info


# ---------------------------------------------------------------------------
# Step 2: Real SBD (Skill Boundary Detection) using SkillDiscovery VPT
# ---------------------------------------------------------------------------

class RealSBD:
    """
    Real Skill Boundary Detection from CraftJarvis/SkillDiscovery.

    Uses VPT's action-prediction loss spikes: when the VPT model's prediction
    error suddenly increases, it indicates a skill/behavior transition.
    IDM predicts actions from raw video (no jsonl needed).
    """

    def __init__(self, device: str = None,
                 vpt_model: str = None, vpt_weights: str = None,
                 idm_model: str = None, idm_weights: str = None,
                 gap_threshold: float = 17.0,
                 min_segment_len: int = 2, max_segment_len: int = 12800):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.gap = gap_threshold
        self.min_seg = min_segment_len
        self.max_seg = max_segment_len

        self.vpt_model_path = vpt_model or os.path.join(
            SBD_LIB_DIR, "models", "foundation-model-3x.model")
        self.vpt_weights_path = vpt_weights or os.path.join(
            SBD_LIB_DIR, "weights", "bc-early-game-3x.weights")
        self.idm_model_path = idm_model or os.path.join(
            SBD_LIB_DIR, "models", "4x_idm.model")
        self.idm_weights_path = idm_weights or os.path.join(
            SBD_LIB_DIR, "weights", "4x_idm.weights")

        self.vpt_agent = None
        self.idm_agent = None

    def _check_weights(self):
        missing = []
        for p in [self.vpt_model_path, self.vpt_weights_path,
                  self.idm_model_path, self.idm_weights_path]:
            if not os.path.exists(p):
                missing.append(p)
        if missing:
            raise FileNotFoundError(
                f"Missing model weights:\n" +
                "\n".join(f"  - {p}" for p in missing) +
                "\n\nRun: bash setup_sbd.sh")

    def load(self):
        if self.vpt_agent is not None:
            return

        self._check_weights()

        sys.path.insert(0, SBD_LIB_DIR)
        from load_model_nojava import load_vpt_model, load_idm_model

        print(f"[SBD] Loading VPT model on {self.device} ...")
        self.vpt_agent = load_vpt_model(
            self.vpt_model_path, self.vpt_weights_path, self.device)
        print(f"[SBD] Loading IDM model on {self.device} ...")
        self.idm_agent = load_idm_model(
            self.idm_model_path, self.idm_weights_path, self.device)
        print(f"[SBD] Models loaded.")

    def _get_frames_and_actions(self, video_path: str,
                                max_frames: int = -1) -> List[Tuple]:
        """Read video, predict actions with IDM, filter no-ops."""
        cap = cv2.VideoCapture(video_path)
        raw_frames = []
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames > 0:
            total = min(total, max_frames)

        for _ in range(total):
            ret, frame = cap.read()
            if not ret:
                break
            raw_frames.append(frame[..., ::-1])  # BGR -> RGB
        cap.release()

        if not raw_frames:
            return []

        N_BATCH = 32
        result = []
        key_filter = {'ESC', 'swapHands', 'pickItem'}

        for start in range(0, len(raw_frames), N_BATCH):
            end = start + N_BATCH
            batch = np.stack(raw_frames[start:end])
            predicted = self.idm_agent.predict_actions(batch)

            for i in range(len(batch)):
                action = {k: v[0][i] for k, v in predicted.items()}
                has_action = False
                for k, v in action.items():
                    if k in key_filter:
                        continue
                    if not np.all(v == 0):
                        has_action = True
                        break
                if has_action:
                    result.append((start + i, batch[i], action, False))

        return result

    def detect_boundaries(self, video_path: str,
                          max_frames: int = -1) -> List[SkillSegment]:
        """Detect skill boundaries using VPT loss spikes."""
        self.load()

        from lib.tree_util import tree_map

        print(f"  [SBD] Reading frames & predicting actions via IDM ...")
        data = self._get_frames_and_actions(video_path, max_frames)
        if len(data) < 2:
            return []
        print(f"  [SBD] Got {len(data)} active frames (no-ops filtered)")

        policy = self.vpt_agent.policy
        agent_state = policy.initial_state(1)
        dummy_first = torch.from_numpy(np.array((False,))).to(self.device)

        losses = []
        boundaries_raw = []
        prev_id = 0
        prev_frame_id = 0
        flag = False

        print(f"  [SBD] Computing VPT loss spikes (GAP={self.gap}) ...")
        for idx, datum in enumerate(data):
            frame_id, image, action, isGuiOpen = datum
            agent_action = self.vpt_agent._env_action_to_agent(
                action, to_torch=True, check_if_null=True)
            agent_obs = self.vpt_agent._env_obs_to_agent({"pov": image})
            pi_dist, _, agent_state = policy.get_output_for_observation(
                agent_obs, agent_state, dummy_first)
            agent_state = tree_map(lambda x: x.detach(), agent_state)

            if agent_action is None:
                continue

            log_prob = policy.get_logprob_of_action(pi_dist, agent_action)
            discount = 0.75 if isGuiOpen else 1.0
            losses.append(-log_prob.item() * discount)

            if len(losses) > 1:
                avg_loss = sum(losses) / len(losses)
                if losses[-1] - avg_loss > self.gap:
                    flag = True

            if (idx - prev_id >= self.min_seg and
                    (flag or idx - prev_id >= self.max_seg)):
                boundaries_raw.append((prev_frame_id, frame_id - 1))
                agent_state = policy.initial_state(1)
                losses = []
                prev_id = idx
                prev_frame_id = frame_id
                flag = False

        if data:
            boundaries_raw.append((prev_frame_id, data[-1][0]))

        segments = []
        for start, end in boundaries_raw:
            if end > start:
                segments.append(SkillSegment(
                    video_path=video_path,
                    start_frame=start,
                    end_frame=end,
                    boundary_score=0.0,
                ))

        return segments


# ---------------------------------------------------------------------------
# Step 3: VLM Classification
# ---------------------------------------------------------------------------

MINECRAFT_EVENT_PROMPT = """You are analyzing a Minecraft gameplay video frame. Determine what interaction event (if any) is happening.

Classify into ONE of these categories:
1. "mine_block" - Player is mining/breaking a block (you can see block-breaking particles or cracks)
2. "kill_entity" - Player is attacking or has just killed a mob/animal
3. "use_item" - Player is using/placing an item (eating, placing block, using tool)
4. "craft_item" - Player has a crafting table or inventory UI open
5. "approach" - Player is walking toward a specific object/entity (no other interaction)
6. "none" - No clear interaction event is visible

Respond in this exact JSON format (no other text):
{"event": "<category>", "object": "<specific object name or empty>", "confidence": <0.0-1.0>}

Examples:
- Mining stone: {"event": "mine_block", "object": "stone", "confidence": 0.9}
- Killing a cow: {"event": "kill_entity", "object": "cow", "confidence": 0.85}
- Crafting table UI visible: {"event": "craft_item", "object": "crafting_table", "confidence": 0.95}
- Just walking around: {"event": "none", "object": "", "confidence": 0.8}
"""


class VLMClassifier:
    """VLM-based event classifier using Molmo (already in the codebase) or Qwen-VL."""

    def __init__(self, model_type: str = "molmo",
                 model_id: str = None,
                 device: str = None):
        self.model_type = model_type
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if model_id is None:
            if model_type == "molmo":
                model_id = "allenai/Molmo-7B-D-0924"
            elif model_type == "qwen":
                model_id = "Qwen/Qwen2-VL-7B-Instruct"
            else:
                raise ValueError(f"Unknown model_type: {model_type}")

        self.model_id = model_id
        self.model = None
        self.processor = None

    def load(self):
        """Lazy-load the model."""
        if self.model is not None:
            return

        print(f"[VLMClassifier] Loading {self.model_type} from {self.model_id} ...")

        if self.model_type == "molmo":
            self._load_molmo()
        elif self.model_type == "qwen":
            self._load_qwen()

        print(f"[VLMClassifier] Model loaded on {self.device}")

    def _load_molmo(self):
        from transformers import AutoModelForCausalLM, AutoProcessor
        dtype = torch.float16 if self.device != "cpu" else torch.float32
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, trust_remote_code=True,
            torch_dtype=dtype).to(self.device).eval()

    def _load_qwen(self):
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=torch.float16,
            device_map=self.device).eval()

    def classify_frame(self, frame: np.ndarray) -> Dict:
        """Classify a single frame. Returns dict with event, object, confidence."""
        self.load()
        if self.model_type == "molmo":
            return self._classify_molmo(frame)
        elif self.model_type == "qwen":
            return self._classify_qwen(frame)

    def _classify_molmo(self, frame: np.ndarray) -> Dict:
        from PIL import Image as PILImage
        from transformers import GenerationConfig
        import re as re_mod

        pil_img = PILImage.fromarray(frame)
        inputs = self.processor.process(
            images=[pil_img], text=MINECRAFT_EVENT_PROMPT)
        inputs = {
            k: v.to(self.device).unsqueeze(0) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        gen_config = GenerationConfig(
            max_new_tokens=128, use_cache=True, do_sample=False,
            stop_strings=["<|endoftext|>"],
        )

        with torch.inference_mode():
            output = self.model.generate_from_batch(
                inputs, generation_config=gen_config,
                tokenizer=self.processor.tokenizer)

        input_len = inputs["input_ids"].shape[1]
        text = self.processor.tokenizer.decode(
            output[0, input_len:], skip_special_tokens=True)

        return self._parse_response(text)

    def _classify_qwen(self, frame: np.ndarray) -> Dict:
        from PIL import Image as PILImage
        from qwen_vl_utils import process_vision_info

        pil_img = PILImage.fromarray(frame)
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": pil_img},
                {"type": "text", "text": MINECRAFT_EVENT_PROMPT},
            ],
        }]

        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text_input], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt").to(self.device)

        with torch.inference_mode():
            ids = self.model.generate(**inputs, max_new_tokens=128)

        trimmed = ids[0, inputs.input_ids.shape[1]:]
        text = self.processor.decode(trimmed, skip_special_tokens=True)

        return self._parse_response(text)

    @staticmethod
    def _parse_response(text: str) -> Dict:
        """Parse VLM JSON response, with fallback."""
        import re as re_mod
        text = text.strip()

        # Try to extract JSON from the response
        json_match = re_mod.search(r'\{[^}]+\}', text)
        if json_match:
            try:
                result = json.loads(json_match.group())
                return {
                    "event": result.get("event", "none"),
                    "object": result.get("object", ""),
                    "confidence": float(result.get("confidence", 0.5)),
                    "raw": text,
                }
            except json.JSONDecodeError:
                pass

        # Fallback: keyword matching
        text_lower = text.lower()
        for event in ["mine_block", "kill_entity", "craft_item", "use_item", "approach"]:
            if event.replace("_", " ") in text_lower or event in text_lower:
                return {"event": event, "object": "", "confidence": 0.3, "raw": text}

        return {"event": "none", "object": "", "confidence": 0.1, "raw": text}

    def classify_segment(self, video_path: str, start_frame: int,
                         end_frame: int, num_keyframes: int = 2) -> Dict:
        """Classify a video segment by sampling keyframes."""
        seg_len = end_frame - start_frame
        if seg_len <= 0:
            return {"event": "none", "object": "", "confidence": 0.0, "raw": ""}

        # Sample keyframes: last 1/3 of the segment (where interaction likely peaks)
        interaction_zone_start = start_frame + int(seg_len * 0.6)
        step = max((end_frame - interaction_zone_start) // (num_keyframes + 1), 1)
        keyframe_indices = [
            interaction_zone_start + step * (i + 1)
            for i in range(num_keyframes)
        ]
        keyframe_indices = [min(idx, end_frame - 1) for idx in keyframe_indices]

        results = []
        for idx in keyframe_indices:
            frames = extract_frames(video_path, idx, idx + 1)
            if frames:
                result = self.classify_frame(frames[0])
                result["frame_idx"] = idx
                results.append(result)

        if not results:
            return {"event": "none", "object": "", "confidence": 0.0, "raw": ""}

        # Pick the result with highest confidence that's not "none"
        event_results = [r for r in results if r["event"] != "none"]
        if event_results:
            best = max(event_results, key=lambda x: x["confidence"])
        else:
            best = max(results, key=lambda x: x["confidence"])

        return best


# ---------------------------------------------------------------------------
# Step 4: SAM-2 Backward Tracking
# ---------------------------------------------------------------------------

class SAM2Tracker:
    """SAM-2 backward tracking for generating segmentation masks."""

    def __init__(self, sam_path: str = "./MineStudio/minestudio/utils/realtime_sam/checkpoints",
                 variant: str = "tiny", device: str = None):
        self.sam_path = sam_path
        self.variant = variant
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.predictor = None

    def load(self):
        if self.predictor is not None:
            return

        sys.path.insert(0, "./MineStudio/minestudio/utils/realtime_sam")
        from sam2.build_sam import build_sam2_camera_predictor

        ckpt_mapping = {
            "large": ("sam2_hiera_large.pt", "sam2_hiera_l.yaml"),
            "base": ("sam2_hiera_base_plus.pt", "sam2_hiera_b+.yaml"),
            "small": ("sam2_hiera_small.pt", "sam2_hiera_s.yaml"),
            "tiny": ("sam2_hiera_tiny.pt", "sam2_hiera_t.yaml"),
        }
        ckpt_file, cfg_file = ckpt_mapping[self.variant]
        ckpt_full = os.path.join(self.sam_path, ckpt_file)
        self.predictor = build_sam2_camera_predictor(
            cfg_file, ckpt_full, device=self.device)
        print(f"[SAM2Tracker] Loaded {self.variant} from {ckpt_full}")

    def track_backward(self, video_path: str, event_frame: int,
                       start_frame: int, point: Tuple[int, int] = None,
                       max_track_frames: int = 60) -> Dict[int, np.ndarray]:
        """
        Given an event frame with an interaction point, track the object
        backward to generate masks for the preceding frames.

        Returns: {frame_idx: binary_mask} dict
        """
        self.load()

        track_start = max(start_frame, event_frame - max_track_frames)
        frames = extract_frames(video_path, track_start, event_frame + 1)
        if not frames:
            return {}

        event_local = len(frames) - 1

        # Auto-detect interaction point if not provided: use frame center
        if point is None:
            h, w = frames[event_local].shape[:2]
            point = (w // 2, h // 2)

        masks = {}

        # Forward pass: load event frame, get initial mask, then track backward
        self.predictor.load_first_frame(frames[event_local])
        _, _, out_mask_logits = self.predictor.add_new_prompt(
            frame_idx=0, obj_id=0,
            points=[list(point)], labels=[1])

        event_mask = (out_mask_logits[0, 0] > 0.0).cpu().numpy().astype(np.uint8)
        masks[event_frame] = event_mask

        # Track backward frame by frame using camera predictor
        for i in range(event_local - 1, -1, -1):
            real_frame = track_start + i
            out_obj_ids, out_mask_logits = self.predictor.track(frames[i])
            if len(out_mask_logits) > 0:
                mask = (out_mask_logits[0, 0] > 0.0).cpu().numpy().astype(np.uint8)
                if mask.sum() > 10:
                    masks[real_frame] = mask

        return masks


# ---------------------------------------------------------------------------
# Step 5: Visualization
# ---------------------------------------------------------------------------

def visualize_segment(video_path: str, event: ClassifiedEvent,
                      masks: Dict[int, np.ndarray],
                      output_path: str):
    """Create a visualization video for one detected event with masks overlaid."""
    start = max(0, event.start_frame)
    end = event.end_frame
    frames = extract_frames(video_path, start, end)
    if not frames:
        return

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, 20.0, (w, h))

    for i, frame in enumerate(frames):
        real_idx = start + i
        vis = frame.copy()

        # Overlay mask if available
        if real_idx in masks:
            mask = masks[real_idx]
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            color_overlay = np.zeros_like(vis)
            color_overlay[:, :, 1] = 255  # green
            vis = np.where(mask[..., None] > 0,
                           cv2.addWeighted(vis, 0.6, color_overlay, 0.4, 0),
                           vis)

        # Draw event info
        is_event_frame = (real_idx == event.event_frame)
        border_color = (0, 0, 255) if is_event_frame else (255, 255, 255)
        cv2.rectangle(vis, (2, 2), (w - 3, h - 3), border_color,
                      3 if is_event_frame else 1)

        label = f"[{event.event_type}] {event.object_name} ({event.confidence:.2f})"
        cv2.putText(vis, label, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis, f"Frame {real_idx}", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        writer.write(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    writer.release()
    print(f"  Saved visualization: {output_path}")


def create_summary_grid(events: List[ClassifiedEvent],
                        output_path: str):
    """Create a summary image grid of all detected events."""
    if not events:
        return

    grid_cols = min(4, len(events))
    grid_rows = (len(events) + grid_cols - 1) // grid_cols
    cell_w, cell_h = 320, 200
    grid = np.zeros((grid_rows * cell_h, grid_cols * cell_w, 3), dtype=np.uint8)

    for idx, ev in enumerate(events):
        row, col = idx // grid_cols, idx % grid_cols
        frames = extract_frames(ev.video_path, ev.event_frame, ev.event_frame + 1,
                                resize=(cell_w, cell_h))
        if frames:
            cell = frames[0].copy()
        else:
            cell = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)

        label = f"{ev.event_type}: {ev.object_name}"
        cv2.putText(cell, label, (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(cell, f"conf={ev.confidence:.2f} f={ev.event_frame}", (5, cell_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        y0, x0 = row * cell_h, col * cell_w
        grid[y0:y0 + cell_h, x0:x0 + cell_w] = cell

    cv2.imwrite(output_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"[Summary] Grid saved to {output_path}")


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

def run_sbd(video_dir: str, output_dir: str, max_frames: int = 6000,
            device: str = None, gap: float = 17.0) -> List[SkillSegment]:
    """Run real VPT-based SBD on all videos in a directory."""
    os.makedirs(output_dir, exist_ok=True)

    sbd = RealSBD(device=device, gap_threshold=gap)

    video_files = sorted(Path(video_dir).glob("*.mp4"))
    if not video_files:
        print(f"[SBD] No .mp4 files found in {video_dir}")
        return []

    all_segments = []
    for vi, vf in enumerate(video_files):
        print(f"\n[SBD] [{vi+1}/{len(video_files)}] Processing: {vf.name}")
        info = get_video_info(str(vf))
        print(f"  Duration: {info['duration_sec']:.1f}s, "
              f"Frames: {info['total_frames']}, FPS: {info['fps']:.1f}")

        t0 = time.time()
        segments = sbd.detect_boundaries(str(vf), max_frames=max_frames)
        elapsed = time.time() - t0

        print(f"  Detected {len(segments)} segments in {elapsed:.1f}s")
        for seg in segments:
            seg_len = seg.end_frame - seg.start_frame
            print(f"    [{seg.start_frame}-{seg.end_frame}] len={seg_len}")

        all_segments.extend(segments)

    save_path = os.path.join(output_dir, "sbd_segments.json")
    with open(save_path, "w") as f:
        json.dump([asdict(s) for s in all_segments], f, indent=2)
    print(f"\n[SBD] Saved {len(all_segments)} segments to {save_path}")

    return all_segments


def run_vlm_classification(segments: List[SkillSegment], output_dir: str,
                           model_type: str = "molmo",
                           model_id: str = None,
                           max_segments: int = 50) -> List[ClassifiedEvent]:
    """Run VLM classification on SBD segments."""
    os.makedirs(output_dir, exist_ok=True)

    classifier = VLMClassifier(model_type=model_type, model_id=model_id)

    events = []
    segments_to_process = segments[:max_segments]
    print(f"\n[VLM] Classifying {len(segments_to_process)} segments "
          f"(of {len(segments)} total) with {model_type} ...")

    for i, seg in enumerate(segments_to_process):
        t0 = time.time()
        result = classifier.classify_segment(
            seg.video_path, seg.start_frame, seg.end_frame)
        elapsed = time.time() - t0

        print(f"  [{i+1}/{len(segments_to_process)}] "
              f"frames [{seg.start_frame}-{seg.end_frame}] -> "
              f"{result['event']}:{result.get('object', '')} "
              f"(conf={result['confidence']:.2f}, {elapsed:.1f}s)")

        if result["event"] != "none" and result["confidence"] >= 0.5:
            # Use the keyframe from the last 1/3 as event frame
            event_frame = seg.start_frame + int(
                (seg.end_frame - seg.start_frame) * 0.8)
            events.append(ClassifiedEvent(
                video_path=seg.video_path,
                event_frame=event_frame,
                start_frame=seg.start_frame,
                end_frame=seg.end_frame,
                event_type=result["event"],
                object_name=result.get("object", ""),
                confidence=result["confidence"],
                vlm_raw_response=result.get("raw", ""),
            ))

    save_path = os.path.join(output_dir, "classified_events.json")
    serializable = []
    for ev in events:
        d = asdict(ev)
        d.pop("mask_frames", None)
        serializable.append(d)
    with open(save_path, "w") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)

    print(f"\n[VLM] Found {len(events)} interaction events "
          f"(out of {len(segments_to_process)} segments)")

    # Print distribution
    from collections import Counter
    dist = Counter(ev.event_type for ev in events)
    print(f"[VLM] Event distribution: {dict(dist)}")

    return events


def run_sam2_tracking(events: List[ClassifiedEvent], output_dir: str,
                      sam_variant: str = "tiny",
                      max_events: int = 20) -> List[TrackedEvent]:
    """Run SAM-2 backward tracking on classified events."""
    os.makedirs(output_dir, exist_ok=True)

    tracker = SAM2Tracker(variant=sam_variant)
    tracked = []

    events_to_process = events[:max_events]
    print(f"\n[SAM2] Tracking {len(events_to_process)} events "
          f"(of {len(events)} total) ...")

    for i, ev in enumerate(events_to_process):
        t0 = time.time()
        masks = tracker.track_backward(
            ev.video_path, ev.event_frame, ev.start_frame,
            max_track_frames=60)
        elapsed = time.time() - t0

        tracked_ev = TrackedEvent(
            video_path=ev.video_path,
            event_frame=ev.event_frame,
            start_frame=ev.start_frame,
            end_frame=ev.end_frame,
            event_type=ev.event_type,
            object_name=ev.object_name,
            confidence=ev.confidence,
            vlm_raw_response=ev.vlm_raw_response,
            mask_frames=masks,
            point=(0, 0),
        )

        # Compute point from event frame mask centroid
        if ev.event_frame in masks:
            m = masks[ev.event_frame]
            ys, xs = np.where(m > 0)
            if len(xs) > 0:
                tracked_ev.point = (int(xs.mean()), int(ys.mean()))

        tracked.append(tracked_ev)
        print(f"  [{i+1}/{len(events_to_process)}] "
              f"{ev.event_type}:{ev.object_name} -> "
              f"{len(masks)} mask frames, point={tracked_ev.point} "
              f"({elapsed:.1f}s)")

    return tracked


def run_visualization(events: List[ClassifiedEvent],
                      tracked_events: List[TrackedEvent],
                      output_dir: str):
    """Generate visualization outputs."""
    os.makedirs(output_dir, exist_ok=True)

    # Summary grid
    create_summary_grid(events,
                        os.path.join(output_dir, "event_summary_grid.png"))

    # Individual event videos with masks
    for i, tev in enumerate(tracked_events):
        vis_path = os.path.join(output_dir,
                                f"event_{i:03d}_{tev.event_type}_{tev.object_name}.mp4")
        visualize_segment(tev.video_path, tev, tev.mask_frames, vis_path)

    # Statistics report
    report = {
        "total_events": len(events),
        "tracked_events": len(tracked_events),
        "event_distribution": {},
        "avg_confidence": 0.0,
        "avg_masks_per_event": 0.0,
    }
    from collections import Counter
    dist = Counter(ev.event_type for ev in events)
    report["event_distribution"] = dict(dist)
    if events:
        report["avg_confidence"] = sum(e.confidence for e in events) / len(events)
    if tracked_events:
        report["avg_masks_per_event"] = sum(
            len(te.mask_frames) for te in tracked_events) / len(tracked_events)

    report_path = os.path.join(output_dir, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[Report] Saved evaluation report to {report_path}")
    print(json.dumps(report, indent=2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Quick evaluation of SBD + VLM pipeline for Minecraft video processing")
    subparsers = parser.add_subparsers(dest="command")

    # SBD
    p_sbd = subparsers.add_parser("sbd", help="Run VPT-based Skill Boundary Detection")
    p_sbd.add_argument("--video-dir", required=True)
    p_sbd.add_argument("--output-dir", default="eval_gallery/sbd_results")
    p_sbd.add_argument("--max-frames", type=int, default=6000,
                       help="Max frames per video to process (default: 6000 = ~3min@30fps)")
    p_sbd.add_argument("--gap", type=float, default=17.0,
                       help="VPT loss spike threshold (lower=more segments, default: 17.0)")
    p_sbd.add_argument("--device", default=None,
                       help="Device (cuda/cpu/mps, auto-detected if omitted)")

    # Classify
    p_cls = subparsers.add_parser("classify", help="Run VLM classification on SBD segments")
    p_cls.add_argument("--sbd-dir", required=True, help="Directory with sbd_segments.json")
    p_cls.add_argument("--output-dir", default="eval_gallery/vlm_results")
    p_cls.add_argument("--model-type", default="molmo", choices=["molmo", "qwen"])
    p_cls.add_argument("--model-id", default=None, help="Override model HF ID")
    p_cls.add_argument("--max-segments", type=int, default=50)

    # Track
    p_trk = subparsers.add_parser("track", help="Run SAM-2 backward tracking")
    p_trk.add_argument("--vlm-dir", required=True, help="Directory with classified_events.json")
    p_trk.add_argument("--output-dir", default="eval_gallery/tracking_results")
    p_trk.add_argument("--sam-variant", default="tiny", choices=["tiny", "small", "base", "large"])
    p_trk.add_argument("--max-events", type=int, default=20)

    # Visualize
    p_vis = subparsers.add_parser("visualize", help="Generate visualization outputs")
    p_vis.add_argument("--results-dir", required=True)

    # All-in-one
    p_all = subparsers.add_parser("all", help="Run full pipeline end-to-end")
    p_all.add_argument("--video-dir", required=True)
    p_all.add_argument("--output-dir", default="eval_gallery/pipeline_results")
    p_all.add_argument("--model-type", default="molmo", choices=["molmo", "qwen"])
    p_all.add_argument("--model-id", default=None)
    p_all.add_argument("--sam-variant", default="tiny", choices=["tiny", "small", "base", "large"])
    p_all.add_argument("--max-frames", type=int, default=6000)
    p_all.add_argument("--max-segments", type=int, default=50)
    p_all.add_argument("--max-events", type=int, default=20)
    p_all.add_argument("--gap", type=float, default=17.0)
    p_all.add_argument("--device", default=None,
                       help="Device (cuda/cpu/mps, auto-detected if omitted)")

    args = parser.parse_args()

    if args.command == "sbd":
        run_sbd(args.video_dir, args.output_dir, args.max_frames,
                device=args.device, gap=args.gap)

    elif args.command == "classify":
        sbd_path = os.path.join(args.sbd_dir, "sbd_segments.json")
        with open(sbd_path) as f:
            segments = [SkillSegment(**s) for s in json.load(f)]
        run_vlm_classification(segments, args.output_dir,
                               args.model_type, args.model_id, args.max_segments)

    elif args.command == "track":
        vlm_path = os.path.join(args.vlm_dir, "classified_events.json")
        with open(vlm_path) as f:
            events = [ClassifiedEvent(**e) for e in json.load(f)]
        run_sam2_tracking(events, args.output_dir,
                          args.sam_variant, args.max_events)

    elif args.command == "visualize":
        # Load events and generate vis
        vlm_path = os.path.join(args.results_dir, "classified_events.json")
        if os.path.exists(vlm_path):
            with open(vlm_path) as f:
                events = [ClassifiedEvent(**e) for e in json.load(f)]
            create_summary_grid(events,
                                os.path.join(args.results_dir, "event_summary_grid.png"))
        else:
            print(f"[Visualize] No classified_events.json found in {args.results_dir}")

    elif args.command == "all":
        print("=" * 60)
        print("  SBD + VLM Pipeline Quick Evaluation")
        print("=" * 60)
        t_start = time.time()

        # Step 1: SBD
        sbd_dir = os.path.join(args.output_dir, "sbd")
        segments = run_sbd(args.video_dir, sbd_dir, args.max_frames,
                           device=args.device, gap=args.gap)

        if not segments:
            print("[Pipeline] No segments detected. Check your videos.")
            return

        # Step 2: VLM Classification
        vlm_dir = os.path.join(args.output_dir, "vlm")
        events = run_vlm_classification(
            segments, vlm_dir, args.model_type, args.model_id, args.max_segments)

        if not events:
            print("[Pipeline] No events classified. Try lowering confidence threshold.")
            return

        # Step 3: SAM-2 Tracking
        track_dir = os.path.join(args.output_dir, "tracking")
        tracked = run_sam2_tracking(events, track_dir,
                                    args.sam_variant, args.max_events)

        # Step 4: Visualization
        vis_dir = os.path.join(args.output_dir, "visualization")
        run_visualization(events, tracked, vis_dir)

        elapsed = time.time() - t_start
        print(f"\n{'=' * 60}")
        print(f"  Pipeline completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f"  Segments: {len(segments)} -> Events: {len(events)} -> Tracked: {len(tracked)}")
        print(f"  Results: {args.output_dir}/")
        print(f"{'=' * 60}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
