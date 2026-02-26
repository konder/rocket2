'''
Interactive Evaluator for ROCKET-2

Launch a web-based interactive evaluation interface for the ROCKET-2 model.
Supports real-time Minecraft rendering, goal setting via SAM-2 segmentation,
and configurable step-by-step model execution.

The evaluator runs as a global singleton so that browser refreshes reconnect
to the same Minecraft session instead of losing state.

Usage:
    python interactive_eval.py --sam-path /path/to/sam2/checkpoints --env-conf ./env_conf
'''

import os
import cv2
import time
import yaml
import json
import torch
import numpy as np
import gradio as gr
import argparse
from PIL import Image
from pathlib import Path


def get_device() -> str:
    env_device = os.environ.get("ROCKET_DEVICE")
    if env_device:
        return env_device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = get_device()

from sam2.build_sam import build_sam2_camera_predictor as build_sam2_predictor
from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import (
    load_callbacks_from_config, PrevActionCallback,
)
from model import CrossViewRocket, load_cross_view_rocket
from cfg_wrapper import CFGWrapper
from draw_action import draw_action

SCALE = 360 / 640
PAGE_WIDTH = 800
PAGE_HEIGHT = int(PAGE_WIDTH * SCALE)

COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (255, 255, 255), (0, 0, 0), (128, 128, 128),
    (128, 0, 0), (128, 128, 0), (0, 128, 0),
    (128, 0, 128), (0, 128, 128), (0, 0, 128),
]

INTERACTION_TYPES = {
    "Hunt": 0, "Use": 3, "Mine": 2, "Interact": 3,
    "Craft": 4, "Switch": 5, "Approach": 6, "None": -1,
}

DEFAULT_ENV_CONFIG = """\
spawn_positions:
  -
    seed: 19961103
    position: [-79, 64, -512]
  -
    seed: 19961103
    position: [-145, 67, -495]
  -
    seed: 19961103
    position: [-312, 69, -509]

init_inventory:
  -
    slot: 0
    type: iron_axe
    quantity: 1

masked_actions:
  hotbar.1: 0
  hotbar.2: 0
  hotbar.3: 0
  hotbar.4: 0
  hotbar.5: 0
  hotbar.6: 0
  hotbar.7: 0
  hotbar.8: 0
  hotbar.9: 0
"""

# Global singleton — survives browser refreshes
_evaluator: "InteractiveEvaluator | None" = None


def _overlay_thumbnail(bg_image, fg_image, scale=0.33):
    """Overlay a cross-view thumbnail in the top-right corner of the main frame."""
    fg = cv2.copyMakeBorder(fg_image, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    fg = cv2.resize(fg, (int(640 * scale), int(360 * scale)), interpolation=cv2.INTER_LINEAR)
    sx = bg_image.shape[1] - 5 - fg.shape[1]
    sy = 5
    bg_image[sy:sy + fg.shape[0], sx:sx + fg.shape[1]] = fg
    return bg_image


class InteractiveEvaluator:
    """Core evaluation session managing Minecraft env, ROCKET-2 model, and SAM-2."""

    def __init__(self, sam_path: str):
        self.sam_path = sam_path
        self.current_image = np.zeros((360, 640, 3), dtype=np.uint8)
        self.cross_view_image = np.zeros((360, 640, 3), dtype=np.uint8)
        self.obj_mask = np.zeros((360, 640), dtype=np.float32)
        self.points = []
        self.points_label = []
        self.segment_type = "None"
        self.sam_choice = "base"
        self.tracking_flag = True
        self.able_to_track = False
        self.num_steps = 0
        self.env = None
        self.env_conf = None
        self.obs_history = []
        self.image_history = []
        self.running = False

        self.load_sam()

    def load_sam(self):
        ckpt_mapping = {
            "large": (os.path.join(self.sam_path, "sam2_hiera_large.pt"), "sam2_hiera_l.yaml"),
            "base":  (os.path.join(self.sam_path, "sam2_hiera_base_plus.pt"), "sam2_hiera_b+.yaml"),
            "small": (os.path.join(self.sam_path, "sam2_hiera_small.pt"), "sam2_hiera_s.yaml"),
            "tiny":  (os.path.join(self.sam_path, "sam2_hiera_tiny.pt"), "sam2_hiera_t.yaml"),
        }
        sam_ckpt, model_cfg = ckpt_mapping[self.sam_choice]
        if hasattr(self, "predictor"):
            del self.predictor
        self.predictor = build_sam2_predictor(model_cfg, sam_ckpt, device=DEVICE)
        print(f"SAM-2 loaded from {sam_ckpt}")
        self.able_to_track = False

    def load_rocket(self, ckpt_path: str, cfg_coef: float = 1.0):
        if ckpt_path.startswith("hf:"):
            ckpt_path = ckpt_path.split(":")[-1]
            agent = CrossViewRocket.from_pretrained(ckpt_path).to(DEVICE)
        else:
            assert os.path.exists(ckpt_path), f"Model path {ckpt_path} not found."
            agent = load_cross_view_rocket(ckpt_path).to(DEVICE)
        agent.eval()
        self.agent = CFGWrapper(agent, k=cfg_coef)
        self.clear_memory()

    def clear_points(self):
        self.points = []
        self.points_label = []

    def clear_memory(self):
        self.num_steps = 0
        if hasattr(self, "agent"):
            self.state = self.agent.initial_state()

    def configure_env(self, yaml_str: str):
        try:
            self.env_conf = yaml.safe_load(yaml_str)
            return True, "Environment configured."
        except Exception as e:
            return False, f"Config error: {e}"

    def reset(self):
        if self.env_conf is None:
            return None, "Configure environment first."
        self.obs_history = []
        self.image_history = []
        callbacks = load_callbacks_from_config(self.env_conf)
        callbacks.append(PrevActionCallback())
        self.env = MinecraftSim(callbacks=callbacks)
        self.obs, self.info = self.env.reset()
        for _ in range(30):
            time.sleep(0.1)
            noop = self.env.noop_action()
            self.obs, self.reward, _, _, self.info = self.env.step(noop)
        self.reward = 0
        self.current_image = self.info["pov"]
        self.obs_history.append(self.current_image)
        self.image_history.append(self.current_image)
        return self.current_image, "Environment reset."

    def segment(self):
        if len(self.points) == 0:
            return self.cross_view_image
        self.predictor.load_first_frame(self.cross_view_image)
        _, _, out_mask_logits = self.predictor.add_new_prompt(
            frame_idx=0, obj_id=0,
            points=self.points, labels=self.points_label,
        )
        self.obj_mask = (out_mask_logits[0, 0] > 0.0).cpu().numpy()
        self.clear_points()
        return self.apply_mask()

    def apply_mask(self):
        image = self.cross_view_image.copy()
        color = np.array(COLORS[INTERACTION_TYPES[self.segment_type]]).reshape(1, 1, 3)[:, :, ::-1]
        mask_overlay = (self.obj_mask[..., None] * color).astype(np.uint8)
        return cv2.addWeighted(image, 1.0, mask_overlay, 0.5, 0.0)

    def _build_cross_view_overlay(self):
        """Build cross-view image with segmentation mask and contour for thumbnail."""
        cv_copy = self.cross_view_image.copy()
        color = np.array(COLORS[INTERACTION_TYPES[self.segment_type]]).reshape(1, 1, 3)[:, :, ::-1]
        mask_overlay = (self.obj_mask[..., None] * color).astype(np.uint8)
        cv_with_mask = cv2.addWeighted(cv_copy, 1.0, mask_overlay, 0.5, 0.0)
        binary_mask = (self.obj_mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(cv_with_mask, contours, -1, (255, 0, 0), 2)
        return cv_with_mask

    def step(self, input_action=None):
        pred_point = None
        pred_bbox = None
        pred_exist = None

        if input_action is not None:
            action = input_action
        else:
            obj_id = torch.tensor(INTERACTION_TYPES[self.segment_type])
            obj_mask = cv2.resize(
                self.obj_mask.astype(np.uint8), (224, 224), interpolation=cv2.INTER_LINEAR
            )
            obj_mask = torch.tensor(obj_mask, dtype=torch.uint8)
            cross_view = cv2.resize(
                self.cross_view_image, (224, 224), interpolation=cv2.INTER_LINEAR
            )
            obs = {
                "image": self.obs["image"],
                "env_prev_action": self.obs["env_prev_action"],
                "cross_view": {
                    "cross_view_image": cross_view,
                    "cross_view_obj_id": obj_id,
                    "cross_view_obj_mask": obj_mask,
                },
            }
            action, self.state = self.agent.get_action(obs, self.state, input_shape="*")
            if "bbox" in self.agent.cache_latents:
                pred_bbox = self.agent.cache_latents["bbox"].cpu().numpy()
            pred_point = self.agent.cache_latents["point"].cpu().numpy()
            pred_exist = self.agent.cache_latents["exist"].sigmoid().cpu().item()

        self.obs, self.reward, _, _, self.info = self.env.step(action)
        self.current_image = self.info["pov"]
        image = self.current_image.copy()

        if self.able_to_track and self.tracking_flag:
            self.segment()
            image = self.apply_mask()
        else:
            time.sleep(0.001)

        if "buttons" in action:
            image = draw_action(image, self.env.agent_action_to_env_action(action))
        else:
            image = draw_action(image, action)

        if pred_point is not None:
            point = (pred_point * np.array([image.shape[0], image.shape[1]])).astype(np.int32)
            cy, cx = point
            cv2.line(image, (cx - 10, cy), (cx + 10, cy), (255, 255, 255), 1)
            cv2.line(image, (cx, cy - 10), (cx, cy + 10), (255, 255, 255), 1)

        if pred_exist is not None:
            cv2.rectangle(image, (10, 40), (260, 50), (100, 100, 100), -1)
            cv2.rectangle(image, (10, 40), (10 + int(250 * pred_exist), 50), (255, 255, 255), -1)

        if pred_bbox is not None:
            s = np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
            x1, y1, x2, y2 = (pred_bbox * s).astype(np.int32)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2)

        self.num_steps += 1
        cv_with_mask = self._build_cross_view_overlay()
        save_image = _overlay_thumbnail(bg_image=image, fg_image=cv_with_mask)

        self.obs_history.append(self.current_image)
        self.image_history.append(save_image)

        return save_image, {
            "visibility": pred_exist,
            "cross_view_image_with_mask": cv_with_mask,
        }

    def stop(self):
        self.running = False


# ---------------------------------------------------------------------------
# Gradio Web Application
# ---------------------------------------------------------------------------

def build_eval_app(args):
    """Build and launch the interactive evaluator Gradio application."""

    global _evaluator
    _evaluator = InteractiveEvaluator(sam_path=args.sam_path)
    ev = _evaluator

    with open("theme.json", "r") as f:
        theme = gr.Theme.from_dict(json.load(f))

    blank = np.zeros((360, 640, 3), dtype=np.uint8)

    gallery_dir = Path(args.gallery_dir)
    gallery_dir.mkdir(parents=True, exist_ok=True)
    gallery_images = sorted(gallery_dir.glob("*.png")) + sorted(gallery_dir.glob("*.jpg"))

    env_files = sorted(Path(args.env_conf).glob("*.yaml")) if args.env_conf else []
    env_choices = [str(x) for x in env_files]
    default_env = next((x for x in env_choices if "wood" in x), env_choices[0] if env_choices else "")

    custom_css = """
    #eval-header { text-align: center; padding: 16px 0 8px 0; }
    #eval-header h1 { margin-bottom: 4px; }
    #eval-header p { color: #666; font-size: 14px; margin: 0; }
    #main-display img { border-radius: 8px; }
    #goal-display img { border-radius: 8px; }
    """

    with gr.Blocks(theme=theme, css=custom_css, title="ROCKET-2 Interactive Evaluator") as app:

        gr.HTML(
            '<div id="eval-header">'
            '<h1>ROCKET-2 Interactive Evaluator</h1>'
            '<p>Configure environment &rarr; Reset &rarr; '
            'Set goal via SAM-2 &rarr; Execute steps &rarr; Observe results</p>'
            '</div>'
        )

        with gr.Row():

            # ========================= Left: Rendering + Controls ========================= #
            with gr.Column(scale=3):

                status_bar = gr.Textbox(
                    show_label=False,
                    placeholder="Ready. Configure environment and click Reset to begin.",
                    interactive=False,
                )

                main_display = gr.Image(
                    value=blank,
                    interactive=False,
                    show_label=False,
                    show_download_button=False,
                    show_fullscreen_button=False,
                    streaming=True,
                    height=int(PAGE_WIDTH * SCALE),
                    width=PAGE_WIDTH,
                    elem_id="main-display",
                )

                with gr.Row(equal_height=True):
                    reset_btn = gr.Button("Reset", scale=1)
                    clr_mem_btn = gr.Button("Clear Memory", scale=1)
                    steps_input = gr.Number(
                        value=30, minimum=1, maximum=1200,
                        show_label=False, interactive=True, scale=1,
                    )
                    go_btn = gr.Button("Go", variant="primary", interactive=False, scale=1)

                with gr.Accordion("Environment Configuration", open=False):
                    with gr.Row(equal_height=True):
                        conf_dropdown = gr.Dropdown(
                            value=default_env, choices=env_choices,
                            show_label=False, interactive=True, scale=4,
                        )
                        load_conf_btn = gr.Button("Load", scale=1)
                        set_conf_btn = gr.Button("Set", variant="primary", scale=1)
                    config_editor = gr.Code(
                        value=DEFAULT_ENV_CONFIG,
                        language="yaml", label="Environment Config", interactive=True,
                    )

                with gr.Accordion("Model Settings", open=False):
                    with gr.Row(equal_height=True):
                        ckpt_dropdown = gr.Dropdown(
                            label="Checkpoint",
                            choices=[
                                "hf:phython96/ROCKET-2-1.5x-17w",
                                "hf:phython96/ROCKET-2-1x-22w",
                            ],
                            show_label=True, interactive=True,
                            allow_custom_value=True, scale=4,
                        )
                        cfg_coef_input = gr.Number(
                            label="CFG Coefficient",
                            value=1.5, minimum=0.0, maximum=10.0, step=0.1,
                            show_label=True, interactive=True, scale=1,
                        )
                        set_model_btn = gr.Button("Set Model", scale=1)

                with gr.Accordion("Commands", open=False):
                    with gr.Row(equal_height=True):
                        command_input = gr.Dropdown(
                            choices=["/setblock ~0 ~0 ~4 minecraft:diamond_block"],
                            show_label=False, interactive=True,
                            allow_custom_value=True, scale=4,
                        )
                        send_cmd_btn = gr.Button("Send", scale=1)

                with gr.Accordion("Record Video", open=False):
                    record_video = gr.Video(show_label=False, interactive=False)
                    with gr.Row(equal_height=True):
                        make_video_btn = gr.Button("Make Video")
                        download_btn = gr.DownloadButton("Download", variant="primary", interactive=False)

            # ========================= Right: Goal Setting ========================= #
            with gr.Column(scale=2):

                gr.Markdown("### Set Evaluation Goal")

                with gr.Row(equal_height=True):
                    upload_btn = gr.UploadButton(
                        label="Upload Image", file_types=["image"], scale=1,
                    )
                    obs_slider = gr.Slider(
                        label="Select from History",
                        minimum=0, maximum=0, value=0, step=1,
                        interactive=False, scale=2,
                    )

                goal_display = gr.Image(
                    value=blank,
                    interactive=False,
                    show_label=False,
                    show_download_button=False,
                    show_fullscreen_button=False,
                    streaming=True,
                    height=280,
                    elem_id="goal-display",
                )

                with gr.Group():
                    with gr.Row(equal_height=True):
                        sam_radio = gr.Radio(
                            choices=["large", "base", "small", "tiny"],
                            value="base",
                            label="Select SAM2 Checkpoint",
                            interactive=True, scale=5,
                        )
                        point_radio = gr.Radio(
                            choices=["Pos", "Neg"], value="Pos",
                            label="Add Point Prompt",
                            interactive=True, scale=3,
                        )
                        clr_pts_btn = gr.Button("Clear Points", scale=1)

                with gr.Row(equal_height=True):
                    gr.Textbox(
                        value="Interaction Type", show_label=False,
                        interactive=False, scale=1,
                    )
                    seg_type_dropdown = gr.Dropdown(
                        choices=["Approach", "Interact", "Hunt", "Mine", "Craft", "None"],
                        value="Approach",
                        show_label=False, interactive=True, scale=4,
                    )
                    segment_btn = gr.Button("Segment", variant="primary", scale=1)

                visibility_label = gr.Label(
                    {"Visibility": 0.00}, show_label=False, show_heading=False,
                )

                gallery_widget = gr.Gallery(
                    gallery_images,
                    label="Select Cross-Episode Images in Gallery",
                    columns=4, rows=1, object_fit="contain", height=150,
                    show_share_button=False, show_download_button=False,
                    interactive=True, show_fullscreen_button=False,
                )

        # ========================= Event Handlers ========================= #
        # All handlers use the global `ev` singleton — no gr.State needed.

        def _on_page_load():
            """Restore UI state from the global evaluator when the page (re)loads."""
            has_env = ev.env is not None
            hist_max = max(len(ev.obs_history) - 1, 0)
            status_msg = ""
            if has_env:
                status_msg = f"Session active. Memory: {ev.num_steps} steps, History: {len(ev.obs_history)} frames."
            return (
                ev.current_image,
                ev.cross_view_image,
                status_msg,
                gr.update(interactive=has_env),
                gr.update(maximum=hist_max, interactive=has_env),
            )

        app.load(
            _on_page_load,
            inputs=[],
            outputs=[main_display, goal_display, status_bar, go_btn, obs_slider],
        )

        def _load_conf(conf_source):
            try:
                with open(conf_source) as f:
                    content = f.read()
                gr.Info(f"Config loaded: {Path(conf_source).name}")
                return content
            except Exception as e:
                gr.Error(f"Failed to load config: {e}")
                return gr.update()

        load_conf_btn.click(_load_conf, inputs=[conf_dropdown], outputs=[config_editor])

        def _set_conf(code):
            ok, msg = ev.configure_env(code)
            if ok:
                gr.Info(msg)
            else:
                gr.Error(msg)
            return msg

        set_conf_btn.click(_set_conf, inputs=[config_editor], outputs=[status_bar])

        def _reset(ckpt, coef):
            if ev.env_conf is None:
                gr.Warning("Please set environment config first.")
                return (
                    blank, blank,
                    "Please set environment config first.",
                    gr.update(interactive=False),
                    gr.update(),
                )
            try:
                ev.load_rocket(ckpt, coef)
                image, msg = ev.reset()
                if image is None:
                    gr.Error(msg)
                    return (
                        blank, blank, msg,
                        gr.update(interactive=False),
                        gr.update(),
                    )
                ev.cross_view_image = image
                gr.Info("Environment reset successfully.")
                return (
                    image, image, msg,
                    gr.update(interactive=True),
                    gr.update(value=0, maximum=len(ev.obs_history) - 1, interactive=True),
                )
            except Exception as e:
                gr.Error(f"Reset failed: {e}")
                return (
                    blank, blank, f"Reset failed: {e}",
                    gr.update(interactive=False),
                    gr.update(),
                )

        reset_btn.click(
            _reset,
            inputs=[ckpt_dropdown, cfg_coef_input],
            outputs=[main_display, goal_display, status_bar, go_btn, obs_slider],
        )

        def _set_model(ckpt, coef):
            try:
                ev.load_rocket(ckpt, coef)
                msg = f"Model loaded: {ckpt}, CFG={coef}"
                gr.Info(msg)
                return msg
            except Exception as e:
                gr.Error(f"Failed to load model: {e}")
                return f"Failed to load model: {e}"

        set_model_btn.click(
            _set_model,
            inputs=[ckpt_dropdown, cfg_coef_input],
            outputs=[status_bar],
        )

        def _upload_image(file):
            try:
                img = np.array(Image.open(file).convert("RGB"))
                img = cv2.resize(img, (640, 360), interpolation=cv2.INTER_LINEAR)
                ev.cross_view_image = img
                gr.Info("Cross-view image uploaded.")
                return img
            except Exception as e:
                gr.Error(f"Failed to upload image: {e}")
                return gr.update()

        upload_btn.upload(_upload_image, inputs=[upload_btn], outputs=[goal_display])

        def _select_from_history(idx):
            img = ev.obs_history[int(idx)]
            ev.cross_view_image = img
            return img

        obs_slider.release(
            _select_from_history, inputs=[obs_slider], outputs=[goal_display],
        )

        def _select_gallery(gallery_val, evt: gr.SelectData):
            img = np.array(Image.open(evt.value["image"]["path"]).convert("RGB"))
            img = cv2.resize(img, (640, 360), interpolation=cv2.INTER_LINEAR)
            ev.cross_view_image = img
            return img

        gallery_widget.select(
            _select_gallery, inputs=[gallery_widget], outputs=[goal_display],
        )

        def _select_sam(choice):
            try:
                ev.sam_choice = choice
                ev.load_sam()
                gr.Info(f"SAM-2 checkpoint switched to: {choice}")
            except Exception as e:
                gr.Error(f"Failed to load SAM-2: {e}")

        sam_radio.select(fn=_select_sam, inputs=[sam_radio], outputs=[], show_progress=True)

        def _draw_point(image, label, evt: gr.SelectData):
            x, y = evt.index[0], evt.index[1]
            x = np.clip(x, 0, image.shape[1] - 1)
            y = np.clip(y, 0, image.shape[0] - 1)
            ev.points.append([x, y])
            ev.points_label.append(1 if label == "Pos" else 0)
            out = np.copy(image)
            color = (0, 255, 0) if label == "Pos" else (255, 0, 0)
            cv2.circle(out, (x, y), 5, color, -1)
            return out

        goal_display.select(
            _draw_point, inputs=[goal_display, point_radio], outputs=[goal_display],
        )

        def _clear_points():
            ev.clear_points()
            gr.Info("Points cleared.")
            return ev.cross_view_image

        clr_pts_btn.click(_clear_points, inputs=[], outputs=[goal_display], show_progress=True)

        def _change_seg_type(seg_type):
            ev.segment_type = seg_type

        seg_type_dropdown.change(
            _change_seg_type, inputs=[seg_type_dropdown], outputs=[],
        )

        def _segment(seg_type):
            ev.segment_type = seg_type
            if len(ev.points) == 0:
                gr.Warning("No point prompts added. Click on the goal image first.")
                return ev.cross_view_image
            try:
                result = ev.segment()
                gr.Info("Segmentation complete.")
                return result
            except Exception as e:
                gr.Error(f"Segmentation failed: {e}")
                return ev.cross_view_image

        segment_btn.click(
            _segment, inputs=[seg_type_dropdown], outputs=[goal_display], show_progress=True,
        )

        def _clear_memory():
            ev.clear_memory()
            gr.Info("Agent memory cleared.")
            return "Agent memory cleared."

        clr_mem_btn.click(_clear_memory, inputs=[], outputs=[status_bar])

        def _go(steps):
            if not hasattr(ev, "agent"):
                gr.Error("Model not loaded. Please click Reset or Set Model first.")
                return
            if ev.env is None:
                gr.Error("Environment not started. Please click Reset first.")
                return

            ev.running = True
            total = int(steps)
            vis_val = 0.0
            frame = ev.current_image
            gr.Info(f"Rollout started: {total} steps.")
            for i in range(total):
                if not ev.running:
                    break
                try:
                    frame, info = ev.step()
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    gr.Error(f"Step failed at {i + 1}/{total}: {e}")
                    break
                vis_val = info["visibility"] if info["visibility"] is not None else 0.0
                yield (
                    frame,
                    f"Step {i + 1}/{total} | Memory: {ev.num_steps}",
                    {"Visibility": vis_val},
                    gr.update(maximum=len(ev.obs_history) - 1, interactive=True),
                    gr.update(),
                )
            ev.running = False
            ev.cross_view_image = ev.current_image.copy()
            ev.obj_mask = np.zeros(ev.cross_view_image.shape[:2], dtype=np.float32)
            ev.clear_points()
            gr.Info(f"Rollout finished. Total memory: {ev.num_steps} steps.")
            yield (
                frame,
                f"Finished {total} steps | Memory: {ev.num_steps}",
                {"Visibility": vis_val},
                gr.update(maximum=len(ev.obs_history) - 1, interactive=True),
                ev.cross_view_image,
            )

        go_btn.click(
            _go,
            inputs=[steps_input],
            outputs=[main_display, status_bar, visibility_label, obs_slider, goal_display],
            show_progress=False,
        )

        def _send_command(cmd):
            try:
                ev.env.env.execute_cmd(cmd)
                for _ in range(10):
                    time.sleep(0.1)
                    noop = ev.env.noop_action()
                    _, _, _, _, info = ev.env.step(noop)
                gr.Info(f"Command executed: {cmd}")
                return info["pov"], f"Command sent: {cmd}"
            except Exception as e:
                gr.Error(f"Command failed: {e}. Reset environment first.")
                return gr.update(), "Reset environment first."

        send_cmd_btn.click(
            _send_command, inputs=[command_input], outputs=[main_display, status_bar],
        )

        def _make_video(progress=gr.Progress()):
            images = ev.image_history
            if not images:
                gr.Warning("No frames recorded. Run some steps first.")
                return gr.update(interactive=False), gr.update(interactive=True), None
            try:
                filepath = "eval_recording.mp4"
                h, w = images[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(filepath, fourcc, 25.0, (w, h))
                for img in progress.tqdm(images):
                    writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                writer.release()
                ev.image_history = []
                gr.Info(f"Video created: {len(images)} frames.")
                return (
                    gr.update(interactive=False),
                    gr.update(value=filepath, interactive=True),
                    filepath,
                )
            except Exception as e:
                gr.Error(f"Failed to create video: {e}")
                return gr.update(), gr.update(), None

        make_video_btn.click(
            _make_video,
            inputs=[],
            outputs=[make_video_btn, download_btn, record_video],
            show_progress=True,
        )

        def _after_download():
            return gr.update(interactive=True), gr.update(interactive=False), None

        download_btn.click(
            _after_download, inputs=[], outputs=[make_video_btn, download_btn, record_video],
        )

        app.queue()
        app.launch(share=False, server_name="0.0.0.0", server_port=args.port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ROCKET-2 Interactive Evaluator")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--env-conf", type=str, default="./env_conf")
    parser.add_argument("--sam-path", type=str, required=True)
    parser.add_argument("--gallery-dir", type=str, default="./eval_gallery")
    args = parser.parse_args()
    build_eval_app(args)
