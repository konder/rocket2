"""
ROCKET-2 agent wrapper for benchmark evaluation.

Wraps CrossViewRocket + CFGWrapper into a self-contained agent that:
  - Loads the model from HuggingFace or local checkpoint.
  - Manages recurrent memory state.
  - Accepts a cross-view goal (image + mask + interaction_type) once per episode.
  - Produces actions from observations via classifier-free guidance.
"""

import os
import sys
import cv2
import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model import CrossViewRocket, load_cross_view_rocket
from cfg_wrapper import CFGWrapper

INTERACTION_TYPES = {
    "hunt": 0,
    "use": 3,
    "mine": 2,
    "interact": 3,
    "craft": 4,
    "switch": 5,
    "approach": 6,
    "none": -1,
}


def get_device() -> str:
    env_device = os.environ.get("ROCKET_DEVICE")
    if env_device:
        return env_device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Rocket2Agent:
    """
    Stateful ROCKET-2 agent for benchmark evaluation.

    Usage:
        agent = Rocket2Agent(ckpt_path="hf:phython96/ROCKET-2-1.5x-17w")
        agent.set_goal(cross_view_image, obj_mask, interaction_type="hunt")
        for step in range(max_steps):
            action = agent.act(obs)
            obs, reward, done, truncated, info = env.step(action)
    """

    def __init__(
        self,
        ckpt_path: str = "hf:phython96/ROCKET-2-1.5x-17w",
        cfg_coef: float = 1.5,
        device: Optional[str] = None,
    ):
        self.device = device or get_device()
        self.cfg_coef = cfg_coef
        self._load_model(ckpt_path)
        self.reset()

    def _load_model(self, ckpt_path: str):
        if ckpt_path.startswith("hf:"):
            repo_id = ckpt_path.split(":", 1)[1]
            model = CrossViewRocket.from_pretrained(repo_id).to(self.device)
        else:
            model = load_cross_view_rocket(ckpt_path).to(self.device)
        model.eval()
        self.model = CFGWrapper(model, k=self.cfg_coef)

    def reset(self):
        """Reset memory state for a new episode."""
        self.state = self.model.initial_state()
        self.cross_view_image = np.zeros((224, 224, 3), dtype=np.uint8)
        self.obj_mask = np.zeros((224, 224), dtype=np.uint8)
        self.obj_id = torch.tensor(-1)

    def set_goal(
        self,
        cross_view_image: np.ndarray,
        obj_mask: np.ndarray,
        interaction_type: str = "none",
    ):
        """
        Set the cross-view goal for the current episode.

        Args:
            cross_view_image: (H, W, 3) RGB image containing the target object.
            obj_mask: (H, W) binary mask highlighting the target object.
            interaction_type: one of INTERACTION_TYPES keys.
        """
        self.cross_view_image = cv2.resize(
            cross_view_image, (224, 224), interpolation=cv2.INTER_LINEAR
        )
        mask_resized = cv2.resize(
            obj_mask.astype(np.uint8), (224, 224), interpolation=cv2.INTER_LINEAR
        )
        self.obj_mask = torch.tensor(mask_resized, dtype=torch.uint8)
        itype = interaction_type.lower()
        self.obj_id = torch.tensor(INTERACTION_TYPES.get(itype, -1))

    def act(self, obs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Produce an action from the current observation.

        Args:
            obs: dict with at least "image" and "env_prev_action" keys,
                 as returned by MinecraftSim.

        Returns:
            action: dict with "buttons" and "camera" keys.
        """
        enriched_obs = {
            "image": obs["image"],
            "env_prev_action": obs["env_prev_action"],
            "cross_view": {
                "cross_view_image": self.cross_view_image,
                "cross_view_obj_id": self.obj_id,
                "cross_view_obj_mask": self.obj_mask,
            },
        }
        action, self.state = self.model.get_action(
            enriched_obs, self.state, input_shape="*"
        )
        return action
