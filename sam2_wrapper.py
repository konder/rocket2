"""
Device-aware SAM2 wrapper.

Encapsulates SAM2 CameraPredictor to support CPU/MPS environments where:
  1. No NVIDIA GPU is available (hardcoded .cuda() calls would fail)
  2. The C extension (_C) for connected components is not compiled

Usage:
    from sam2_wrapper import build_sam2_predictor
    predictor = build_sam2_predictor(model_cfg, sam_ckpt, device="cpu")
"""

import cv2
import torch
import numpy as np

from sam2.build_sam import build_sam2_camera_predictor
from sam2.sam2_camera_predictor import SAM2CameraPredictor
import sam2.utils.misc as _sam2_misc

_patches_applied = False


def build_sam2_predictor(model_cfg, sam_ckpt, device="cuda"):
    """Build a SAM2 camera predictor with device-aware support.

    Applies compatibility patches once on first call, then delegates to
    the standard build_sam2_camera_predictor with the specified device.

    Args:
        model_cfg: SAM2 model config name (e.g. "sam2_hiera_b+.yaml")
        sam_ckpt: path to SAM2 checkpoint file
        device: target device ("cuda", "cpu", "mps")
    """
    _apply_patches()
    return build_sam2_camera_predictor(model_cfg, sam_ckpt, device=device)


def _apply_patches():
    """Apply SAM2 compatibility patches (idempotent, safe to call multiple times).

    Patches applied:
      1. get_connected_components: CPU fallback via cv2 when _C extension unavailable
      2. _init_state: use model device instead of hardcoded torch.device("cuda")
      3. _get_image_feature: use condition_state["device"] instead of .cuda()
      4. _get_feature: use condition_state["device"] instead of .cuda()
    """
    global _patches_applied
    if _patches_applied:
        return
    _patches_applied = True

    # --- Patch 1: get_connected_components CPU fallback ---
    _orig_gcc = _sam2_misc.get_connected_components

    def _get_connected_components(mask):
        try:
            from sam2 import _C
            return _C.get_connected_componnets(mask.to(torch.uint8).contiguous())
        except ImportError:
            return _connected_components_cpu(mask)

    _sam2_misc.get_connected_components = _get_connected_components

    # --- Patch 2: device-aware _init_state ---
    _orig_init_state = SAM2CameraPredictor._init_state

    def _init_state(self, *args, **kwargs):
        result = _orig_init_state(self, *args, **kwargs)
        _device = next(self.parameters()).device
        self.condition_state["device"] = _device
        if not self.condition_state.get("offload_state_to_cpu", False):
            self.condition_state["storage_device"] = _device
        return result

    SAM2CameraPredictor._init_state = _init_state

    # --- Patch 3: device-aware _get_image_feature ---
    def _get_image_feature(self, frame_idx, batch_size):
        image, backbone_out = self.condition_state["cached_features"].get(
            frame_idx, (None, None)
        )
        if backbone_out is None:
            device = self.condition_state["device"]
            image = (
                self.condition_state["images"][frame_idx]
                .to(device).float().unsqueeze(0)
            )
            backbone_out = self.forward_image(image)
            self.condition_state["cached_features"] = {
                frame_idx: (image, backbone_out)
            }
        expanded_image = image.expand(batch_size, -1, -1, -1)
        expanded_backbone_out = {
            "backbone_fpn": backbone_out["backbone_fpn"].copy(),
            "vision_pos_enc": backbone_out["vision_pos_enc"].copy(),
        }
        for i, feat in enumerate(expanded_backbone_out["backbone_fpn"]):
            expanded_backbone_out["backbone_fpn"][i] = feat.expand(
                batch_size, -1, -1, -1
            )
        for i, pos in enumerate(expanded_backbone_out["vision_pos_enc"]):
            expanded_backbone_out["vision_pos_enc"][i] = pos.expand(
                batch_size, -1, -1, -1
            )
        features = self._prepare_backbone_features(expanded_backbone_out)
        features = (expanded_image,) + features
        return features

    SAM2CameraPredictor._get_image_feature = _get_image_feature

    # --- Patch 4: device-aware _get_feature ---
    def _get_feature(self, img, batch_size):
        device = self.condition_state["device"]
        image = img.to(device).float().unsqueeze(0)
        backbone_out = self.forward_image(image)
        expanded_image = image.expand(batch_size, -1, -1, -1)
        expanded_backbone_out = {
            "backbone_fpn": backbone_out["backbone_fpn"].copy(),
            "vision_pos_enc": backbone_out["vision_pos_enc"].copy(),
        }
        for i, feat in enumerate(expanded_backbone_out["backbone_fpn"]):
            expanded_backbone_out["backbone_fpn"][i] = feat.expand(
                batch_size, -1, -1, -1
            )
        for i, pos in enumerate(expanded_backbone_out["vision_pos_enc"]):
            expanded_backbone_out["vision_pos_enc"][i] = pos.expand(
                batch_size, -1, -1, -1
            )
        features = self._prepare_backbone_features(expanded_backbone_out)
        features = (expanded_image,) + features
        return features

    SAM2CameraPredictor._get_feature = _get_feature


def _connected_components_cpu(mask):
    """Pure-Python connected components using cv2.connectedComponentsWithStats.

    Fallback for when the SAM2 C extension (_C) is not available.

    Args:
        mask: binary mask tensor of shape (N, 1, H, W)

    Returns:
        (labels, counts): tensors of shape (N, 1, H, W)
    """
    device = mask.device
    mask_np = mask.to(torch.uint8).cpu().numpy()
    N = mask_np.shape[0]
    labels_out = np.zeros_like(mask_np, dtype=np.int32)
    counts_out = np.zeros_like(mask_np, dtype=np.int32)

    for i in range(N):
        img = mask_np[i, 0]  # (H, W)
        num_labels, label_map, stats, _ = cv2.connectedComponentsWithStats(
            img, connectivity=8, ltype=cv2.CV_32S
        )
        areas = stats[:, cv2.CC_STAT_AREA]
        area_map = np.zeros_like(label_map, dtype=np.int32)
        for lbl in range(1, num_labels):
            area_map[label_map == lbl] = areas[lbl]
        labels_out[i, 0] = label_map
        counts_out[i, 0] = area_map

    return (
        torch.from_numpy(labels_out).to(device),
        torch.from_numpy(counts_out).to(device),
    )
