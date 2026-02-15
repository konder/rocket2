"""
Utility functions for model weight loading with local cache and retry support.

Handles:
  - Loading timm pretrained weights from local files (avoids network downloads)
  - Retry logic for HuggingFace downloads (handles intermittent SSL failures)
  - Simulator engine discovery and extraction
"""

import os
import time

import timm
from safetensors.torch import load_file as load_safetensors

# Directory for pre-downloaded timm weights.
# Set TIMM_LOCAL_WEIGHTS env var to override.
TIMM_LOCAL_WEIGHTS = os.environ.get("TIMM_LOCAL_WEIGHTS", "/app/timm_weights")


def create_timm_model(model_name, max_retries=5, **kwargs):
    """Create a timm model, loading from local files if available.

    Priority:
      1. Local safetensors file at TIMM_LOCAL_WEIGHTS/{short_name}/model.safetensors
      2. Download from HuggingFace with retry logic (for SSL/network failures)

    Args:
        model_name: timm model name (e.g. 'timm/vit_base_patch16_224.dino')
        max_retries: max download retry attempts for network failures
        **kwargs: passed to timm.create_model (e.g. pretrained=True, features_only=True)
    """
    # --- Try local weights first ---
    if kwargs.get("pretrained", False):
        short_name = model_name.split("/")[-1]
        local_path = os.path.join(TIMM_LOCAL_WEIGHTS, short_name, "model.safetensors")
        if os.path.isfile(local_path):
            try:
                local_kwargs = dict(kwargs)
                local_kwargs["pretrained"] = False
                model = timm.create_model(model_name, **local_kwargs)
                state_dict = load_safetensors(local_path)
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                print(f"Loaded {model_name} from local: {local_path}")
                if missing:
                    print(f"  (missing keys: {len(missing)}, expected for features_only wrapper)")
                return model
            except Exception as e:
                print(f"WARNING: Failed to load local weights from {local_path}: {e}")
                print("  Falling back to network download...")

    # --- Fallback: download with retry ---
    for attempt in range(max_retries):
        try:
            return timm.create_model(model_name, **kwargs)
        except Exception as e:
            err_msg = str(e).lower()
            is_network_error = any(kw in err_msg for kw in [
                "ssl", "decryption", "connection", "timeout", "network", "read error"
            ])
            if is_network_error and attempt < max_retries - 1:
                wait = min(2 ** attempt, 30)
                print(f"[Retry {attempt+1}/{max_retries}] Failed to download {model_name}: {e}")
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise


def ensure_engine():
    """Ensure the Minecraft simulator engine is available.

    Searches common Docker paths for engine.zip and extracts it if needed.
    Sets MINESTUDIO_DIR env var if the engine is found in a non-default location.
    On a properly configured server, this is a no-op.
    """
    from minestudio.utils import get_mine_studio_dir

    studio_dir = get_mine_studio_dir()
    jar_path = os.path.join(studio_dir, "engine", "build", "libs", "mcprec-6.13.jar")
    if os.path.exists(jar_path):
        return  # engine already in place

    # Search common locations for engine.zip
    search_paths = [
        os.path.join(studio_dir, "engine.zip"),
        "/app/minestudio_temp_dir/engine.zip",
        "/app/minestudio_dir/engine.zip",
    ]
    for zip_path in search_paths:
        if os.path.exists(zip_path):
            import zipfile
            print(f"Extracting simulator engine from {zip_path} to {studio_dir}")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(studio_dir)
            if os.path.exists(jar_path):
                print("Simulator engine extracted successfully.")
                return

    # Check if engine was extracted elsewhere
    alt_dirs = ["/app/minestudio_temp_dir", "/app/minestudio_dir"]
    for alt in alt_dirs:
        alt_jar = os.path.join(alt, "engine", "build", "libs", "mcprec-6.13.jar")
        if os.path.exists(alt_jar) and alt != studio_dir:
            os.environ["MINESTUDIO_DIR"] = alt
            print(f"Found engine in {alt}, setting MINESTUDIO_DIR={alt}")
            return

    print("WARNING: Simulator engine not found. check_engine() will attempt to download it.")


def auto_detect_timm_local_weights():
    """Auto-detect local timm weights directory and set TIMM_LOCAL_WEIGHTS env var.

    Searches common paths used in Docker dev environments.
    """
    if os.environ.get("TIMM_LOCAL_WEIGHTS"):
        return
    for candidate in ["/app/timm_weights", "/app/ROCKET-2/timm_models"]:
        test_path = os.path.join(candidate, "vit_base_patch16_224.dino", "model.safetensors")
        if os.path.isfile(test_path):
            os.environ["TIMM_LOCAL_WEIGHTS"] = candidate
            print(f"TIMM_LOCAL_WEIGHTS set to {candidate}")
            return
