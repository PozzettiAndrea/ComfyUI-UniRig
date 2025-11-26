"""
Base setup and shared utilities for UniRig nodes.

Handles path configuration, Blender setup, and HuggingFace cache management.
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch
import base64
from io import BytesIO

import folder_paths

# Try to import PIL for texture handling
try:
    from PIL import Image as PILImage
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("[UniRig] Warning: PIL not available, texture preview will be limited")


# Get paths relative to this file
NODE_DIR = Path(__file__).parent.parent.absolute()  # Go up from nodes/ to ComfyUI-UniRig/
LIB_DIR = NODE_DIR / "lib"
UNIRIG_PATH = str(LIB_DIR / "unirig")
BLENDER_SCRIPT = str(LIB_DIR / "blender_extract.py")
BLENDER_PARSE_SKELETON = str(LIB_DIR / "blender_parse_skeleton.py")
BLENDER_EXTRACT_MESH_INFO = str(LIB_DIR / "blender_extract_mesh_info.py")

# Set up UniRig models directory in ComfyUI's models folder
# IMPORTANT: This must happen BEFORE any HuggingFace imports
UNIRIG_MODELS_DIR = Path(folder_paths.models_dir) / "unirig"
UNIRIG_MODELS_DIR.mkdir(parents=True, exist_ok=True)
(UNIRIG_MODELS_DIR / "hub").mkdir(parents=True, exist_ok=True)

# Set HuggingFace cache to use ComfyUI's models folder (never ~/.cache)
os.environ['HF_HOME'] = str(UNIRIG_MODELS_DIR)
os.environ['TRANSFORMERS_CACHE'] = str(UNIRIG_MODELS_DIR / "transformers")
os.environ['HF_HUB_CACHE'] = str(UNIRIG_MODELS_DIR / "hub")

print(f"[UniRig] Models cache location: {UNIRIG_MODELS_DIR}")

# Find Blender executable
BLENDER_DIR = LIB_DIR / "blender"
BLENDER_EXE = None
if BLENDER_DIR.exists():
    # Support both relative imports (ComfyUI) and absolute imports (testing)
    try:
        from ..install import find_blender_executable
    except ImportError:
        from install import find_blender_executable
    blender_bin = find_blender_executable(str(BLENDER_DIR))
    if blender_bin:
        BLENDER_EXE = str(blender_bin)
        os.environ['BLENDER_EXE'] = BLENDER_EXE
        print(f"[UniRig] Found Blender: {BLENDER_EXE}")

# Install Blender if not found (unless disabled via env var)
SKIP_BLENDER_INSTALL = os.environ.get('UNIRIG_SKIP_BLENDER_INSTALL', '0') == '1'

if not BLENDER_EXE and not SKIP_BLENDER_INSTALL:
    print("[UniRig] Blender not found, installing...")
    try:
        # Import from parent package
        sys.path.insert(0, str(NODE_DIR))
        try:
            from ..install import install_blender
        except ImportError:
            from install import install_blender
        BLENDER_EXE = install_blender(target_dir=BLENDER_DIR)
        if BLENDER_EXE:
            print(f"[UniRig] Blender installed: {BLENDER_EXE}")
        else:
            print("[UniRig] Warning: Blender installation failed")
    except Exception as e:
        print(f"[UniRig] Warning: Could not install Blender: {e}")
elif not BLENDER_EXE and SKIP_BLENDER_INSTALL:
    print("[UniRig] Blender not found, auto-install skipped (UNIRIG_SKIP_BLENDER_INSTALL=1)")

# Add local UniRig to path
if UNIRIG_PATH not in sys.path:
    sys.path.insert(0, UNIRIG_PATH)


def decode_texture_to_comfy_image(texture_data_base64: str):
    """
    Decode base64 texture to ComfyUI IMAGE format (torch tensor).

    Args:
        texture_data_base64: Base64-encoded image data

    Returns:
        tuple: (torch tensor [1, H, W, 3], width, height) or (None, 0, 0)
    """
    if not texture_data_base64 or not HAS_PIL:
        return None, 0, 0

    try:
        # Decode base64
        image_data = base64.b64decode(texture_data_base64)
        pil_image = PILImage.open(BytesIO(image_data))

        # Convert to RGB if necessary
        if pil_image.mode == 'RGBA':
            pil_image = pil_image.convert('RGB')
        elif pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        # Convert to numpy array
        img_array = np.array(pil_image).astype(np.float32) / 255.0

        # Convert to torch tensor [1, H, W, 3] for ComfyUI
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)

        return img_tensor, pil_image.width, pil_image.height

    except Exception as e:
        print(f"[UniRig] Error decoding texture: {e}")
        return None, 0, 0


def create_placeholder_texture(width: int = 256, height: int = 256, text: str = "No Texture"):
    """
    Create a placeholder image when no texture is available.

    Args:
        width: Image width
        height: Image height
        text: Text to display (not currently rendered, just for reference)

    Returns:
        torch.Tensor: Placeholder image tensor [1, H, W, 3]
    """
    try:
        # Create a gray image with text
        img_array = np.full((height, width, 3), 0.3, dtype=np.float32)

        # Add a simple pattern to indicate placeholder
        # Create a grid pattern
        for i in range(0, height, 32):
            img_array[i:i+2, :, :] = 0.4
        for j in range(0, width, 32):
            img_array[:, j:j+2, :] = 0.4

        img_tensor = torch.from_numpy(img_array).unsqueeze(0)
        return img_tensor

    except Exception as e:
        print(f"[UniRig] Error creating placeholder: {e}")
        # Return minimal gray image
        return torch.full((1, 64, 64, 3), 0.3)


def normalize_skeleton(vertices: np.ndarray) -> tuple:
    """
    Normalize skeleton vertices to [-1, 1] range.

    Args:
        vertices: Array of vertex positions

    Returns:
        tuple: (normalized_vertices, normalization_params)
            normalization_params contains 'center' and 'scale' for denormalization
    """
    min_coords = vertices.min(axis=0)
    max_coords = vertices.max(axis=0)
    center = (min_coords + max_coords) / 2
    vertices_centered = vertices - center
    scale = (max_coords - min_coords).max() / 2

    if scale > 0:
        vertices_normalized = vertices_centered / scale
    else:
        vertices_normalized = vertices_centered

    normalization_params = {
        'center': center,
        'scale': scale,
        'min_coords': min_coords,
        'max_coords': max_coords
    }

    return vertices_normalized, normalization_params


def setup_subprocess_env() -> dict:
    """
    Set up environment variables for UniRig subprocess calls.

    Returns:
        dict: Environment dictionary with Blender and HuggingFace paths configured
    """
    env = os.environ.copy()

    if BLENDER_EXE:
        env['BLENDER_EXE'] = BLENDER_EXE

    # Set PyOpenGL to use OSMesa for headless rendering (no EGL/X11 needed)
    env['PYOPENGL_PLATFORM'] = 'osmesa'

    # Ensure HuggingFace cache is set for subprocess
    if UNIRIG_MODELS_DIR:
        env['HF_HOME'] = str(UNIRIG_MODELS_DIR)
        env['TRANSFORMERS_CACHE'] = str(UNIRIG_MODELS_DIR / "transformers")
        env['HF_HUB_CACHE'] = str(UNIRIG_MODELS_DIR / "hub")

    return env
