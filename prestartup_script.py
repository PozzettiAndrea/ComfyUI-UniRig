"""ComfyUI-UniRig Prestartup Script."""

from pathlib import Path
from comfy_env import setup_env, copy_files
from comfy_3d_viewers import copy_viewer

setup_env()

SCRIPT_DIR = Path(__file__).parent
COMFYUI_DIR = SCRIPT_DIR.parent.parent

# Copy viewers
copy_viewer("fbx", SCRIPT_DIR / "web")
copy_viewer("fbx_debug", SCRIPT_DIR / "web")
copy_viewer("fbx_compare", SCRIPT_DIR / "web")

# Copy assets
copy_files(SCRIPT_DIR / "assets", COMFYUI_DIR / "input/3d")
copy_files(SCRIPT_DIR / "assets/animation_templates", COMFYUI_DIR / "input/animation_templates", "**/*")
copy_files(SCRIPT_DIR / "assets/animation_characters", COMFYUI_DIR / "input/animation_characters")
