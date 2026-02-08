"""ComfyUI-UniRig - Automatic rigging and skeleton extraction."""

import os
import sys
from pathlib import Path


print("[geompack] loading...", file=sys.stderr, flush=True)
from comfy_env import register_nodes
print("[geompack] calling register_nodes", file=sys.stderr, flush=True)
NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = register_nodes()

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
