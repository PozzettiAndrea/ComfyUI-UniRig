"""
ComfyUI-UniRig

UniRig integration for ComfyUI - State-of-the-art automatic rigging and skeleton extraction.

Based on: One Model to Rig Them All (SIGGRAPH 2025)
Repository: https://github.com/VAST-AI-Research/UniRig
"""

from .unirig_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Set web directory for JavaScript extensions (FBX viewer widget)
# This tells ComfyUI where to find our JavaScript files and HTML viewer
# Files will be served at /extensions/ComfyUI-UniRig/*
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
__version__ = "1.0.0"
