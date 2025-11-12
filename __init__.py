"""
ComfyUI-UniRig

UniRig integration for ComfyUI - State-of-the-art automatic rigging and skeleton extraction.

Based on: One Model to Rig Them All (SIGGRAPH 2025)
Repository: https://github.com/VAST-AI-Research/UniRig
"""

from .unirig_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
__version__ = "1.0.0"
