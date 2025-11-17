"""
ComfyUI-UniRig

UniRig integration for ComfyUI - State-of-the-art automatic rigging and skeleton extraction.

Based on: One Model to Rig Them All (SIGGRAPH 2025)
Repository: https://github.com/VAST-AI-Research/UniRig
"""

import os
import sys
import traceback

# Track initialization status
INIT_SUCCESS = False
INIT_ERRORS = []

# Set web directory for JavaScript extensions (FBX viewer widget)
WEB_DIRECTORY = "./web"

# Only run initialization and imports when loaded by ComfyUI, not during pytest
# This prevents relative import errors when pytest collects test modules
if 'pytest' not in sys.modules:
    print("[ComfyUI-UniRig] Initializing custom node...")

    # Import node classes
    try:
        from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
        print("[ComfyUI-UniRig] [OK] Node classes imported successfully")
        INIT_SUCCESS = True
    except Exception as e:
        error_msg = f"Failed to import node classes: {str(e)}"
        INIT_ERRORS.append(error_msg)
        print(f"[ComfyUI-UniRig] [WARNING] {error_msg}")
        print(f"[ComfyUI-UniRig] Traceback:\n{traceback.format_exc()}")

        # Set to empty if import failed
        NODE_CLASS_MAPPINGS = {}
        NODE_DISPLAY_NAME_MAPPINGS = {}

    # Add static route for Three.js and other libraries
    try:
        from server import PromptServer
        from aiohttp import web

        static_path = os.path.join(os.path.dirname(__file__), "static")
        if os.path.exists(static_path):
            PromptServer.instance.app.add_routes([
                web.static('/extensions/ComfyUI-UniRig/static', static_path)
            ])
    except Exception as e:
        print(f"[ComfyUI-UniRig] Warning: Could not add static route: {e}")

    # Report final status
    if INIT_SUCCESS:
        print("[ComfyUI-UniRig] [OK] Loaded successfully!")
    else:
        print(f"[ComfyUI-UniRig] [ERROR] Failed to load ({len(INIT_ERRORS)} error(s)):")
        for error in INIT_ERRORS:
            print(f"  - {error}")
        print("[ComfyUI-UniRig] Please check the errors above and your installation.")

else:
    # During testing, set dummy values to prevent import errors
    print("[ComfyUI-UniRig] Running in pytest mode - skipping initialization")
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
__version__ = "1.0.0"
