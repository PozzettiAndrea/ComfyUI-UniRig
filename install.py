"""
Installation script for ComfyUI-UniRig dependencies.

This script is automatically run by ComfyUI Manager to install
dependencies that require special handling (torch-cluster, torch-scatter, spconv).

The actual installation logic is in the 'installer' package for modularity.
This file is a thin wrapper for backwards compatibility.
"""

import sys
import os

# Ensure installer package is importable when this module is loaded from different contexts
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)


def main():
    """Main installation routine."""
    from installer import install

    # Run the main installation (Blender is installed separately via blender_install.py)
    result = install()

    if not result.get("success"):
        print("[UniRig Install] Installation failed!")
        sys.exit(1)

    print("[UniRig Install] Installation complete!")


# Legacy exports for backwards compatibility with nodes/base.py
def find_blender_executable(blender_dir):
    """Find the blender executable in the extracted directory."""
    from installer.blender import find_blender_executable as _find_blender_executable
    return _find_blender_executable(blender_dir)


def install_blender(target_dir=None):
    """Install Blender for mesh preprocessing."""
    from installer.blender import install_blender as _install_blender
    return _install_blender(target_dir)


def get_platform_info():
    """Detect current platform and architecture."""
    from installer.utils import get_platform_info as _get_platform_info
    info = _get_platform_info()
    return info["platform"], info["arch"]


def get_torch_info():
    """Get PyTorch and CUDA versions."""
    from installer.detector import DependencyDetector
    info = DependencyDetector.get_torch_info()
    if not info.get("installed"):
        print("[UniRig Install] ERROR: PyTorch not found. Please install PyTorch first.")
        sys.exit(1)
    return info["version"], info["cuda_suffix"]


if __name__ == "__main__":
    main()
