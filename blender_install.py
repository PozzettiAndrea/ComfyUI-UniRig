#!/usr/bin/env python3
"""
Manual Blender installation script for ComfyUI-UniRig.

Run this script to download and install Blender 4.2.3 to lib/blender/

Usage:
    python blender_install.py
"""
import sys
import os

# Add repo to path for installer imports
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

from pathlib import Path


def main():
    # Import here after path setup
    from installer.blender import install_blender

    target_dir = Path(REPO_ROOT) / "lib" / "blender"

    print("=" * 50)
    print("ComfyUI-UniRig Blender Installer")
    print("=" * 50)
    print(f"Target: {target_dir}")
    print()

    result = install_blender(target_dir=str(target_dir))

    if result:
        print()
        print("=" * 50)
        print("SUCCESS! Blender installed to:")
        print(f"  {result}")
        print()
        print("Restart ComfyUI to use the new Blender installation.")
        print("=" * 50)
    else:
        print()
        print("=" * 50)
        print("FAILED! Could not install Blender.")
        print("Please install Blender manually from:")
        print("  https://www.blender.org/download/")
        print("=" * 50)
        sys.exit(1)


if __name__ == "__main__":
    main()
