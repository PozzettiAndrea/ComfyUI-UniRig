"""
Installation script for ComfyUI-UniRig dependencies.

This script is automatically run by ComfyUI Manager to install
dependencies that require special handling (torch-cluster, torch-scatter, spconv).
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install basic requirements from requirements.txt."""
    requirements_file = Path(__file__).parent / "requirements.txt"

    if not requirements_file.exists():
        print(f"[UniRig Install] Warning: requirements.txt not found at {requirements_file}")
        return True

    print(f"[UniRig Install] Installing basic dependencies from requirements.txt...")

    cmd = [
        sys.executable, "-m", "pip", "install",
        "-r", str(requirements_file)
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("[UniRig Install] ✓ Basic dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print("[UniRig Install] ✗ Failed to install basic dependencies")
        print(f"[UniRig Install] Error: {e.stderr}")
        return False

def get_torch_info():
    """Get PyTorch and CUDA versions."""
    try:
        import torch
        torch_version = torch.__version__.split('+')[0]  # e.g., "2.3.1"

        if torch.cuda.is_available():
            # Get CUDA version from torch
            cuda_version = torch.version.cuda  # e.g., "12.1"
            if cuda_version:
                # Convert to format like cu121
                cuda_suffix = 'cu' + cuda_version.replace('.', '')
            else:
                cuda_suffix = 'cpu'
        else:
            cuda_suffix = 'cpu'

        return torch_version, cuda_suffix
    except ImportError:
        print("[UniRig Install] ERROR: PyTorch not found. Please install PyTorch first.")
        sys.exit(1)

def install_torch_geometric_deps(torch_version, cuda_suffix):
    """Install torch-cluster and torch-scatter."""
    print(f"[UniRig Install] Detected PyTorch {torch_version} with {cuda_suffix}")

    # Check if already installed
    try:
        import torch_cluster
        import torch_scatter
        print("[UniRig Install] torch-cluster and torch-scatter already installed")
        return True
    except ImportError:
        pass

    # Construct the PyTorch Geometric wheel URL
    # Format: https://data.pyg.org/whl/torch-{version}+{cuda}/torch_cluster-{version}.html
    base_url = f"https://data.pyg.org/whl/torch-{torch_version}+{cuda_suffix}.html"

    print(f"[UniRig Install] Installing torch-cluster and torch-scatter from PyTorch Geometric...")
    print(f"[UniRig Install] Wheel URL: {base_url}")

    # Install both packages
    packages = ["torch-scatter", "torch-cluster"]

    for package in packages:
        print(f"[UniRig Install] Installing {package}...")
        cmd = [
            sys.executable, "-m", "pip", "install",
            package,
            "-f", base_url,
            "--no-cache-dir"
        ]

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"[UniRig Install] ✓ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"[UniRig Install] ✗ Failed to install {package}")
            print(f"[UniRig Install] Error: {e.stderr}")
            return False

    return True

def install_spconv(cuda_suffix):
    """Install spconv if CUDA is available."""
    if cuda_suffix == 'cpu':
        print("[UniRig Install] Skipping spconv (CPU-only environment)")
        return True

    # Check if already installed
    try:
        import spconv
        print("[UniRig Install] spconv already installed")
        return True
    except ImportError:
        pass

    print(f"[UniRig Install] Installing spconv for {cuda_suffix}...")

    # Try different spconv versions/names
    # spconv package naming can vary: spconv-cu120, spconv-cu118, etc.
    # Newer CUDA versions may need to fall back to older spconv versions

    # Map CUDA versions to compatible spconv versions
    cuda_to_spconv = {
        'cu128': ['cu121', 'cu120'],  # CUDA 12.8 -> try cu121 or cu120
        'cu127': ['cu121', 'cu120'],
        'cu126': ['cu121', 'cu120'],
        'cu125': ['cu121', 'cu120'],
        'cu124': ['cu121', 'cu120'],
        'cu123': ['cu121', 'cu120'],
        'cu122': ['cu121', 'cu120'],
        'cu121': ['cu121', 'cu120'],
        'cu120': ['cu120'],
        'cu118': ['cu118'],
        'cu117': ['cu117'],
    }

    # Get list of versions to try
    versions_to_try = cuda_to_spconv.get(cuda_suffix, [cuda_suffix])

    for spconv_cuda in versions_to_try:
        spconv_package = f"spconv-{spconv_cuda}"
        print(f"[UniRig Install] Trying {spconv_package}...")

        cmd = [
            sys.executable, "-m", "pip", "install",
            spconv_package
        ]

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"[UniRig Install] ✓ {spconv_package} installed successfully")
            return True
        except subprocess.CalledProcessError:
            continue

    print(f"[UniRig Install] ✗ Failed to install spconv for {cuda_suffix}")
    print(f"[UniRig Install] Note: spconv is optional. UniRig may work without it.")
    print(f"[UniRig Install] For manual installation, see https://github.com/traveller59/spconv")
    # Don't fail the entire installation if spconv fails - it might not be critical
    return True

def install_flash_attn():
    """Try to install flash-attn (optional, can be tricky)."""
    # Check if already installed
    try:
        import flash_attn
        print("[UniRig Install] flash-attn already installed")
        return True
    except ImportError:
        pass

    print("[UniRig Install] Attempting to install flash-attn (optional)...")
    print("[UniRig Install] Note: This may take a while and requires compilation...")

    cmd = [
        sys.executable, "-m", "pip", "install",
        "flash-attn", "--no-build-isolation"
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600)
        print("[UniRig Install] ✓ flash-attn installed successfully")
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print("[UniRig Install] ✗ flash-attn installation failed or timed out")
        print("[UniRig Install] Note: flash-attn is optional but recommended for performance.")
        print("[UniRig Install] For manual installation, see https://github.com/Dao-AILab/flash-attention")
        return True  # Don't fail - it's optional

def main():
    """Main installation routine."""
    print("=" * 60)
    print("ComfyUI-UniRig: Installing dependencies")
    print("=" * 60)

    # Install basic requirements first (trimesh, numpy, etc.)
    if not install_requirements():
        print("[UniRig Install] Failed to install basic requirements")
        print("[UniRig Install] You may need to install manually:")
        print("[UniRig Install]   pip install -r requirements.txt")
        sys.exit(1)

    # Get PyTorch info
    torch_version, cuda_suffix = get_torch_info()

    # Install torch-cluster and torch-scatter
    if not install_torch_geometric_deps(torch_version, cuda_suffix):
        print("[UniRig Install] Failed to install PyTorch Geometric dependencies")
        print("[UniRig Install] You may need to install manually:")
        print(f"[UniRig Install]   pip install torch-scatter torch-cluster -f https://data.pyg.org/whl/torch-{torch_version}+{cuda_suffix}.html")
        sys.exit(1)

    # Install spconv
    install_spconv(cuda_suffix)

    # Try to install flash-attn (optional)
    install_flash_attn()

    print("=" * 60)
    print("[UniRig Install] Installation complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
