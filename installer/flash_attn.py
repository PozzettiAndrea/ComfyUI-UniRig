"""
flash-attn installation for ComfyUI-UniRig.
"""

import subprocess
import sys

from .utils import InstallResult, InstallLogger, check_package_installed
from .config import FLASH_ATTN_VERSION, PIP_TIMEOUT


def install_flash_attn() -> InstallResult:
    """
    Install flash-attn from official prebuilt wheels.

    Returns:
        InstallResult with success status
    """
    # Check if already installed
    if check_package_installed("flash_attn"):
        InstallLogger.info("flash-attn already installed")
        return InstallResult(success=True, method="already_installed")

    InstallLogger.info("Installing flash-attn from official prebuilt wheel...")

    # Get PyTorch and CUDA info
    try:
        import torch
        torch_version = torch.__version__.split('+')[0]
        torch_major_minor = '.'.join(torch_version.split('.')[:2])

        if not torch.cuda.is_available():
            InstallLogger.info("CUDA not available, skipping flash-attn (GPU-only)")
            return InstallResult(success=True, method="skipped", optional=True)

        cuda_version = torch.version.cuda
        cuda_major = cuda_version.split('.')[0] if cuda_version else None

        if not cuda_major:
            InstallLogger.info("Could not detect CUDA version, skipping flash-attn")
            return InstallResult(success=True, method="skipped", optional=True)

    except ImportError:
        InstallLogger.info("PyTorch not found, skipping flash-attn")
        return InstallResult(success=True, method="skipped", optional=True)

    # Construct the official wheel URL
    python_version = f"cp{sys.version_info.major}{sys.version_info.minor}"

    wheel_url = (
        f"https://github.com/Dao-AILab/flash-attention/releases/download/"
        f"v{FLASH_ATTN_VERSION}/flash_attn-{FLASH_ATTN_VERSION}%2Bcu{cuda_major}"
        f"torch{torch_major_minor}cxx11abiTRUE-{python_version}-{python_version}-linux_x86_64.whl"
    )

    InstallLogger.info(f"PyTorch {torch_version}, CUDA {cuda_version}, Python {python_version}")
    InstallLogger.info(f"Downloading from: {wheel_url}")

    cmd = [
        sys.executable, "-m", "pip", "install",
        wheel_url
    ]

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=PIP_TIMEOUT
        )
        InstallLogger.success("flash-attn installed")
        return InstallResult(success=True, method="wheel")

    except subprocess.CalledProcessError as e:
        InstallLogger.warning("flash-attn installation failed (optional)")
        InstallLogger.info("You may need to install manually from:")
        InstallLogger.info("  https://github.com/Dao-AILab/flash-attention/releases")
        return InstallResult(success=True, method="skipped", optional=True, error=e.stderr)

    except subprocess.TimeoutExpired:
        InstallLogger.warning("flash-attn installation timed out (optional)")
        return InstallResult(success=True, method="skipped", optional=True, error="Timeout")
