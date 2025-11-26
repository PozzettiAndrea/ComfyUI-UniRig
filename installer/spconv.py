"""
spconv installation for ComfyUI-UniRig.
"""

import subprocess
import sys

from .utils import InstallResult, InstallLogger, check_package_installed
from .config import CUDA_TO_SPCONV, PIP_TIMEOUT


def install_spconv(cuda_suffix: str) -> InstallResult:
    """
    Install spconv if CUDA is available.

    Args:
        cuda_suffix: CUDA suffix (e.g., "cu121", "cpu")

    Returns:
        InstallResult with success status
    """
    if cuda_suffix == 'cpu':
        InstallLogger.info("Skipping spconv (CPU-only environment)")
        return InstallResult(success=True, method="skipped", optional=True)

    # Check if already installed
    if check_package_installed("spconv"):
        InstallLogger.info("spconv already installed")
        return InstallResult(success=True, method="already_installed")

    InstallLogger.info(f"Installing spconv for {cuda_suffix}...")

    # Get list of versions to try
    versions_to_try = CUDA_TO_SPCONV.get(cuda_suffix, [cuda_suffix])

    for spconv_cuda in versions_to_try:
        spconv_package = f"spconv-{spconv_cuda}"
        InstallLogger.info(f"Trying {spconv_package}...")

        cmd = [
            sys.executable, "-m", "pip", "install",
            spconv_package
        ]

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=PIP_TIMEOUT
            )
            InstallLogger.success(f"{spconv_package} installed")
            return InstallResult(success=True, method="pip")

        except subprocess.CalledProcessError:
            continue
        except subprocess.TimeoutExpired:
            continue

    # All versions failed
    InstallLogger.warning("Failed to install spconv (optional)")
    InstallLogger.info("For manual installation, see https://github.com/traveller59/spconv")

    # Return success=True because spconv is optional
    return InstallResult(
        success=True,
        method="skipped",
        optional=True,
        error="No compatible spconv version found"
    )
