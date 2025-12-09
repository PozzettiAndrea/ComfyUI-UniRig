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

    # All versions failed - this is important to communicate clearly
    InstallLogger.error("=" * 60)
    InstallLogger.error("spconv installation FAILED")
    InstallLogger.error("=" * 60)
    InstallLogger.error("spconv is REQUIRED for UniRig inference.")
    InstallLogger.error("Without spconv, UniRig skeleton extraction will NOT work.")
    InstallLogger.error("")
    InstallLogger.error("To install manually:")
    InstallLogger.error(f"  pip install spconv-{cuda_suffix}")
    InstallLogger.error("")
    InstallLogger.error("If no wheel is available for your CUDA version, try:")
    InstallLogger.error("  pip install spconv-cu121  # CUDA 12.1")
    InstallLogger.error("  pip install spconv-cu120  # CUDA 12.0")
    InstallLogger.error("  pip install spconv-cu118  # CUDA 11.8")
    InstallLogger.error("")
    InstallLogger.error("See: https://github.com/traveller59/spconv")
    InstallLogger.error("=" * 60)

    # Return success=False because spconv is required for inference
    # We don't block installation, but mark it as failed/not optional
    return InstallResult(
        success=False,
        method="failed",
        optional=False,
        error="spconv is required for UniRig inference - install manually"
    )
