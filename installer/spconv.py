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
    InstallLogger.warning("=" * 60)
    InstallLogger.warning("spconv installation FAILED")
    InstallLogger.warning("=" * 60)
    InstallLogger.warning("spconv is REQUIRED for GPU-accelerated skeleton extraction.")
    InstallLogger.warning("Without spconv, UniRig nodes will NOT work.")
    InstallLogger.warning("")
    InstallLogger.warning("To install manually:")
    InstallLogger.warning(f"  pip install spconv-{cuda_suffix}")
    InstallLogger.warning("")
    InstallLogger.warning("If no wheel is available for your CUDA version, try:")
    InstallLogger.warning("  pip install spconv-cu121  # CUDA 12.1")
    InstallLogger.warning("  pip install spconv-cu120  # CUDA 12.0")
    InstallLogger.warning("  pip install spconv-cu118  # CUDA 11.8")
    InstallLogger.warning("")
    InstallLogger.warning("See: https://github.com/traveller59/spconv")
    InstallLogger.warning("=" * 60)

    # Return success=True to not block installation, but with clear error
    return InstallResult(
        success=True,
        method="skipped",
        optional=True,
        error="spconv required for GPU inference - install manually"
    )
