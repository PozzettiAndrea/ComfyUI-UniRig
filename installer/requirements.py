"""
Requirements.txt installation for ComfyUI-UniRig.
"""

import subprocess
import sys
from pathlib import Path

from .utils import InstallResult, InstallLogger
from .config import PIP_TIMEOUT


def install_requirements(requirements_file: Path = None) -> InstallResult:
    """
    Install basic requirements from requirements.txt.

    Args:
        requirements_file: Path to requirements.txt. If None, uses default location.

    Returns:
        InstallResult with success status
    """
    if requirements_file is None:
        # Default to requirements.txt in parent directory
        requirements_file = Path(__file__).parent.parent / "requirements.txt"

    if not requirements_file.exists():
        InstallLogger.warning(f"requirements.txt not found at {requirements_file}")
        return InstallResult(success=True, method="skipped")

    InstallLogger.info("Installing basic dependencies from requirements.txt...")

    cmd = [
        sys.executable, "-m", "pip", "install",
        "-r", str(requirements_file)
    ]

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=PIP_TIMEOUT
        )
        InstallLogger.success("Basic dependencies installed")
        return InstallResult(success=True, method="pip")

    except subprocess.CalledProcessError as e:
        InstallLogger.error("Failed to install basic dependencies")
        InstallLogger.error(f"Error: {e.stderr}")
        return InstallResult(success=False, error=e.stderr)

    except subprocess.TimeoutExpired:
        InstallLogger.error(f"Installation timed out after {PIP_TIMEOUT}s")
        return InstallResult(success=False, error="Timeout")
