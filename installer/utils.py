"""
Utility functions for ComfyUI-UniRig installer.
"""

import subprocess
import sys
import os
import platform
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class InstallResult:
    """Result of an installation attempt."""
    success: bool
    method: Optional[str] = None  # "wheel", "source", "already_installed", etc.
    error: Optional[str] = None
    optional: bool = False


class InstallLogger:
    """Structured logging with consistent prefixes."""
    PREFIX = "[UniRig Install]"

    @classmethod
    def info(cls, msg: str):
        print(f"{cls.PREFIX} {msg}")

    @classmethod
    def success(cls, msg: str):
        print(f"{cls.PREFIX} [OK] {msg}")

    @classmethod
    def warning(cls, msg: str):
        print(f"{cls.PREFIX} [WARN] {msg}")

    @classmethod
    def error(cls, msg: str):
        print(f"{cls.PREFIX} [ERROR] {msg}")

    @classmethod
    def step(cls, step_num: int, total: int, msg: str):
        print(f"{cls.PREFIX} [{step_num}/{total}] {msg}")

    @classmethod
    def header(cls, title: str):
        print("=" * 60)
        print(f"{cls.PREFIX} {title}")
        print("=" * 60)


def run_pip_install(
    packages: List[str],
    extra_args: Optional[List[str]] = None,
    timeout: int = 300,
    env: Optional[dict] = None
) -> InstallResult:
    """
    Run pip install with proper error handling.

    Args:
        packages: List of package names or URLs to install
        extra_args: Additional pip arguments (e.g., ["-f", "url"])
        timeout: Timeout in seconds
        env: Optional environment variables

    Returns:
        InstallResult with success status and error message if failed
    """
    cmd = [sys.executable, "-m", "pip", "install"] + packages

    if extra_args:
        cmd.extend(extra_args)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env or os.environ.copy()
        )

        if result.returncode == 0:
            return InstallResult(success=True, method="pip")
        else:
            return InstallResult(success=False, error=result.stderr)

    except subprocess.TimeoutExpired:
        return InstallResult(success=False, error=f"Installation timed out after {timeout}s")
    except Exception as e:
        return InstallResult(success=False, error=str(e))


def get_platform_info() -> dict:
    """Get comprehensive platform information."""
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "darwin":
        plat = "macos"
        arch = "arm64" if machine == "arm64" else "x64"
    elif system == "linux":
        plat = "linux"
        arch = "x64"
    elif system == "windows":
        plat = "windows"
        arch = "x64"
    else:
        plat = None
        arch = None

    return {
        "system": system,
        "machine": machine,
        "platform": plat,
        "arch": arch,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "python_full": sys.version,
    }


def get_cuda_build_env() -> dict:
    """
    Get environment variables for building CUDA extensions.

    Sets up CUDA_HOME, PATH, and TORCH_CUDA_ARCH_LIST for source builds.
    """
    from .config import TORCH_CUDA_ARCH_LIST

    env = os.environ.copy()

    # Find CUDA home
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")

    if not cuda_home:
        # Try common locations
        for path in ["/usr/local/cuda", "/opt/cuda", "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA"]:
            if os.path.exists(path):
                cuda_home = path
                break

    if cuda_home and os.path.exists(cuda_home):
        env["CUDA_HOME"] = cuda_home
        env["CUDA_PATH"] = cuda_home

        # Add to PATH
        cuda_bin = os.path.join(cuda_home, "bin")
        if os.path.exists(cuda_bin):
            env["PATH"] = cuda_bin + os.pathsep + env.get("PATH", "")

        # Add to CPATH for headers
        cuda_include = os.path.join(cuda_home, "include")
        if os.path.exists(cuda_include):
            env["CPATH"] = cuda_include + os.pathsep + env.get("CPATH", "")

        # Add to LD_LIBRARY_PATH
        cuda_lib = os.path.join(cuda_home, "lib64")
        if os.path.exists(cuda_lib):
            env["LD_LIBRARY_PATH"] = cuda_lib + os.pathsep + env.get("LD_LIBRARY_PATH", "")

    # Set CUDA architectures for builds
    if "TORCH_CUDA_ARCH_LIST" not in env:
        env["TORCH_CUDA_ARCH_LIST"] = TORCH_CUDA_ARCH_LIST

    return env


def check_package_installed(package_name: str) -> bool:
    """Check if a package is installed."""
    try:
        # Handle package name vs import name differences
        import_name = package_name.replace("-", "_")
        __import__(import_name)
        return True
    except ImportError:
        return False
