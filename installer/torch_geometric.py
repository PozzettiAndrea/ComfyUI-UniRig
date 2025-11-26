"""
torch-scatter and torch-cluster installation with source build fallback.
"""

import subprocess
import sys
import os

from .utils import InstallResult, InstallLogger, get_cuda_build_env, check_package_installed
from .config import SOURCE_REPOS, PIP_TIMEOUT, SOURCE_BUILD_TIMEOUT


class TorchGeometricInstaller:
    """Install torch-scatter and torch-cluster with wheel/source fallback."""

    PACKAGES = ["torch-scatter", "torch-cluster"]

    def __init__(self, torch_version: str, cuda_suffix: str):
        """
        Initialize installer.

        Args:
            torch_version: PyTorch version (e.g., "2.9.1")
            cuda_suffix: CUDA suffix (e.g., "cu121", "cpu")
        """
        self.torch_version = torch_version
        self.cuda_suffix = cuda_suffix
        self.python_version = f"{sys.version_info.major}.{sys.version_info.minor}"

    def install(self) -> dict:
        """
        Install all packages, trying wheel first then source.

        Returns:
            dict mapping package name to InstallResult
        """
        results = {}

        InstallLogger.info(f"PyTorch {self.torch_version} with {self.cuda_suffix}")

        for package in self.PACKAGES:
            # Check if already installed
            if check_package_installed(package):
                InstallLogger.info(f"{package} already installed")
                results[package] = InstallResult(success=True, method="already_installed")
                continue

            # Try wheel installation
            InstallLogger.info(f"Installing {package}...")
            wheel_result = self._try_wheel_install(package)

            if wheel_result.success:
                InstallLogger.success(f"{package} installed from wheel")
                results[package] = wheel_result
                continue

            # Fall back to source build
            InstallLogger.warning(f"No wheel found for {package}, attempting source build...")
            InstallLogger.info("This may take 5-10 minutes...")

            source_result = self._try_source_build(package)

            if source_result.success:
                InstallLogger.success(f"{package} installed from source")
            else:
                InstallLogger.error(f"Failed to install {package}")
                InstallLogger.error(f"Error: {source_result.error}")

            results[package] = source_result

        return results

    def _try_wheel_install(self, package: str) -> InstallResult:
        """Try to install from PyTorch Geometric wheel index."""
        wheel_url = self._get_wheel_url()

        InstallLogger.info(f"Trying wheel from {wheel_url}")

        cmd = [
            sys.executable, "-m", "pip", "install",
            package,
            "-f", wheel_url,
            "--no-cache-dir"
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=PIP_TIMEOUT
            )

            if result.returncode == 0:
                return InstallResult(success=True, method="wheel")

            return InstallResult(success=False, error=result.stderr)

        except subprocess.TimeoutExpired:
            return InstallResult(success=False, error="Wheel installation timed out")
        except Exception as e:
            return InstallResult(success=False, error=str(e))

    def _try_source_build(self, package: str) -> InstallResult:
        """Build from source using pip install git+URL."""
        repo_url = SOURCE_REPOS.get(package)

        if not repo_url:
            return InstallResult(success=False, error=f"No source repo for {package}")

        # Set up build environment
        env = get_cuda_build_env()

        InstallLogger.info(f"Building {package} from source...")
        InstallLogger.info(f"Repository: {repo_url}")

        # Use pip install git+URL with --no-build-isolation
        # This allows using the existing torch installation
        cmd = [
            sys.executable, "-m", "pip", "install",
            f"git+{repo_url}",
            "--no-build-isolation"
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=SOURCE_BUILD_TIMEOUT,
                env=env
            )

            if result.returncode == 0:
                return InstallResult(success=True, method="source")

            # If --no-build-isolation fails, try without it
            InstallLogger.warning("Retrying without --no-build-isolation...")

            cmd_retry = [
                sys.executable, "-m", "pip", "install",
                f"git+{repo_url}"
            ]

            result_retry = subprocess.run(
                cmd_retry,
                capture_output=True,
                text=True,
                timeout=SOURCE_BUILD_TIMEOUT,
                env=env
            )

            if result_retry.returncode == 0:
                return InstallResult(success=True, method="source")

            return InstallResult(success=False, error=result_retry.stderr)

        except subprocess.TimeoutExpired:
            return InstallResult(
                success=False,
                error=f"Source build timed out after {SOURCE_BUILD_TIMEOUT}s"
            )
        except Exception as e:
            return InstallResult(success=False, error=str(e))

    def _get_wheel_url(self) -> str:
        """Construct PyTorch Geometric wheel index URL."""
        return f"https://data.pyg.org/whl/torch-{self.torch_version}+{self.cuda_suffix}.html"


def install_torch_geometric(torch_version: str, cuda_suffix: str) -> dict:
    """
    Convenience function to install torch-scatter and torch-cluster.

    Args:
        torch_version: PyTorch version
        cuda_suffix: CUDA suffix

    Returns:
        dict mapping package name to InstallResult
    """
    installer = TorchGeometricInstaller(torch_version, cuda_suffix)
    return installer.install()
