"""
Dependency detection for ComfyUI-UniRig installer.
"""

import sys
from typing import Optional


class DependencyDetector:
    """Detect installed packages and their versions."""

    @staticmethod
    def get_torch_info() -> dict:
        """
        Get PyTorch version and CUDA info.

        Returns:
            dict with keys: installed, version, cuda_available, cuda_version, cuda_suffix
        """
        try:
            import torch
            torch_version = torch.__version__.split('+')[0]  # e.g., "2.9.1"

            if torch.cuda.is_available():
                cuda_version = torch.version.cuda  # e.g., "12.1"
                if cuda_version:
                    cuda_suffix = 'cu' + cuda_version.replace('.', '')
                else:
                    cuda_suffix = 'cpu'
            else:
                cuda_version = None
                cuda_suffix = 'cpu'

            return {
                "installed": True,
                "version": torch_version,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": cuda_version,
                "cuda_suffix": cuda_suffix,
            }

        except ImportError:
            return {"installed": False}

    @staticmethod
    def check_package(package_name: str) -> dict:
        """
        Check if package is installed and get version.

        Args:
            package_name: Package name (will convert - to _ for import)

        Returns:
            dict with keys: installed, version (if available)
        """
        import_name = package_name.replace("-", "_")
        try:
            module = __import__(import_name)
            version = getattr(module, "__version__", "unknown")
            return {"installed": True, "version": version}
        except ImportError:
            return {"installed": False}

    @classmethod
    def check_all_dependencies(cls) -> dict:
        """
        Check all UniRig dependencies.

        Returns:
            dict with status of each dependency
        """
        return {
            "torch": cls.get_torch_info(),
            "torch_scatter": cls.check_package("torch_scatter"),
            "torch_cluster": cls.check_package("torch_cluster"),
            "spconv": cls.check_package("spconv"),
            "flash_attn": cls.check_package("flash_attn"),
            "python_box": cls.check_package("box"),
            "trimesh": cls.check_package("trimesh"),
        }

    @classmethod
    def get_missing_critical_dependencies(cls) -> list:
        """
        Get list of missing critical dependencies.

        Returns:
            List of (package_name, install_hint) tuples for missing deps
        """
        deps = cls.check_all_dependencies()
        missing = []

        if not deps["torch"]["installed"]:
            missing.append(("torch", "pip install torch"))

        if not deps["torch_scatter"]["installed"]:
            missing.append(("torch-scatter", "Run install.py or pip install torch-scatter"))

        if not deps["torch_cluster"]["installed"]:
            missing.append(("torch-cluster", "Run install.py or pip install torch-cluster"))

        if not deps["spconv"]["installed"]:
            missing.append(("spconv", "pip install spconv-cu121 (match your CUDA version)"))

        return missing

    @classmethod
    def print_dependency_status(cls):
        """Print a summary of all dependencies."""
        deps = cls.check_all_dependencies()

        print("\n[UniRig] Dependency Status:")
        print("-" * 40)

        for name, info in deps.items():
            if info.get("installed"):
                version = info.get("version", "?")
                print(f"  {name}: OK (v{version})")
            else:
                print(f"  {name}: MISSING")

        print("-" * 40)
