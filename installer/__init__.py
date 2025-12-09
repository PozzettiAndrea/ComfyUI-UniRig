"""
ComfyUI-UniRig Installer Package

Modular installation system for UniRig dependencies with:
- Source build fallback for torch-scatter/torch-cluster
- Proper CUDA environment setup
- Clear step-by-step logging
- Graceful degradation for optional dependencies
"""

import sys

from .config import SUPPORTED_PYTHON
from .utils import InstallResult, InstallLogger
from .detector import DependencyDetector
from .requirements import install_requirements
from .torch_geometric import install_torch_geometric, TorchGeometricInstaller
from .spconv import install_spconv
from .flash_attn import install_flash_attn
from .blender import install_blender, find_blender_executable


def install(skip_optional: bool = False, verbose: bool = True) -> dict:
    """
    Main installation entry point.

    Args:
        skip_optional: If True, skip optional dependencies (currently unused, flash-attn is required)
        verbose: If True, print detailed output

    Returns:
        dict with installation results for each component

    Environment Variables:
        UNIRIG_SKIP_INSTALL: Set to '1' to skip all dependency installation (for Docker)
        UNIRIG_SKIP_BLENDER_INSTALL: Set to '1' to skip Blender download (assumes Blender in PATH)
    """
    import os

    # Check if installation is disabled via environment variable
    if os.environ.get('UNIRIG_SKIP_INSTALL', '').lower() in ('1', 'true', 'yes'):
        InstallLogger.info("UNIRIG_SKIP_INSTALL is set - skipping dependency installation")
        InstallLogger.info("Ensure all dependencies are pre-installed in your environment")
        return {"success": True, "skipped": True}

    results = {}
    total_steps = 5

    InstallLogger.header("ComfyUI-UniRig: Installing dependencies")

    # Step 1: Check Python/PyTorch versions
    InstallLogger.step(1, total_steps, "Checking environment...")

    deps = DependencyDetector.check_all_dependencies()

    if not deps["torch"]["installed"]:
        InstallLogger.error("PyTorch not found. Please install PyTorch first.")
        return {"success": False, "error": "PyTorch not installed"}

    torch_info = deps["torch"]
    InstallLogger.info(f"PyTorch {torch_info['version']} with {torch_info['cuda_suffix']}")
    InstallLogger.info(f"Python {sys.version_info.major}.{sys.version_info.minor}")

    # Step 2: Install requirements.txt
    InstallLogger.step(2, total_steps, "Installing basic requirements...")
    results["requirements"] = install_requirements()

    if not results["requirements"].success:
        InstallLogger.error("Failed to install basic requirements")
        InstallLogger.info("You may need to install manually:")
        InstallLogger.info("  pip install -r requirements.txt")
        return {"success": False, "error": "Requirements failed", "results": results}

    # Step 3: Install torch-geometric deps (with source fallback)
    InstallLogger.step(3, total_steps, "Installing torch-scatter and torch-cluster...")
    results["torch_geometric"] = install_torch_geometric(
        torch_info["version"],
        torch_info["cuda_suffix"]
    )

    # Check if critical deps failed
    tg_results = results["torch_geometric"]
    for pkg, result in tg_results.items():
        if not result.success:
            InstallLogger.error(f"Failed to install {pkg}")
            InstallLogger.info("You may need to install manually:")
            InstallLogger.info(f"  pip install {pkg}")

    # Step 4: Install spconv (required for inference, but may not have wheels available)
    InstallLogger.step(4, total_steps, "Installing spconv (required for inference)...")
    results["spconv"] = install_spconv(torch_info["cuda_suffix"])

    if not results["spconv"].success:
        InstallLogger.warning("spconv installation failed - UniRig skeleton extraction will NOT work")
        InstallLogger.warning("Please install spconv manually before using UniRig nodes")

    # Step 5: Install flash-attn (required for model compatibility)
    InstallLogger.step(5, total_steps, "Installing flash-attn (required)...")
    results["flash_attn"] = install_flash_attn()

    if not results["flash_attn"].success and not results["flash_attn"].optional:
        InstallLogger.warning("flash-attn installation failed - UniRig may not work correctly")

    # Summary
    _print_summary(results)

    return {"success": True, "results": results}


def _print_summary(results: dict):
    """Print installation summary."""
    print("\n" + "=" * 60)
    print("[UniRig Install] Installation Summary")
    print("=" * 60)

    for component, result in results.items():
        if isinstance(result, dict):
            # Multi-package result (torch_geometric)
            for pkg, pkg_result in result.items():
                status = "[OK]" if pkg_result.success else "[FAIL]"
                method = f" ({pkg_result.method})" if pkg_result.method else ""
                print(f"  {pkg}: {status}{method}")
        else:
            status = "[OK]" if result.success else "[FAIL]"
            if result.optional and not result.success:
                status = "[SKIP]"
            method = f" ({result.method})" if result.method else ""
            print(f"  {component}: {status}{method}")

    print("=" * 60)


# Public exports
__all__ = [
    "install",
    "install_requirements",
    "install_torch_geometric",
    "install_spconv",
    "install_flash_attn",
    "install_blender",
    "find_blender_executable",
    "DependencyDetector",
    "InstallResult",
    "InstallLogger",
    "TorchGeometricInstaller",
]
