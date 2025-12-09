"""
flash-attn installation for ComfyUI-UniRig.

Multi-source wheel support:
- Linux: Dao-AILab official wheels, fallback to mjun0812 prebuilts
- Windows: bdashore3 prebuilt wheels
"""

import platform
import subprocess
import sys
from typing import List, Tuple, Optional

from .utils import InstallResult, InstallLogger, check_package_installed
from .config import FLASH_ATTN_VERSION, PIP_TIMEOUT


# Wheel sources configuration
# Format: (base_url, version, cuda_versions, torch_versions, python_versions, platform_tag, cxx11abi)

WHEEL_SOURCES = {
    "linux": [
        # Primary: Official Dao-AILab wheels (most comprehensive for Linux)
        {
            "name": "Dao-AILab (official)",
            "base_url": "https://github.com/Dao-AILab/flash-attention/releases/download",
            "version": "2.8.3",
            "cuda_versions": ["12"],  # Just major version
            "torch_versions": ["2.4", "2.5", "2.6", "2.7", "2.8"],
            "python_versions": ["cp39", "cp310", "cp311", "cp312", "cp313"],
            "platform": "linux_x86_64",
            "cxx11abi": "TRUE",
            "url_pattern": "{base_url}/v{version}/flash_attn-{version}%2Bcu{cuda}torch{torch}cxx11abi{abi}-{python}-{python}-{platform}.whl",
        },
        # Fallback: mjun0812 prebuilt wheels (more CUDA versions)
        {
            "name": "mjun0812",
            "base_url": "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download",
            "version": "2.8.3",
            "cuda_versions": ["124", "126", "128", "129", "130"],
            "torch_versions": ["2.5", "2.6", "2.7", "2.8", "2.9"],
            "python_versions": ["cp310", "cp311", "cp312"],
            "platform": "linux_x86_64",
            "cxx11abi": None,  # Not in filename
            "url_pattern": "{base_url}/v0.5.4/flash_attn-{version}%2Bcu{cuda}torch{torch}-{python}-{python}-{platform}.whl",
        },
    ],
    "windows": [
        # Primary: bdashore3 Windows builds
        {
            "name": "bdashore3",
            "base_url": "https://github.com/bdashore3/flash-attention/releases/download",
            "version": "2.8.3",
            "cuda_versions": ["124", "128"],
            "torch_versions": ["2.6.0", "2.7.0", "2.8.0"],
            "python_versions": ["cp310", "cp311", "cp312", "cp313"],
            "platform": "win_amd64",
            "cxx11abi": "FALSE",
            "url_pattern": "{base_url}/v{version}/flash_attn-{version}%2Bcu{cuda}torch{torch}cxx11abi{abi}-{python}-{python}-{platform}.whl",
        },
    ],
}

# CUDA version mapping - map detected version to available wheel versions
CUDA_VERSION_MAP = {
    # For sources that use major version only (Dao-AILab)
    "major": {
        "13": "12",
        "12": "12",
        "11": "12",  # Try 12 as fallback
    },
    # For sources that use major+minor (mjun0812, bdashore3)
    "full": {
        "130": ["130", "128", "126", "124"],
        "129": ["128", "126", "124"],  # 12.9 -> try 12.8
        "128": ["128", "126", "124"],
        "127": ["126", "124"],
        "126": ["126", "124"],
        "125": ["124"],
        "124": ["124"],
        "123": ["124"],
        "122": ["124"],
        "121": ["124"],
        "120": ["124"],
        "118": ["124"],  # Try 12.4 as fallback
    },
}


def _get_system_info() -> Optional[Tuple[str, str, str, str]]:
    """
    Get system information for wheel selection.

    Returns:
        Tuple of (os_name, torch_version, cuda_version, python_version) or None if unavailable
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None

        os_name = "windows" if platform.system() == "Windows" else "linux"
        torch_version = torch.__version__.split('+')[0]
        cuda_version = torch.version.cuda
        python_version = f"cp{sys.version_info.major}{sys.version_info.minor}"

        if not cuda_version:
            return None

        return (os_name, torch_version, cuda_version, python_version)

    except ImportError:
        return None


def _build_wheel_urls(os_name: str, torch_version: str, cuda_version: str, python_version: str) -> List[Tuple[str, str]]:
    """
    Build list of wheel URLs to try, in priority order.

    Returns:
        List of (source_name, url) tuples
    """
    urls = []
    sources = WHEEL_SOURCES.get(os_name, [])

    torch_major_minor = '.'.join(torch_version.split('.')[:2])
    cuda_major = cuda_version.split('.')[0]
    cuda_full = cuda_version.replace('.', '')[:3]  # e.g., "12.8" -> "128"

    for source in sources:
        # Check Python version compatibility
        if python_version not in source["python_versions"]:
            continue

        # Get CUDA versions to try
        if len(source["cuda_versions"][0]) <= 2:
            # Source uses major version only
            cuda_to_try = [CUDA_VERSION_MAP["major"].get(cuda_major, cuda_major)]
        else:
            # Source uses full version
            cuda_to_try = CUDA_VERSION_MAP["full"].get(cuda_full, [cuda_full])

        # Get torch versions to try (exact match first, then close versions)
        torch_to_try = []
        for tv in source["torch_versions"]:
            if tv.startswith(torch_major_minor) or torch_major_minor.startswith(tv.split('.')[0] + '.' + tv.split('.')[1] if '.' in tv else tv):
                torch_to_try.insert(0, tv)  # Exact/close match first
            else:
                torch_to_try.append(tv)

        # Filter to only compatible torch versions
        torch_to_try = [tv for tv in torch_to_try if _torch_compatible(torch_major_minor, tv)]

        for cuda in cuda_to_try:
            if cuda not in source["cuda_versions"]:
                continue
            for torch_v in torch_to_try:
                url = source["url_pattern"].format(
                    base_url=source["base_url"],
                    version=source["version"],
                    cuda=cuda,
                    torch=torch_v,
                    abi=source.get("cxx11abi", ""),
                    python=python_version,
                    platform=source["platform"],
                )
                urls.append((source["name"], url))

    return urls


def _torch_compatible(detected: str, available: str) -> bool:
    """Check if detected torch version is compatible with available wheel."""
    # Extract major.minor from both
    det_parts = detected.split('.')[:2]
    avail_parts = available.split('.')[:2]

    try:
        det_major, det_minor = int(det_parts[0]), int(det_parts[1])
        avail_major, avail_minor = int(avail_parts[0]), int(avail_parts[1])

        # Same major version, detected >= available (backward compatible)
        if det_major == avail_major and det_minor >= avail_minor:
            return True
        # Allow one minor version difference
        if det_major == avail_major and abs(det_minor - avail_minor) <= 1:
            return True

        return False
    except (ValueError, IndexError):
        return detected.startswith(available) or available.startswith(detected)


def _try_install_wheel(url: str) -> bool:
    """Attempt to install a wheel from URL."""
    cmd = [sys.executable, "-m", "pip", "install", url]

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=PIP_TIMEOUT
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


def install_flash_attn() -> InstallResult:
    """
    Install flash-attn from prebuilt wheels.

    Automatically selects the best wheel source based on:
    - Operating system (Linux vs Windows)
    - CUDA version
    - PyTorch version
    - Python version

    Returns:
        InstallResult with success status
    """
    # Check if already installed
    if check_package_installed("flash_attn"):
        InstallLogger.info("flash-attn already installed")
        return InstallResult(success=True, method="already_installed")

    # Get system info
    system_info = _get_system_info()

    if system_info is None:
        InstallLogger.info("CUDA not available or PyTorch not found, skipping flash-attn")
        return InstallResult(success=True, method="skipped", optional=True)

    os_name, torch_version, cuda_version, python_version = system_info

    InstallLogger.info(f"Detecting flash-attn wheel for: {os_name}, PyTorch {torch_version}, CUDA {cuda_version}, Python {python_version}")

    # Check if platform is supported
    if os_name not in WHEEL_SOURCES:
        InstallLogger.warning(f"No flash-attn wheels available for {os_name}")
        InstallLogger.info("flash-attn is required for UniRig. You may need to build from source.")
        return InstallResult(success=False, method="failed", optional=False, error=f"No wheels for {os_name}")

    # Build list of URLs to try
    wheel_urls = _build_wheel_urls(os_name, torch_version, cuda_version, python_version)

    if not wheel_urls:
        InstallLogger.warning("No compatible flash-attn wheel found for your configuration")
        InstallLogger.info(f"  OS: {os_name}")
        InstallLogger.info(f"  PyTorch: {torch_version}")
        InstallLogger.info(f"  CUDA: {cuda_version}")
        InstallLogger.info(f"  Python: {python_version}")
        InstallLogger.info("")
        InstallLogger.info("Please install flash-attn manually from one of:")
        InstallLogger.info("  Linux: https://github.com/Dao-AILab/flash-attention/releases")
        InstallLogger.info("  Linux: https://github.com/mjun0812/flash-attention-prebuild-wheels/releases")
        InstallLogger.info("  Windows: https://github.com/bdashore3/flash-attention/releases")
        return InstallResult(success=False, method="failed", optional=False, error="No compatible wheel")

    # Try each URL in order
    for source_name, url in wheel_urls:
        InstallLogger.info(f"Trying {source_name}: {url[:80]}...")

        if _try_install_wheel(url):
            InstallLogger.success(f"flash-attn installed from {source_name}")
            return InstallResult(success=True, method=f"wheel ({source_name})")

    # All attempts failed
    InstallLogger.error("=" * 60)
    InstallLogger.error("flash-attn installation FAILED")
    InstallLogger.error("=" * 60)
    InstallLogger.error("flash-attn is REQUIRED for UniRig model inference.")
    InstallLogger.error("Without flash-attn, UniRig nodes will NOT work correctly.")
    InstallLogger.error("")
    InstallLogger.error("Please install manually from one of these sources:")
    if os_name == "linux":
        InstallLogger.error("  https://github.com/Dao-AILab/flash-attention/releases")
        InstallLogger.error("  https://github.com/mjun0812/flash-attention-prebuild-wheels/releases")
    else:
        InstallLogger.error("  https://github.com/bdashore3/flash-attention/releases")
    InstallLogger.error("")
    InstallLogger.error(f"Your configuration: {os_name}, PyTorch {torch_version}, CUDA {cuda_version}, Python {python_version}")
    InstallLogger.error("=" * 60)

    return InstallResult(success=False, method="failed", optional=False, error="All wheel sources failed")
