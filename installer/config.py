"""
Configuration constants for ComfyUI-UniRig installer.
"""

# Supported Python versions
SUPPORTED_PYTHON = ("3.10", "3.11", "3.12", "3.13")

# Source repositories for fallback builds
SOURCE_REPOS = {
    "torch-scatter": "https://github.com/rusty1s/pytorch_scatter.git",
    "torch-cluster": "https://github.com/rusty1s/pytorch_cluster.git",
}

# CUDA version mapping for spconv compatibility
# Newer CUDA versions fall back to older spconv versions
CUDA_TO_SPCONV = {
    'cu130': ['cu121', 'cu120'],  # CUDA 13.0 -> try cu121 or cu120
    'cu129': ['cu121', 'cu120'],  # CUDA 12.9 -> try cu121 or cu120
    'cu128': ['cu121', 'cu120'],
    'cu127': ['cu121', 'cu120'],
    'cu126': ['cu121', 'cu120'],
    'cu125': ['cu121', 'cu120'],
    'cu124': ['cu121', 'cu120'],
    'cu123': ['cu121', 'cu120'],
    'cu122': ['cu121', 'cu120'],
    'cu121': ['cu121', 'cu120'],
    'cu120': ['cu120'],
    'cu118': ['cu118'],
    'cu117': ['cu117'],
}

# CUDA architectures for source builds
TORCH_CUDA_ARCH_LIST = "7.0;7.5;8.0;8.6;8.9;9.0"

# Blender version for mesh preprocessing
BLENDER_VERSION = "4.2.3"
BLENDER_BASE_URL = "https://download.blender.org/release/Blender4.2"

# flash-attn version
FLASH_ATTN_VERSION = "2.8.3"

# Timeouts (in seconds)
PIP_TIMEOUT = 300  # 5 minutes for normal pip installs
SOURCE_BUILD_TIMEOUT = 1200  # 20 minutes for source builds
DOWNLOAD_TIMEOUT = 600  # 10 minutes for downloads
