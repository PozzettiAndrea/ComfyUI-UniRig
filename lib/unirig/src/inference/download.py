import os
from pathlib import Path
from huggingface_hub import hf_hub_download

def _find_cached_model(filename: str) -> str | None:
    """Check if model exists in HuggingFace cache (either ComfyUI or user cache)."""
    cache_locations = [
        os.environ.get('HF_HUB_CACHE'),  # ComfyUI's models/unirig/hub
        Path.home() / ".cache" / "huggingface" / "hub",  # User's default cache
    ]

    for cache_dir in cache_locations:
        if not cache_dir:
            continue
        cache_path = Path(cache_dir)
        model_dir = cache_path / "models--VAST-AI--UniRig" / "snapshots"

        if not model_dir.exists():
            continue

        # Check all snapshot directories for the file
        for snapshot in model_dir.iterdir():
            if snapshot.is_dir():
                file_path = snapshot / filename
                if file_path.exists():
                    print(f"[UniRig] Found cached model: {file_path}")
                    return str(file_path)

    return None

def download(ckpt_name: str) -> str:
    MAP = {
        'experiments/skeleton/articulation-xl_quantization_256/model.ckpt': 'skeleton/articulation-xl_quantization_256/model.ckpt',
        'experiments/skin/articulation-xl/model.ckpt': 'skin/articulation-xl/model.ckpt',
        'experiments/skin/skeleton/model.ckpt': 'skin/skeleton/model.ckpt',
    }

    try:
        if ckpt_name not in MAP:
            print(f"not found: {ckpt_name}")
            return ckpt_name

        filename = MAP[ckpt_name]

        # Check if already cached (avoid re-download)
        cached_path = _find_cached_model(filename)
        if cached_path:
            return cached_path

        # Download to ComfyUI's models folder (HF_HUB_CACHE is set in base.py)
        print(f"[UniRig] Downloading {filename}...")
        return hf_hub_download(
            repo_id='VAST-AI/UniRig',
            filename=filename,
        )
    except Exception as e:
        print(f"Failed to download {ckpt_name}: {e}")
        return ckpt_name