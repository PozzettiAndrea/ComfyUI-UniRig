"""
Model loader nodes for UniRig - Pre-load and cache ML models for faster inference.

These nodes download and cache models so subsequent inference runs are faster.
When cache_to_gpu is enabled, models are loaded directly into GPU memory in the main process.
"""

import os
import sys
import yaml
from pathlib import Path
from box import Box

# Support both relative imports (ComfyUI) and absolute imports (testing)
try:
    from .base import UNIRIG_PATH, UNIRIG_MODELS_DIR, LIB_DIR
except ImportError:
    from base import UNIRIG_PATH, UNIRIG_MODELS_DIR, LIB_DIR

# Global model cache (for config dicts)
_MODEL_CACHE = {}

# In-process model cache module
_MODEL_CACHE_MODULE = None


def _get_model_cache():
    """Get the in-process model cache module."""
    global _MODEL_CACHE_MODULE
    if _MODEL_CACHE_MODULE is None:
        # Use sys.modules to ensure same instance across all imports
        if "unirig_model_cache" in sys.modules:
            _MODEL_CACHE_MODULE = sys.modules["unirig_model_cache"]
        else:
            cache_path = os.path.join(LIB_DIR, "model_cache.py")
            if os.path.exists(cache_path):
                import importlib.util
                spec = importlib.util.spec_from_file_location("unirig_model_cache", cache_path)
                _MODEL_CACHE_MODULE = importlib.util.module_from_spec(spec)
                sys.modules["unirig_model_cache"] = _MODEL_CACHE_MODULE
                spec.loader.exec_module(_MODEL_CACHE_MODULE)
            else:
                print(f"[UniRig] Warning: Model cache module not found at {cache_path}")
                _MODEL_CACHE_MODULE = False
    return _MODEL_CACHE_MODULE if _MODEL_CACHE_MODULE else None


def _ensure_unirig_in_path():
    """Ensure UniRig is in Python path."""
    if UNIRIG_PATH not in sys.path:
        sys.path.insert(0, UNIRIG_PATH)


def _load_yaml_config(config_path: str) -> Box:
    """Load a YAML config file."""
    if config_path.endswith('.yaml'):
        config_path = config_path.removesuffix('.yaml')
    config_path += '.yaml'
    return Box(yaml.safe_load(open(config_path, 'r')))


class UniRigLoadSkeletonModel:
    """
    Load and cache the UniRig skeleton extraction model.

    This pre-downloads the model weights and prepares configuration
    for faster skeleton inference. Connect this to UniRigExtractSkeleton
    to avoid model reload on each run.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_id": ("STRING", {
                    "default": "VAST-AI/UniRig",
                    "tooltip": "HuggingFace model ID for skeleton model"
                }),
                "cache_to_gpu": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep model cached on GPU for faster inference. Disable to offload to CPU after inference (saves VRAM)."
                }),
            },
        }

    RETURN_TYPES = ("UNIRIG_SKELETON_MODEL",)
    RETURN_NAMES = ("skeleton_model",)
    FUNCTION = "load_model"
    CATEGORY = "UniRig/Models"

    def load_model(self, model_id="VAST-AI/UniRig", cache_to_gpu=True):
        """Load and cache skeleton model configuration."""
        print(f"[UniRigLoadSkeletonModel] Loading skeleton model: {model_id}")
        print(f"[UniRigLoadSkeletonModel] GPU caching: {'enabled' if cache_to_gpu else 'disabled (will offload to CPU)'}")

        cache_key = f"skeleton_{model_id}"

        # Check cache
        if cache_key in _MODEL_CACHE:
            cached_model = _MODEL_CACHE[cache_key]
            # Update cache_to_gpu setting in case it changed
            cached_model["cache_to_gpu"] = cache_to_gpu
            print(f"[UniRigLoadSkeletonModel] Using cached model configuration")
            return (cached_model,)

        _ensure_unirig_in_path()

        # Pre-download model weights
        try:
            from src.inference.download import download

            # Load task config to get checkpoint path
            task_config_path = os.path.join(
                UNIRIG_PATH,
                "configs/task/quick_inference_skeleton_articulationxl_ar_256.yaml"
            )
            task_config = _load_yaml_config(task_config_path)

            # Download checkpoint if needed
            checkpoint_path = task_config.get('resume_from_checkpoint', None)
            if checkpoint_path:
                print(f"[UniRigLoadSkeletonModel] Downloading/verifying checkpoint...")
                local_checkpoint = download(checkpoint_path)
                print(f"[UniRigLoadSkeletonModel] Checkpoint ready: {local_checkpoint}")
            else:
                local_checkpoint = None

            # Load model config
            model_config_name = task_config.components.get('model', None)
            if model_config_name:
                model_config = _load_yaml_config(
                    os.path.join(UNIRIG_PATH, 'configs/model', model_config_name)
                )
            else:
                model_config = {}

            # Load tokenizer config
            tokenizer_config_name = task_config.components.get('tokenizer', None)
            if tokenizer_config_name:
                tokenizer_config = _load_yaml_config(
                    os.path.join(UNIRIG_PATH, 'configs/tokenizer', tokenizer_config_name)
                )
            else:
                tokenizer_config = None

            # Create model wrapper
            model_wrapper = {
                "type": "skeleton",
                "model_id": model_id,
                "task_config_path": task_config_path,
                "checkpoint_path": local_checkpoint,
                "model_config": model_config.to_dict() if hasattr(model_config, 'to_dict') else dict(model_config),
                "tokenizer_config": tokenizer_config.to_dict() if tokenizer_config and hasattr(tokenizer_config, 'to_dict') else (dict(tokenizer_config) if tokenizer_config else None),
                "unirig_path": UNIRIG_PATH,
                "models_dir": str(UNIRIG_MODELS_DIR),
                "cached": True,
                "cache_to_gpu": cache_to_gpu,
                "model_cache_key": None,
            }

            # If cache_to_gpu is enabled, load model into GPU memory
            if cache_to_gpu:
                model_cache = _get_model_cache()
                if model_cache:
                    print(f"[UniRigLoadSkeletonModel] Loading model into GPU memory...")
                    try:
                        model_cache_key = model_cache.load_model_into_memory(
                            model_type="skeleton",
                            task_config_path=task_config_path,
                            cache_to_gpu=True
                        )
                        model_wrapper["model_cache_key"] = model_cache_key
                        print(f"[UniRigLoadSkeletonModel] Model loaded into GPU memory")
                    except Exception as e:
                        print(f"[UniRigLoadSkeletonModel] Warning: Failed to load into GPU memory: {e}")
                        print(f"[UniRigLoadSkeletonModel] Will use subprocess fallback")
                else:
                    print(f"[UniRigLoadSkeletonModel] Warning: Model cache not available")

            # Cache it
            _MODEL_CACHE[cache_key] = model_wrapper

            print(f"[UniRigLoadSkeletonModel] Model configuration cached successfully")
            print(f"[UniRigLoadSkeletonModel] Checkpoint: {local_checkpoint}")

            return (model_wrapper,)

        except Exception as e:
            print(f"[UniRigLoadSkeletonModel] Error loading model: {e}")
            import traceback
            traceback.print_exc()

            # Return minimal config that will trigger full load in inference
            model_wrapper = {
                "type": "skeleton",
                "model_id": model_id,
                "task_config_path": os.path.join(
                    UNIRIG_PATH,
                    "configs/task/quick_inference_skeleton_articulationxl_ar_256.yaml"
                ),
                "unirig_path": UNIRIG_PATH,
                "models_dir": str(UNIRIG_MODELS_DIR),
                "cached": False,
            }
            return (model_wrapper,)


class UniRigLoadSkinningModel:
    """
    Load and cache the UniRig skinning weight prediction model.

    This pre-downloads the model weights and prepares configuration
    for faster skinning inference. Connect this to UniRigApplySkinningML
    to avoid model reload on each run.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_id": ("STRING", {
                    "default": "VAST-AI/UniRig",
                    "tooltip": "HuggingFace model ID for skinning model"
                }),
                "cache_to_gpu": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep model cached on GPU for faster inference. Disable to offload to CPU after inference (saves VRAM)."
                }),
            },
        }

    RETURN_TYPES = ("UNIRIG_SKINNING_MODEL",)
    RETURN_NAMES = ("skinning_model",)
    FUNCTION = "load_model"
    CATEGORY = "UniRig/Models"

    def load_model(self, model_id="VAST-AI/UniRig", cache_to_gpu=True):
        """Load and cache skinning model configuration."""
        print(f"[UniRigLoadSkinningModel] Loading skinning model: {model_id}")
        print(f"[UniRigLoadSkinningModel] GPU caching: {'enabled' if cache_to_gpu else 'disabled (will offload to CPU)'}")

        cache_key = f"skinning_{model_id}"

        # Check cache
        if cache_key in _MODEL_CACHE:
            cached_model = _MODEL_CACHE[cache_key]
            # Update cache_to_gpu setting in case it changed
            cached_model["cache_to_gpu"] = cache_to_gpu
            print(f"[UniRigLoadSkinningModel] Using cached model configuration")
            return (cached_model,)

        _ensure_unirig_in_path()

        # Pre-download model weights
        try:
            from src.inference.download import download

            # Load task config to get checkpoint path
            task_config_path = os.path.join(
                UNIRIG_PATH,
                "configs/task/quick_inference_unirig_skin.yaml"
            )
            task_config = _load_yaml_config(task_config_path)

            # Download checkpoint if needed
            checkpoint_path = task_config.get('resume_from_checkpoint', None)
            if checkpoint_path:
                print(f"[UniRigLoadSkinningModel] Downloading/verifying checkpoint...")
                local_checkpoint = download(checkpoint_path)
                print(f"[UniRigLoadSkinningModel] Checkpoint ready: {local_checkpoint}")
            else:
                local_checkpoint = None

            # Load model config
            model_config_name = task_config.components.get('model', None)
            if model_config_name:
                model_config = _load_yaml_config(
                    os.path.join(UNIRIG_PATH, 'configs/model', model_config_name)
                )
            else:
                model_config = {}

            # Create model wrapper
            model_wrapper = {
                "type": "skinning",
                "model_id": model_id,
                "task_config_path": task_config_path,
                "checkpoint_path": local_checkpoint,
                "model_config": model_config.to_dict() if hasattr(model_config, 'to_dict') else dict(model_config),
                "unirig_path": UNIRIG_PATH,
                "models_dir": str(UNIRIG_MODELS_DIR),
                "cached": True,
                "cache_to_gpu": cache_to_gpu,
                "model_cache_key": None,
            }

            # If cache_to_gpu is enabled, load model into GPU memory
            if cache_to_gpu:
                model_cache = _get_model_cache()
                if model_cache:
                    print(f"[UniRigLoadSkinningModel] Loading model into GPU memory...")
                    try:
                        model_cache_key = model_cache.load_model_into_memory(
                            model_type="skinning",
                            task_config_path=task_config_path,
                            cache_to_gpu=True
                        )
                        model_wrapper["model_cache_key"] = model_cache_key
                        print(f"[UniRigLoadSkinningModel] Model loaded into GPU memory")
                    except Exception as e:
                        print(f"[UniRigLoadSkinningModel] Warning: Failed to load into GPU memory: {e}")
                        print(f"[UniRigLoadSkinningModel] Will use subprocess fallback")
                else:
                    print(f"[UniRigLoadSkinningModel] Warning: Model cache not available")

            # Cache it
            _MODEL_CACHE[cache_key] = model_wrapper

            print(f"[UniRigLoadSkinningModel] Model configuration cached successfully")
            print(f"[UniRigLoadSkinningModel] Checkpoint: {local_checkpoint}")

            return (model_wrapper,)

        except Exception as e:
            print(f"[UniRigLoadSkinningModel] Error loading model: {e}")
            import traceback
            traceback.print_exc()

            # Return minimal config
            model_wrapper = {
                "type": "skinning",
                "model_id": model_id,
                "task_config_path": os.path.join(
                    UNIRIG_PATH,
                    "configs/task/quick_inference_unirig_skin.yaml"
                ),
                "unirig_path": UNIRIG_PATH,
                "models_dir": str(UNIRIG_MODELS_DIR),
                "cached": False,
            }
            return (model_wrapper,)


def clear_model_cache():
    """Clear the global model cache."""
    global _MODEL_CACHE
    _MODEL_CACHE.clear()
    print("[UniRig] Model cache cleared")


def get_cached_models():
    """Get list of cached model keys."""
    return list(_MODEL_CACHE.keys())
