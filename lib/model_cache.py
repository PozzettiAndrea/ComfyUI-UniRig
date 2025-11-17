"""
UniRig In-Process Model Cache

Keeps ML models loaded in GPU memory for fast inference.
No separate server process needed - models live in the main ComfyUI process.
"""

import os
import sys
from pathlib import Path
import torch
import lightning as L
import yaml
from box import Box
import numpy as np

# Add UniRig to path
LIB_DIR = Path(__file__).parent.resolve()
UNIRIG_PATH = LIB_DIR / "unirig"
if str(UNIRIG_PATH) not in sys.path:
    sys.path.insert(0, str(UNIRIG_PATH))

# Set BLENDER_EXE environment variable if not already set
if 'BLENDER_EXE' not in os.environ:
    import platform
    # Check common Blender locations
    blender_candidates = []
    if platform.system() == "Windows":
        blender_candidates = [
            LIB_DIR / "blender" / "blender.exe",
        ]
    elif platform.system() == "Darwin":  # Mac
        blender_candidates = [
            LIB_DIR / "blender" / "Blender.app" / "Contents" / "MacOS" / "Blender",
            LIB_DIR / "blender" / "blender",
        ]
    else:  # Linux
        blender_candidates = [
            LIB_DIR / "blender" / "blender-4.2.3-linux-x64" / "blender",
            LIB_DIR / "blender" / "blender",
        ]

    for blender_path in blender_candidates:
        if blender_path.exists():
            os.environ['BLENDER_EXE'] = str(blender_path)
            print(f"[UniRigCache] Found Blender: {blender_path}")
            break

from src.inference.download import download
from src.data.extract import get_files
from src.data.dataset import UniRigDatasetModule, DatasetConfig
from src.data.datapath import Datapath
from src.data.transform import TransformConfig
from src.tokenizer.spec import TokenizerConfig
from src.tokenizer.parse import get_tokenizer
from src.model.parse import get_model
from src.system.parse import get_system, get_writer

# Global model cache - keeps models in GPU memory
_LOADED_MODELS = {}


def load_yaml_config(path: str) -> Box:
    """Load a YAML config file."""
    if path.endswith('.yaml'):
        path = path.removesuffix('.yaml')
    path += '.yaml'
    return Box(yaml.safe_load(open(path, 'r')))


def load_model_into_memory(model_type: str, task_config_path: str, cache_to_gpu: bool = True):
    """
    Load a model into GPU memory and cache it.

    Args:
        model_type: "skeleton" or "skinning"
        task_config_path: Path to task config YAML
        cache_to_gpu: If True, keep model on GPU; if False, keep on CPU

    Returns:
        Cache key for the loaded model
    """
    cache_key = f"{model_type}_{task_config_path}"

    if cache_key in _LOADED_MODELS:
        print(f"[UniRigCache] Model {model_type} already loaded")
        return cache_key

    print(f"[UniRigCache] Loading {model_type} model...")

    # Change to UniRig directory for relative paths
    original_cwd = os.getcwd()
    os.chdir(str(UNIRIG_PATH))

    try:
        task = load_yaml_config(task_config_path)

        # Load tokenizer config
        tokenizer_config = task.components.get('tokenizer', None)
        if tokenizer_config is not None:
            tokenizer_config = load_yaml_config(os.path.join(str(UNIRIG_PATH), 'configs/tokenizer', task.components.tokenizer))
            tokenizer_config = TokenizerConfig.parse(config=tokenizer_config)

        # Load model config
        model_config_name = task.components.get('model', None)
        if model_config_name is not None:
            model_config = load_yaml_config(os.path.join(str(UNIRIG_PATH), 'configs/model', model_config_name))
            if tokenizer_config is not None:
                tokenizer = get_tokenizer(config=tokenizer_config)
            else:
                tokenizer = None
            model = get_model(tokenizer=tokenizer, **model_config)
        else:
            model = None

        # Load checkpoint
        resume_from_checkpoint = task.get('resume_from_checkpoint', None)
        checkpoint_path = download(resume_from_checkpoint)

        # Load system
        system_config_name = task.components.get('system', None)
        if system_config_name is not None:
            system_config = load_yaml_config(os.path.join(str(UNIRIG_PATH), 'configs/system', system_config_name))
            system = get_system(
                **system_config,
                model=model,
                optimizer_config=None,
                loss_config=None,
                scheduler_config=None,
                steps_per_epoch=1,
            )
        else:
            system = None

        # Load checkpoint weights into system
        if checkpoint_path and system is not None:
            print(f"[UniRigCache] Loading checkpoint weights...")
            # PyTorch 2.6+ changed default to weights_only=True
            torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
            if torch_version >= (2, 6):
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            else:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
            system.load_state_dict(checkpoint['state_dict'], strict=False)

            if cache_to_gpu and torch.cuda.is_available():
                system = system.cuda()
                print(f"[UniRigCache] Model moved to GPU")
            else:
                print(f"[UniRigCache] Model on CPU")

            system.eval()

        # Cache the model components
        _LOADED_MODELS[cache_key] = {
            "task": task,
            "model": model,
            "system": system,
            "tokenizer_config": tokenizer_config,
            "checkpoint_path": checkpoint_path,
            "cache_to_gpu": cache_to_gpu,
        }

        print(f"[UniRigCache] Model {model_type} loaded and cached")
        return cache_key

    finally:
        os.chdir(original_cwd)


def apply_config_overrides(config, overrides: dict):
    """
    Apply config overrides to a Box/dict config object.

    Args:
        config: Box or dict config object
        overrides: Dictionary of override values

    Returns:
        Modified config object
    """
    if not overrides:
        return config

    # Apply overrides to transform config (sampler and vertex group)
    if hasattr(config, 'predict_transform_config') or 'predict_transform_config' in config:
        ptc = config.get('predict_transform_config', {})

        # Override sampler config
        if 'sampler_config' in ptc:
            if 'num_samples' in overrides:
                ptc['sampler_config']['num_samples'] = overrides['num_samples']
            if 'vertex_samples' in overrides:
                ptc['sampler_config']['vertex_samples'] = overrides['vertex_samples']

        # Override vertex group config (voxel_skin)
        if 'vertex_group_config' in ptc and 'kwargs' in ptc['vertex_group_config']:
            vg_kwargs = ptc['vertex_group_config']['kwargs']
            if 'voxel_skin' in vg_kwargs:
                vs = vg_kwargs['voxel_skin']
                if 'voxel_grid_size' in overrides:
                    vs['grid'] = overrides['voxel_grid_size']
                # Map voxel_mask_power to alpha (voxel_mask_power is the UI name)
                if 'voxel_mask_power' in overrides:
                    vs['alpha'] = overrides['voxel_mask_power']
                elif 'alpha' in overrides:
                    vs['alpha'] = overrides['alpha']
                if 'grid_query' in overrides:
                    vs['grid_query'] = overrides['grid_query']
                if 'vertex_query' in overrides:
                    vs['vertex_query'] = overrides['vertex_query']
                if 'grid_weight' in overrides:
                    vs['grid_weight'] = overrides['grid_weight']

    return config


def run_inference(cache_key: str, request_data: dict) -> dict:
    """
    Run inference using a cached model.

    Args:
        cache_key: Key for cached model
        request_data: Dictionary containing inference parameters

    Returns:
        Dictionary with inference results
    """
    if cache_key not in _LOADED_MODELS:
        return {"error": f"Model {cache_key} not loaded"}

    cached = _LOADED_MODELS[cache_key]
    task = cached["task"]
    system = cached["system"]
    tokenizer_config = cached["tokenizer_config"]

    # Extract request parameters
    seed = request_data.get("seed", 123)
    input_file = request_data.get("input")
    output_file = request_data.get("output")
    npz_dir = request_data.get("npz_dir")
    cls = request_data.get("cls")
    data_name = request_data.get("data_name")
    config_overrides = request_data.get("config_overrides", {})

    if not all([input_file, output_file, npz_dir]):
        return {"error": "Missing required parameters: input, output, npz_dir"}

    # Change to UniRig directory for relative paths
    original_cwd = os.getcwd()
    os.chdir(str(UNIRIG_PATH))

    try:
        # Set seed
        L.seed_everything(seed, workers=True)

        # Prepare data
        files = get_files(
            data_name=task.components.data_name,
            inputs=input_file,
            input_dataset_dir=None,
            output_dataset_dir=npz_dir,
            force_override=True,
            warning=False,
        )
        files = [f[1] for f in files]
        datapath = Datapath(files=files, cls=cls)

        # Load configs
        data_config = load_yaml_config(os.path.join(str(UNIRIG_PATH), 'configs/data', task.components.data))
        transform_config = load_yaml_config(os.path.join(str(UNIRIG_PATH), 'configs/transform', task.components.transform))

        # Apply config overrides to transform config
        if config_overrides:
            print(f"[UniRigCache] Applying config overrides: {config_overrides}")
            transform_config = apply_config_overrides(transform_config, config_overrides)

        # Get data name
        data_name_actual = task.components.get('data_name', 'raw_data.npz')
        if data_name is not None:
            data_name_actual = data_name

        # Get predict dataset config
        predict_dataset_config = data_config.get('predict_dataset_config', None)
        if predict_dataset_config is not None:
            predict_dataset_config = DatasetConfig.parse(config=predict_dataset_config).split_by_cls()

        # Get predict transform config
        predict_transform_config = transform_config.get('predict_transform_config', None)
        if predict_transform_config is not None:
            predict_transform_config = TransformConfig.parse(config=predict_transform_config)

        # Create data module
        data = UniRigDatasetModule(
            process_fn=None if system is None or system.model is None else system.model._process_fn,
            train_dataset_config=None,
            predict_dataset_config=predict_dataset_config,
            predict_transform_config=predict_transform_config,
            validate_dataset_config=None,
            train_transform_config=None,
            validate_transform_config=None,
            tokenizer_config=tokenizer_config,
            debug=False,
            data_name=data_name_actual,
            datapath=datapath,
            cls=cls,
        )

        # Setup callbacks
        callbacks = []
        writer_config = task.get('writer', None)
        if writer_config is not None:
            if output_file.endswith('.fbx'):
                writer_config['npz_dir'] = npz_dir
                writer_config['output_dir'] = None
                writer_config['output_name'] = output_file
                writer_config['user_mode'] = True
            callbacks.append(get_writer(**writer_config, order_config=predict_transform_config.order_config))

        # Create trainer with progress bar disabled
        trainer_config = task.get('trainer', {})
        trainer = L.Trainer(
            callbacks=callbacks,
            logger=None,
            enable_progress_bar=False,
            **trainer_config,
        )

        # Run prediction
        trainer.predict(system, datamodule=data, ckpt_path=None, return_predictions=False)
        return {"success": True, "output": output_file}

    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}
    finally:
        os.chdir(original_cwd)


def unload_model(cache_key: str):
    """Unload a model from cache to free GPU memory."""
    if cache_key in _LOADED_MODELS:
        cached = _LOADED_MODELS[cache_key]
        # Move model to CPU and delete
        if cached.get("system") is not None:
            cached["system"].cpu()
        del _LOADED_MODELS[cache_key]
        torch.cuda.empty_cache()
        print(f"[UniRigCache] Model {cache_key} unloaded")
        return True
    return False


def list_loaded_models():
    """List all loaded models."""
    return list(_LOADED_MODELS.keys())


def is_model_loaded(cache_key: str) -> bool:
    """Check if a model is loaded."""
    return cache_key in _LOADED_MODELS


def clear_cache():
    """Clear all cached models."""
    for key in list(_LOADED_MODELS.keys()):
        unload_model(key)
    print("[UniRigCache] All models unloaded")
