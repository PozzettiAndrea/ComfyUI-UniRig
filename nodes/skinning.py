"""
Skinning nodes for UniRig - Apply skinning weights using ML models.
"""

import os
import sys
import subprocess
import tempfile
import numpy as np
import time
import shutil
import glob
import json
import folder_paths

# Support both relative imports (ComfyUI) and absolute imports (testing)
try:
    from ..constants import BLENDER_TIMEOUT, INFERENCE_TIMEOUT
except ImportError:
    from constants import BLENDER_TIMEOUT, INFERENCE_TIMEOUT

try:
    from .base import (
        UNIRIG_PATH,
        BLENDER_EXE,
        UNIRIG_MODELS_DIR,
        LIB_DIR,
        setup_subprocess_env,
        decode_texture_to_comfy_image,
        create_placeholder_texture,
    )
except ImportError:
    from base import (
        UNIRIG_PATH,
        BLENDER_EXE,
        UNIRIG_MODELS_DIR,
        LIB_DIR,
        setup_subprocess_env,
        decode_texture_to_comfy_image,
        create_placeholder_texture,
    )

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


class UniRigApplySkinningMLNew:
    """
    Apply skinning weights using ML - CACHED MODEL ONLY.

    Takes skeleton dict and mesh, prepares data and runs ML inference.

    This version uses ONLY in-process GPU cached models for faster inference.
    Requires pre-loaded model from UniRigLoadSkinningModel.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "normalized_mesh": ("TRIMESH",),
                "skeleton": ("SKELETON",),
                "skinning_model": ("UNIRIG_SKINNING_MODEL", {
                    "tooltip": "Pre-loaded skinning model (from UniRigLoadSkinningModel) - REQUIRED"
                }),
            },
            "optional": {
                "voxel_grid_size": ("INT", {
                    "default": 196,
                    "min": 64,
                    "max": 512,
                    "step": 64,
                    "tooltip": "Voxel grid resolution for spatial weight distribution. Higher = better quality, more VRAM. Default: 196 (model trained with this)"
                }),
                "num_samples": ("INT", {
                    "default": 32768,
                    "min": 8192,
                    "max": 131072,
                    "step": 8192,
                    "tooltip": "Number of surface samples for weight calculation. Higher = more accurate, slower. Default: 32768"
                }),
                "vertex_samples": ("INT", {
                    "default": 8192,
                    "min": 2048,
                    "max": 32768,
                    "step": 2048,
                    "tooltip": "Number of vertex samples. Higher = more accurate vertex processing. Default: 8192"
                }),
                "voxel_mask_power": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Power for voxel mask weight sharpness (alpha). Lower = smoother transitions. Default: 0.5 (model trained with this)"
                }),
            }
        }

    RETURN_TYPES = ("RIGGED_MESH", "IMAGE")
    RETURN_NAMES = ("rigged_mesh", "texture_preview")
    FUNCTION = "apply_skinning"
    CATEGORY = "UniRig"

    def apply_skinning(self, normalized_mesh, skeleton, skinning_model,
                       voxel_grid_size=None, num_samples=None, vertex_samples=None,
                       voxel_mask_power=None):
        print(f"[UniRigApplySkinningMLNew] Starting ML skinning (cached model only)...")

        # Validate model is provided
        if skinning_model is None:
            raise RuntimeError(
                "skinning_model is required for UniRigApplySkinningMLNew. "
                "Please connect a UniRigLoadSkinningModel node."
            )

        # Validate model has cache key
        if not skinning_model.get("model_cache_key"):
            raise RuntimeError(
                "skinning_model does not have a cached model. "
                "Ensure UniRigLoadSkinningModel has 'cache_to_gpu' enabled."
            )

        print(f"[UniRigApplySkinningMLNew] Using pre-loaded cached model")
        task_config_path = skinning_model.get("task_config_path")

        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix="unirig_skinning_new_")
        predict_skeleton_dir = os.path.join(temp_dir, "input")
        os.makedirs(predict_skeleton_dir, exist_ok=True)

        # Prepare skeleton NPZ from dict
        predict_skeleton_path = os.path.join(predict_skeleton_dir, "predict_skeleton.npz")
        save_data = {
            'joints': skeleton['joints'],
            'names': skeleton['names'],
            'parents': skeleton['parents'],
            'tails': skeleton['tails'],
        }

        # Add mesh data
        mesh_data_mapping = {
            'mesh_vertices': 'vertices',
            'mesh_faces': 'faces',
            'mesh_vertex_normals': 'vertex_normals',
            'mesh_face_normals': 'face_normals',
        }
        for skel_key, npz_key in mesh_data_mapping.items():
            if skel_key in skeleton:
                save_data[npz_key] = skeleton[skel_key]

        # Add optional RawData fields
        save_data['skin'] = None
        save_data['no_skin'] = None
        save_data['matrix_local'] = skeleton.get('matrix_local')
        save_data['path'] = None
        save_data['cls'] = skeleton.get('cls')

        # Add UV data if available
        if skeleton.get('uv_coords') is not None:
            save_data['uv_coords'] = skeleton['uv_coords']
            save_data['uv_faces'] = skeleton.get('uv_faces')
            print(f"[UniRigApplySkinningMLNew] UV data included: {len(skeleton['uv_coords'])} UVs")
        else:
            save_data['uv_coords'] = np.array([], dtype=np.float32)
            save_data['uv_faces'] = np.array([], dtype=np.int32)

        # Add texture data if available
        if skeleton.get('texture_data_base64') is not None:
            save_data['texture_data_base64'] = skeleton['texture_data_base64']
            save_data['texture_format'] = skeleton.get('texture_format', 'PNG')
            save_data['texture_width'] = skeleton.get('texture_width', 0)
            save_data['texture_height'] = skeleton.get('texture_height', 0)
            save_data['material_name'] = skeleton.get('material_name', '')
            print(f"[UniRigApplySkinningMLNew] Texture data included: {skeleton['texture_width']}x{skeleton['texture_height']} {skeleton['texture_format']}")
        else:
            save_data['texture_data_base64'] = ""
            save_data['texture_format'] = ""
            save_data['texture_width'] = 0
            save_data['texture_height'] = 0
            save_data['material_name'] = skeleton.get('material_name', '')

        np.savez(predict_skeleton_path, **save_data)
        print(f"[UniRigApplySkinningMLNew] Prepared skeleton NPZ: {predict_skeleton_path}")

        # DEBUG: Show what we're sending to ML
        print(f"[UniRigApplySkinningMLNew] DEBUG - NPZ joints bounds: {save_data['joints'].min(axis=0)} to {save_data['joints'].max(axis=0)}")
        print(f"[UniRigApplySkinningMLNew] DEBUG - NPZ tails bounds: {save_data['tails'].min(axis=0)} to {save_data['tails'].max(axis=0)}")
        if 'vertices' in save_data and save_data['vertices'] is not None:
            print(f"[UniRigApplySkinningMLNew] DEBUG - NPZ mesh vertices bounds: {np.array(save_data['vertices']).min(axis=0)} to {np.array(save_data['vertices']).max(axis=0)}")

        # DIAGNOSTIC: Compute checksums and shapes for all data
        import hashlib
        print(f"[UniRigApplySkinningMLNew] ===== DATA DIAGNOSTIC =====")
        for key, value in save_data.items():
            if value is not None and hasattr(value, '__len__'):
                if isinstance(value, np.ndarray):
                    checksum = hashlib.md5(value.tobytes()).hexdigest()[:8]
                    print(f"[UniRigApplySkinningMLNew]   {key}: shape={value.shape}, dtype={value.dtype}, checksum={checksum}")
                    # Add min/max/mean for numerical arrays
                    if value.size > 0 and np.issubdtype(value.dtype, np.number):
                        print(f"[UniRigApplySkinningMLNew]     → min={np.min(value):.8f}, max={np.max(value):.8f}, mean={np.mean(value):.8f}")
                elif isinstance(value, (list, tuple)):
                    print(f"[UniRigApplySkinningMLNew]   {key}: len={len(value)}, type={type(value).__name__}")
                else:
                    print(f"[UniRigApplySkinningMLNew]   {key}: type={type(value).__name__}, value={str(value)[:50]}")
            else:
                print(f"[UniRigApplySkinningMLNew]   {key}: {value}")
        print(f"[UniRigApplySkinningMLNew] ==========================")

        # Export mesh to GLB
        input_glb = os.path.join(temp_dir, "input.glb")

        normalized_mesh.export(input_glb)
        print(f"[UniRigApplySkinningMLNew] Exported mesh: {normalized_mesh.vertices.shape[0]} vertices, {normalized_mesh.faces.shape[0]} faces")
        print(f"[UniRigApplySkinningMLNew] DEBUG - GLB mesh bounds: {normalized_mesh.vertices.min(axis=0)} to {normalized_mesh.vertices.max(axis=0)}")

        # Run skinning inference with CACHED MODEL ONLY
        step_start = time.time()
        print(f"[UniRigApplySkinningMLNew] Running skinning inference with cached model...")

        output_fbx = os.path.join(temp_dir, "rigged.fbx")

        # Build config overrides from optional parameters
        config_overrides = {}
        if voxel_grid_size is not None:
            config_overrides['voxel_grid_size'] = voxel_grid_size
        if num_samples is not None:
            config_overrides['num_samples'] = num_samples
        if vertex_samples is not None:
            config_overrides['vertex_samples'] = vertex_samples
        if voxel_mask_power is not None:
            config_overrides['voxel_mask_power'] = voxel_mask_power

        if config_overrides:
            print(f"[UniRigApplySkinningMLNew] Config overrides: {config_overrides}")

        model_cache = _get_model_cache()
        if not model_cache:
            raise RuntimeError(
                "Model cache module not available. "
                "Cannot run cached inference."
            )

        cache_key = skinning_model["model_cache_key"]
        print(f"[UniRigApplySkinningMLNew] Using cached model: {cache_key}")

        request_data = {
            "seed": 123,
            "input": input_glb,
            "output": output_fbx,
            "npz_dir": temp_dir,
            "cls": skeleton.get('cls'),
            "data_name": "predict_skeleton.npz",
            "config_overrides": config_overrides,
        }

        # DIAGNOSTIC: Log request parameters
        print(f"[UniRigApplySkinningMLNew] ===== REQUEST DIAGNOSTIC =====")
        for key, value in request_data.items():
            if key != "config_overrides":
                print(f"[UniRigApplySkinningMLNew]   {key}: {value}")
            else:
                print(f"[UniRigApplySkinningMLNew]   config_overrides: {value}")
        print(f"[UniRigApplySkinningMLNew] ==============================")

        try:
            result = model_cache.run_inference(cache_key, request_data)
            if "error" in result:
                error_msg = result['error']
                traceback_msg = result.get('traceback', 'No traceback available')
                raise RuntimeError(
                    f"Cached model inference failed: {error_msg}\n"
                    f"Traceback:\n{traceback_msg}\n\n"
                    f"This node requires a working cached model. "
                    f"If you need fallback support, use UniRigApplySkinningML instead."
                )

            inference_time = time.time() - step_start
            print(f"[UniRigApplySkinningMLNew] ✓ Cached inference completed in {inference_time:.2f}s")

        except Exception as e:
            raise RuntimeError(
                f"Cached model inference exception: {str(e)}\n\n"
                f"This node requires a working cached model. "
                f"If you need fallback support, use UniRigApplySkinningML instead."
            )

        print(f"[UniRigApplySkinningMLNew] Skinning completed")

        # Find output FBX
        possible_paths = [
            output_fbx,
            os.path.join(temp_dir, "rigged.fbx"),
            os.path.join(temp_dir, "output", "rigged.fbx"),
        ]

        fbx_path = None
        for path in possible_paths:
            if os.path.exists(path):
                fbx_path = path
                break

        if not fbx_path:
            # Search for any FBX files
            search_paths = [temp_dir, os.path.join(temp_dir, "output")]
            for search_dir in search_paths:
                if os.path.exists(search_dir):
                    fbx_files = glob.glob(os.path.join(search_dir, "*.fbx"))
                    if fbx_files:
                        fbx_path = fbx_files[0]
                        break

        if not fbx_path or not os.path.exists(fbx_path):
            raise RuntimeError(f"Skinning output FBX not found. Searched: {possible_paths}")

        print(f"[UniRigApplySkinningMLNew] Found output FBX: {fbx_path}")
        print(f"[UniRigApplySkinningMLNew] FBX file size: {os.path.getsize(fbx_path)} bytes")

        # Create rigged mesh dict
        rigged_mesh = {
            "fbx_path": fbx_path,
            "has_skinning": True,
            "has_skeleton": True,
        }

        # Create texture preview output
        texture_preview = None
        if skeleton.get('texture_data_base64'):
            texture_preview, tex_w, tex_h = decode_texture_to_comfy_image(skeleton['texture_data_base64'])
            if texture_preview is not None:
                print(f"[UniRigApplySkinningMLNew] Texture preview created: {tex_w}x{tex_h}")
            else:
                print(f"[UniRigApplySkinningMLNew] Warning: Could not decode texture for preview")
                texture_preview = create_placeholder_texture()
        else:
            print(f"[UniRigApplySkinningMLNew] No texture available for preview")
            texture_preview = create_placeholder_texture()

        print(f"[UniRigApplySkinningMLNew] Skinning application complete!")

        return (rigged_mesh, texture_preview)
