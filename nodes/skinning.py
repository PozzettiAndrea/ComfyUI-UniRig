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
import folder_paths

# Support both relative imports (ComfyUI) and absolute imports (testing)
try:
    from ..constants import BLENDER_TIMEOUT, INFERENCE_TIMEOUT
    from ..lib.skinning_config import SkinningConfig
except ImportError:
    from constants import BLENDER_TIMEOUT, INFERENCE_TIMEOUT
    from lib.skinning_config import SkinningConfig

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


# Global cache for model_cache module to ensure same instance is used
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
                _MODEL_CACHE_MODULE = False
    return _MODEL_CACHE_MODULE if _MODEL_CACHE_MODULE else None


class UniRigApplySkinningML:
    """
    Apply skinning weights using ML.
    Takes skeleton dict and mesh, prepares data and runs ML inference.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "normalized_mesh": ("TRIMESH",),
                "skeleton": ("SKELETON",),
                "skinning_model": ("UNIRIG_SKINNING_MODEL", {
                    "tooltip": "Pre-loaded skinning model (from UniRigLoadSkinningModel). Required for inference."
                }),
            },
            "optional": {
                "voxel_grid_size": ("INT", {
                    "default": 256,
                    "min": 64,
                    "max": 512,
                    "step": 64,
                    "tooltip": "Voxel grid resolution for spatial weight distribution. Higher = better quality, more VRAM. Default: 256 (was 196)"
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
                "grid_size": ("FLOAT", {
                    "default": 0.005,
                    "min": 0.001,
                    "max": 0.01,
                    "step": 0.001,
                    "tooltip": "Grid size for point cloud spatial features. Lower = finer detail, more compute. Default: 0.005"
                }),
                "voxel_mask_power": ("FLOAT", {
                    "default": 3.0,
                    "min": 1.0,
                    "max": 5.0,
                    "step": 0.5,
                    "tooltip": "Power for voxel mask weight sharpness. Lower = smoother transitions. Default: 3.0"
                }),
                "num_train_vertex": ("INT", {
                    "default": 512,
                    "min": 256,
                    "max": 2048,
                    "step": 256,
                    "tooltip": "Vertices processed per batch. Higher = faster but more VRAM. Default: 512 (Note: Requires model reload to take effect)"
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Force re-execution when quality parameters change
        return float("nan")

    RETURN_TYPES = ("RIGGED_MESH", "IMAGE")
    RETURN_NAMES = ("rigged_mesh", "texture_preview")
    FUNCTION = "apply_skinning"
    CATEGORY = "UniRig"

    def apply_skinning(self, normalized_mesh, skeleton, skinning_model,
                        voxel_grid_size=None, num_samples=None, vertex_samples=None,
                        grid_size=None, voxel_mask_power=None, num_train_vertex=None):
        start_time = time.time()

        # Create skinning config with user overrides
        config_kwargs = {}
        if voxel_grid_size is not None:
            config_kwargs['voxel_grid_size'] = voxel_grid_size
        if num_samples is not None:
            config_kwargs['num_samples'] = num_samples
        if vertex_samples is not None:
            config_kwargs['vertex_samples'] = vertex_samples
        if grid_size is not None:
            config_kwargs['grid_size'] = grid_size
        if voxel_mask_power is not None:
            config_kwargs['voxel_mask_power'] = voxel_mask_power
        if num_train_vertex is not None:
            config_kwargs['num_train_vertex'] = num_train_vertex

        skinning_config = SkinningConfig(**config_kwargs)

        print(f"[UniRigApplySkinningML] Starting ({len(normalized_mesh.vertices)} verts, {len(skeleton['names'])} bones)")
        print(f"[UniRigApplySkinningML] Config: grid={skinning_config.voxel_grid_size}, samples={skinning_config.num_samples}, vertex_samples={skinning_config.vertex_samples}")

        # Note: grid_size, voxel_mask_power, and num_train_vertex are model-level parameters
        # that require model reload to take effect (not runtime configurable yet)
        if grid_size is not None and grid_size != 0.005:
            print(f"[UniRigApplySkinningML] Note: grid_size={grid_size} is a model-level parameter (requires model reload)")
        if voxel_mask_power is not None and voxel_mask_power != 3.0:
            print(f"[UniRigApplySkinningML] Note: voxel_mask_power={voxel_mask_power} is a model-level parameter (requires model reload)")
        if num_train_vertex is not None and num_train_vertex != 512:
            print(f"[UniRigApplySkinningML] Note: num_train_vertex={num_train_vertex} is a model-level parameter (requires model reload)")

        # Validate skinning_model
        if skinning_model is None:
            raise ValueError("skinning_model is required. Use UniRigLoadSkinningModel to load the model first.")

        task_config_path = skinning_model.get("task_config_path")
        if not task_config_path:
            raise ValueError("skinning_model missing 'task_config_path'. Model may be corrupted.")

        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix="unirig_skinning_")
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
        else:
            save_data['texture_data_base64'] = ""
            save_data['texture_format'] = ""
            save_data['texture_width'] = 0
            save_data['texture_height'] = 0
            save_data['material_name'] = skeleton.get('material_name', '')

        np.savez(predict_skeleton_path, **save_data)

        # Export mesh to GLB
        input_glb = os.path.join(temp_dir, "input.glb")
        normalized_mesh.export(input_glb)

        # Run skinning inference
        output_fbx = os.path.join(temp_dir, "rigged.fbx")

        # Check if we can use the in-process cached model
        model_cache_key = skinning_model.get("model_cache_key")
        cache_to_gpu = skinning_model.get("cache_to_gpu", True)
        use_cache = False

        if model_cache_key and cache_to_gpu:
            model_cache = _get_model_cache()
            if model_cache and model_cache.is_model_loaded(model_cache_key):
                use_cache = True

        if use_cache:
            # Use in-process cached model for inference
            print(f"[UniRigApplySkinningML] Running inference (cached)")
            try:
                response = model_cache.run_inference(
                    cache_key=model_cache_key,
                    request_data={
                        "input": input_glb,
                        "output": output_fbx,
                        "npz_dir": temp_dir,
                        "seed": 123,
                        "config_overrides": skinning_config.to_dict(),
                    }
                )

                if response.get("error"):
                    if response.get("traceback"):
                        print(response.get("traceback"))
                    raise RuntimeError(f"Cached model inference failed: {response.get('error')}")

            except Exception as e:
                raise
        else:
            # Fallback to subprocess
            print(f"[UniRigApplySkinningML] Running inference (subprocess)")

            python_exe = sys.executable
            run_script = os.path.join(UNIRIG_PATH, "run.py")

            cmd = [
                python_exe, run_script,
                "--task", task_config_path,
                "--input", input_glb,
                "--output", output_fbx,
                "--npz_dir", temp_dir,
                "--seed", "123"
            ]

            # Set BLENDER_EXE environment variable for FBX export
            env = os.environ.copy()
            if BLENDER_EXE:
                env['BLENDER_EXE'] = BLENDER_EXE

            # Pass GPU caching preference to subprocess
            env["UNIRIG_CACHE_TO_GPU"] = "1" if cache_to_gpu else "0"

            # Pass config overrides via environment variable as JSON
            import json
            env["UNIRIG_CONFIG_OVERRIDES"] = json.dumps(skinning_config.to_dict())

            result = subprocess.run(
                cmd,
                cwd=UNIRIG_PATH,
                capture_output=True,
                text=True,
                timeout=INFERENCE_TIMEOUT,
                env=env
            )

            if result.returncode != 0:
                # Show full output on failure
                if result.stdout:
                    print(f"[UniRigApplySkinningML] Skinning stdout:\n{result.stdout}")
                if result.stderr:
                    print(f"[UniRigApplySkinningML] Skinning stderr:\n{result.stderr}")
                raise RuntimeError(f"Skinning generation failed with exit code {result.returncode}")
            else:
                # On success, only show actual errors from stderr (not warnings)
                if result.stderr:
                    stderr_lines = result.stderr.split('\n')
                    error_lines = [l for l in stderr_lines if 'error' in l.lower() and 'warning' not in l.lower()]
                    if error_lines:
                        print(f"[UniRigApplySkinningML] Errors:\n" + '\n'.join(error_lines[:5]))

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
            if texture_preview is None:
                texture_preview = create_placeholder_texture()
        else:
            texture_preview = create_placeholder_texture()

        total_time = time.time() - start_time
        print(f"[UniRigApplySkinningML] Complete! FBX: {os.path.getsize(fbx_path)//1024}KB, {total_time:.2f}s total")

        return (rigged_mesh, texture_preview)
