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
                "fbx_name": ("STRING", {
                    "default": "",
                    "tooltip": "Custom filename for saved FBX (without extension). If empty, uses rigged_<timestamp>.fbx"
                }),
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

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("fbx_output_path", "texture_preview")
    FUNCTION = "apply_skinning"
    CATEGORY = "UniRig"

    def apply_skinning(self, normalized_mesh, skeleton, skinning_model,
                       fbx_name=None, voxel_grid_size=None, num_samples=None, vertex_samples=None,
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
            # Check if there was an import error that caused cache loading to fail
            try:
                from .model_loaders import get_model_cache_error
            except ImportError:
                from model_loaders import get_model_cache_error

            import_error = get_model_cache_error()

            if import_error:
                error_str = str(import_error).lower()
                if "spconv" in error_str:
                    raise RuntimeError(
                        "UniRig model loading failed: spconv is not installed.\n\n"
                        "spconv is required for GPU-accelerated point cloud processing.\n\n"
                        "Install with:\n"
                        "  pip install spconv-cu118  # For CUDA 11.8\n"
                        "  pip install spconv-cu120  # For CUDA 12.0\n"
                        "  pip install spconv-cu121  # For CUDA 12.1\n\n"
                        "Choose the version matching your CUDA installation.\n"
                        "Check CUDA version with: nvidia-smi"
                    )
                elif "torch_scatter" in error_str:
                    raise RuntimeError(
                        "UniRig model loading failed: torch-scatter is not installed.\n\n"
                        "torch-scatter is required for efficient scatter operations.\n\n"
                        "Install with:\n"
                        "  pip install torch-scatter\n\n"
                        "Note: If no wheel is available for your PyTorch version,\n"
                        "you may need to build from source:\n"
                        "  pip install git+https://github.com/rusty1s/pytorch_scatter.git"
                    )
                elif "torch_cluster" in error_str:
                    raise RuntimeError(
                        "UniRig model loading failed: torch-cluster is not installed.\n\n"
                        "Install with:\n"
                        "  pip install torch-cluster\n\n"
                        "Note: If no wheel is available for your PyTorch version,\n"
                        "you may need to build from source:\n"
                        "  pip install git+https://github.com/rusty1s/pytorch_cluster.git"
                    )
                else:
                    raise RuntimeError(
                        f"UniRig model loading failed due to import error:\n{import_error}\n\n"
                        "Check that all required dependencies are installed.\n"
                        "Run: pip install -r requirements.txt"
                    )
            else:
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

        # Export mesh to GLB
        input_glb = os.path.join(temp_dir, "input.glb")

        normalized_mesh.export(input_glb)
        print(f"[UniRigApplySkinningMLNew] Exported mesh: {normalized_mesh.vertices.shape[0]} vertices, {normalized_mesh.faces.shape[0]} faces")

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

        # Auto-save FBX to output directory
        output_dir = folder_paths.get_output_directory()

        # Determine output filename
        if fbx_name and fbx_name.strip():
            # Use custom name from user
            output_filename = fbx_name.strip()
            # Ensure .fbx extension
            if not output_filename.lower().endswith('.fbx'):
                output_filename = output_filename + '.fbx'
        else:
            # Use auto-generated name with timestamp
            output_filename = f"rigged_{int(time.time())}.fbx"

        output_path = os.path.join(output_dir, output_filename)
        shutil.copy(fbx_path, output_path)

        print(f"[UniRigApplySkinningMLNew] Auto-saved FBX to output: {output_filename}")
        print(f"[UniRigApplySkinningMLNew] Full path: {output_path}")

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

        # Clean up temporary directory
        # Windows-specific: Force garbage collection and retry logic to release file handles
        # This helps prevent "file in use by another process" errors during cleanup
        if sys.platform == 'win32':
            import gc
            gc.collect()
            # Give Windows a moment to release file handles
            time.sleep(0.1)

        try:
            # First attempt with ignore_errors
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"[UniRigApplySkinningMLNew] Cleaned up temp directory")
        except Exception as e:
            # Don't fail the whole operation if cleanup fails
            print(f"[UniRigApplySkinningMLNew] Warning: Could not clean up temp directory: {e}")

        # Windows: If directory still exists, schedule for deletion on restart
        if sys.platform == 'win32' and os.path.exists(temp_dir):
            print(f"[UniRigApplySkinningMLNew] Note: Temp directory will be cleaned on next restart: {temp_dir}")

        return (output_filename, texture_preview)
