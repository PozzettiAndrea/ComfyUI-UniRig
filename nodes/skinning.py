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
    from ..constants import BLENDER_TIMEOUT, INFERENCE_TIMEOUT, PARSE_TIMEOUT
except ImportError:
    from constants import BLENDER_TIMEOUT, INFERENCE_TIMEOUT, PARSE_TIMEOUT

try:
    from .base import (
        UNIRIG_PATH,
        BLENDER_EXE,
        BLENDER_PARSE_SKELETON,
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
        BLENDER_PARSE_SKELETON,
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


class UniRigApplySkinning:
    """
    Apply skinning weights to a mesh using an extracted skeleton.

    This node takes a skeleton (from UniRigExtractSkeleton) and applies
    skinning weights to create a rigged mesh ready for animation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "skeleton": ("SKELETON",),
            }
        }

    RETURN_TYPES = ("RIGGED_MESH",)
    RETURN_NAMES = ("rigged_mesh",)
    FUNCTION = "apply_skinning"
    CATEGORY = "UniRig"

    def apply_skinning(self, trimesh, skeleton):
        """Apply skinning weights to mesh using skeleton."""
        total_start = time.time()
        print(f"[UniRigApplySkinning] Starting skinning application...")

        if not BLENDER_EXE or not os.path.exists(BLENDER_EXE):
            raise RuntimeError(f"Blender not found. Please run install_blender.py or install manually.")

        if not os.path.exists(UNIRIG_PATH):
            raise RuntimeError(f"UniRig not found at {UNIRIG_PATH}")

        skeleton_npz_path = skeleton.get("npz_path")
        if not skeleton_npz_path or not os.path.exists(skeleton_npz_path):
            raise RuntimeError(f"Skeleton NPZ not found: {skeleton_npz_path}. Make sure skeleton was extracted with UniRigExtractSkeleton.")

        print(f"[UniRigApplySkinning] Using skeleton NPZ: {skeleton_npz_path}")

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.glb")
            output_path = os.path.join(tmpdir, "result_fbx.fbx")

            # Export mesh to GLB
            step_start = time.time()
            print(f"[UniRigApplySkinning] Exporting mesh: {len(trimesh.vertices)} vertices, {len(trimesh.faces)} faces")
            trimesh.export(input_path)
            export_time = time.time() - step_start
            print(f"[UniRigApplySkinning] Mesh exported in {export_time:.2f}s")

            # Denormalize skeleton and save to expected location
            input_subdir = os.path.join(tmpdir, "input")
            os.makedirs(input_subdir, exist_ok=True)
            predict_skeleton_path = os.path.join(input_subdir, "predict_skeleton.npz")

            # Load normalized skeleton
            print(f"[UniRigApplySkinning] Loading and denormalizing skeleton...")
            skeleton_data = np.load(skeleton_npz_path, allow_pickle=True)

            mesh_center = skeleton.get("mesh_center")
            mesh_scale = skeleton.get("mesh_scale")

            if mesh_center is None or mesh_scale is None:
                print(f"[UniRigApplySkinning] WARNING: Missing mesh bounds in skeleton, using normalized coordinates")
                shutil.copy(skeleton_npz_path, predict_skeleton_path)
            else:
                # Denormalize joint positions
                joints_normalized = skeleton_data['joints']
                joints_denormalized = joints_normalized * mesh_scale + mesh_center

                # Denormalize tail positions
                tails_normalized = skeleton_data['tails']
                tails_denormalized = tails_normalized * mesh_scale + mesh_center

                print(f"[UniRigApplySkinning] Denormalization:")
                print(f"  Mesh center: {mesh_center}")
                print(f"  Mesh scale: {mesh_scale}")
                print(f"  Joint extents before: {joints_normalized.min(axis=0)} to {joints_normalized.max(axis=0)}")
                print(f"  Joint extents after: {joints_denormalized.min(axis=0)} to {joints_denormalized.max(axis=0)}")

                # Save denormalized skeleton
                save_data = {
                    'bone_names': skeleton_data['names'],
                    'bone_parents': skeleton_data['parents'],
                    'bone_to_head_vertex': joints_denormalized,
                    'tails': tails_denormalized,
                }

                # Copy optional fields
                if 'matrix_local' in skeleton_data:
                    save_data['matrix_local'] = skeleton_data['matrix_local']
                if 'path' in skeleton_data:
                    save_data['path'] = skeleton_data['path']
                if 'cls' in skeleton_data:
                    save_data['cls'] = skeleton_data['cls']

                np.savez(predict_skeleton_path, **save_data)

            print(f"[UniRigApplySkinning] Saved denormalized skeleton to: {predict_skeleton_path}")

            # Run skinning inference
            step_start = time.time()
            print(f"[UniRigApplySkinning] Applying skinning weights...")
            skin_cmd = [
                sys.executable, os.path.join(UNIRIG_PATH, "run.py"),
                "--task", os.path.join(UNIRIG_PATH, "configs/task/quick_inference_unirig_skin.yaml"),
                "--input", input_path,
                "--output", output_path,
                "--npz_dir", tmpdir,
            ]

            env = setup_subprocess_env()

            try:
                result = subprocess.run(skin_cmd, cwd=UNIRIG_PATH, env=env, capture_output=True, text=True, timeout=INFERENCE_TIMEOUT)
                if result.stdout:
                    print(f"[UniRigApplySkinning] Skinning stdout:\n{result.stdout}")
                if result.stderr:
                    print(f"[UniRigApplySkinning] Skinning stderr:\n{result.stderr}")

                if result.returncode != 0:
                    print(f"[UniRigApplySkinning] Skinning failed with return code: {result.returncode}")
                    raise RuntimeError(f"Skinning generation failed with exit code {result.returncode}")

                print(f"[UniRigApplySkinning] Skinning inference completed successfully")

                # Look for the output FBX
                print(f"[UniRigApplySkinning] Looking for FBX output at: {output_path}")
                if not os.path.exists(output_path):
                    print(f"[UniRigApplySkinning] FBX not found at primary location")
                    print(f"[UniRigApplySkinning] Searching alternative paths...")

                    # List all files in tmpdir for debugging
                    print(f"[UniRigApplySkinning] Contents of {tmpdir}:")
                    for root, dirs, files in os.walk(tmpdir):
                        level = root.replace(tmpdir, '').count(os.sep)
                        indent = ' ' * 2 * level
                        print(f"{indent}{os.path.basename(root)}/")
                        subindent = ' ' * 2 * (level + 1)
                        for file in files:
                            file_path = os.path.join(root, file)
                            file_size = os.path.getsize(file_path)
                            print(f"{subindent}{file} ({file_size} bytes)")

                    alt_paths = [
                        os.path.join(tmpdir, "results", "result_fbx.fbx"),
                        os.path.join(tmpdir, "input", "result_fbx.fbx"),
                        os.path.join(tmpdir, "results", "input", "result_fbx.fbx"),
                    ]

                    found = False
                    for alt_path in alt_paths:
                        print(f"[UniRigApplySkinning] Checking: {alt_path}")
                        if os.path.exists(alt_path):
                            print(f"[UniRigApplySkinning] Found FBX at: {alt_path}")
                            shutil.copy(alt_path, output_path)
                            found = True
                            break

                    if not found:
                        print(f"[UniRigApplySkinning] FBX not found in any expected location")
                        raise RuntimeError(f"Skinned FBX not found: {output_path}")
                else:
                    print(f"[UniRigApplySkinning] Found FBX at primary location: {output_path}")

                skinning_time = time.time() - step_start
                print(f"[UniRigApplySkinning] Skinning applied in {skinning_time:.2f}s")

            except subprocess.TimeoutExpired:
                raise RuntimeError(f"Skinning generation timed out (>{INFERENCE_TIMEOUT}s)")
            except Exception as e:
                print(f"[UniRigApplySkinning] Skinning error: {e}")
                raise

            # Load the rigged mesh
            print(f"[UniRigApplySkinning] Loading rigged mesh from {output_path}...")

            rigged_mesh = {
                "mesh": trimesh,
                "fbx_path": output_path,
                "has_skinning": True,
                "has_skeleton": True,
            }

            # Copy to a persistent location
            persistent_fbx = os.path.join(folder_paths.get_temp_directory(), f"rigged_mesh_skinning_{int(time.time())}.fbx")
            shutil.copy(output_path, persistent_fbx)
            rigged_mesh["fbx_path"] = persistent_fbx

            total_time = time.time() - total_start
            print(f"[UniRigApplySkinning] Skinning application complete!")
            print(f"[UniRigApplySkinning] TOTAL TIME: {total_time:.2f}s")

            return (rigged_mesh,)


class UniRigApplySkinningMLNew:
    """
    Apply skinning weights using ML - New implementation with model caching.
    Takes skeleton dict and mesh, prepares data and runs ML inference with in-process model caching.

    Note: For the original simple subprocess approach, use UniRigApplySkinningML (without 'New' suffix).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "skeleton_npz_path": ("STRING", {
                    "tooltip": "Path to skeleton.npz file from Extract Skeleton node"
                }),
            },
            "optional": {
                "skinning_model": ("UNIRIG_SKINNING_MODEL", {
                    "tooltip": "Pre-loaded skinning model (from UniRigLoadSkinningModel)"
                }),
                "use_subprocess": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use subprocess mode instead of cached model (slower but may give better quality)"
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
    RETURN_NAMES = ("rigged_fbx_path", "texture_preview")
    FUNCTION = "apply_skinning"
    CATEGORY = "UniRig"

    def apply_skinning(self, skeleton_npz_path, skinning_model=None, use_subprocess=False,
                       voxel_grid_size=None, num_samples=None, vertex_samples=None,
                       voxel_mask_power=None):
        print(f"[UniRigApplySkinningML] Starting ML skinning...")
        print(f"[UniRigApplySkinningML] Using skeleton NPZ: {skeleton_npz_path}")

        # Verify file exists
        if not os.path.exists(skeleton_npz_path):
            raise RuntimeError(f"Skeleton NPZ not found: {skeleton_npz_path}")

        # Use pre-loaded model if available
        if skinning_model is not None:
            print(f"[UniRigApplySkinningML] Using pre-loaded model configuration")
            if skinning_model.get("cached", False):
                print(f"[UniRigApplySkinningML] Model weights already downloaded and cached")
            task_config_path = skinning_model.get("task_config_path")
        else:
            task_config_path = os.path.join(UNIRIG_PATH, "configs", "task", "quick_inference_unirig_skin.yaml")

        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix="unirig_skinning_")
        predict_skeleton_dir = os.path.join(temp_dir, "input")
        os.makedirs(predict_skeleton_dir, exist_ok=True)

        # Load skeleton data from NPZ
        print(f"[UniRigApplySkinningML] Loading skeleton data from NPZ...")
        skeleton_data = np.load(skeleton_npz_path, allow_pickle=True)

        # Copy skeleton NPZ to expected location (predict_skeleton.npz)
        predict_skeleton_path = os.path.join(predict_skeleton_dir, "predict_skeleton.npz")
        shutil.copy2(skeleton_npz_path, predict_skeleton_path)
        print(f"[UniRigApplySkinningML] Copied skeleton NPZ to: {predict_skeleton_path}")

        # Extract mesh from NPZ and export to GLB
        joints = skeleton_data['joints']
        vertices = skeleton_data.get('vertices', [])
        faces = skeleton_data.get('faces', [])

        if len(vertices) == 0 or len(faces) == 0:
            raise RuntimeError(f"Skeleton NPZ contains no mesh data (vertices: {len(vertices)}, faces: {len(faces)})")

        # Export mesh to input.glb for skinning model
        from trimesh import Trimesh
        input_glb = os.path.join(temp_dir, "input.glb")

        # Get vertex normals if available
        vertex_normals = skeleton_data.get('vertex_normals')

        mesh = Trimesh(
            vertices=vertices,
            faces=faces,
            vertex_normals=vertex_normals,
            process=False
        )
        mesh.export(input_glb)
        print(f"[UniRigApplySkinningML] Exported mesh from NPZ to GLB: {input_glb}")
        print(f"[UniRigApplySkinningML] Skeleton: {len(joints)} joints, Mesh: {len(vertices)} vertices, {len(faces)} faces")

        # Log coordinate information from NPZ
        if len(joints) > 0:
            joints_min = joints.min(axis=0)
            joints_max = joints.max(axis=0)
            joints_center = (joints_min + joints_max) / 2
            print(f"[UniRigApplySkinningML] Skeleton joints from NPZ:")
            print(f"  First joint: ({joints[0][0]:.2f}, {joints[0][1]:.2f}, {joints[0][2]:.2f})")
            print(f"  Bounds: min({joints_min[0]:.2f}, {joints_min[1]:.2f}, {joints_min[2]:.2f}) max({joints_max[0]:.2f}, {joints_max[1]:.2f}, {joints_max[2]:.2f})")
            print(f"  Center: ({joints_center[0]:.2f}, {joints_center[1]:.2f}, {joints_center[2]:.2f})")

        if len(vertices) > 0:
            vertices_min = vertices.min(axis=0)
            vertices_max = vertices.max(axis=0)
            vertices_center = (vertices_min + vertices_max) / 2
            print(f"[UniRigApplySkinningML] Mesh vertices from NPZ:")
            print(f"  First vertex: ({vertices[0][0]:.2f}, {vertices[0][1]:.2f}, {vertices[0][2]:.2f})")
            print(f"  Bounds: min({vertices_min[0]:.2f}, {vertices_min[1]:.2f}, {vertices_min[2]:.2f}) max({vertices_max[0]:.2f}, {vertices_max[1]:.2f}, {vertices_max[2]:.2f})")
            print(f"  Center: ({vertices_center[0]:.2f}, {vertices_center[1]:.2f}, {vertices_center[2]:.2f})")

            # Comparison if both exist
            if len(joints) > 0:
                center_dist = np.linalg.norm(vertices_center - joints_center)
                vertices_size_scalar = np.linalg.norm(vertices_max - vertices_min)
                joints_size_scalar = np.linalg.norm(joints_max - joints_min)
                size_ratio = vertices_size_scalar / joints_size_scalar
                print(f"[UniRigApplySkinningML] === COORDINATE COMPARISON ===")
                print(f"  Center distance: {center_dist:.4f} units")
                print(f"  Size ratio (mesh/skeleton): {size_ratio:.3f}")
                if center_dist > 1.0:
                    print(f"  WARNING: Large center distance detected!")
                else:
                    print(f"  ✓ Mesh and skeleton centers are well aligned")

        # Run skinning inference
        step_start = time.time()
        print(f"[UniRigApplySkinningML] Running skinning inference...")

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
            print(f"[UniRigApplySkinningML] Config overrides: {config_overrides}")

        # Choose between subprocess and in-process cached model
        if use_subprocess:
            # Use subprocess mode for potentially better quality
            print(f"[UniRigApplySkinningML] Running subprocess inference (fresh process)...")

            if skinning_model is None:
                raise RuntimeError("Skinning model not loaded - use UniRigLoadSkinningModel node first")

            checkpoint = skinning_model.get("checkpoint_name")
            if not checkpoint:
                raise RuntimeError("No checkpoint name found in skinning model")

            # Build subprocess command
            subprocess_script = os.path.join(LIB_DIR, "run_skinning_subprocess.py")
            env = setup_subprocess_env()

            cmd = [
                sys.executable,
                subprocess_script,
                '--input', input_glb,
                '--output', output_fbx,
                '--npz_dir', temp_dir,
                '--data_name', 'predict_skeleton.npz',
                '--seed', '123',
                '--checkpoint', checkpoint,
            ]

            # Add config overrides as command line args
            if 'voxel_grid_size' in config_overrides:
                cmd.extend(['--voxel_grid_size', str(config_overrides['voxel_grid_size'])])
            if 'num_samples' in config_overrides:
                cmd.extend(['--num_samples', str(config_overrides['num_samples'])])
            if 'vertex_samples' in config_overrides:
                cmd.extend(['--vertex_samples', str(config_overrides['vertex_samples'])])
            if 'voxel_mask_power' in config_overrides:
                cmd.extend(['--voxel_mask_power', str(config_overrides['voxel_mask_power'])])

            print(f"[UniRigApplySkinningML] Subprocess command: {' '.join(cmd[:6])}...")

            # Run subprocess
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                cwd=os.path.dirname(subprocess_script)
            )

            if result.returncode != 0:
                error_msg = f"Subprocess skinning failed (exit code {result.returncode})"
                if result.stderr:
                    error_msg += f"\n\nStderr:\n{result.stderr}"
                if result.stdout:
                    error_msg += f"\n\nStdout:\n{result.stdout}"
                raise RuntimeError(error_msg)

            if result.stdout:
                print(f"[UniRigApplySkinningML] Subprocess output:")
                for line in result.stdout.split('\n'):
                    if line.strip():
                        print(f"  {line}")

            inference_time = time.time() - step_start
            print(f"[UniRigApplySkinningML] ✓ Subprocess inference completed in {inference_time:.2f}s")

        else:
            # Use in-process cached model (faster but may have quality differences)
            if skinning_model is None or not skinning_model.get("model_cache_key"):
                raise RuntimeError("Skinning model not loaded - use UniRigLoadSkinningModel node first")

            model_cache = _get_model_cache()
            if not model_cache:
                raise RuntimeError("Model cache not available - GPU caching must be enabled")

            cache_key = skinning_model["model_cache_key"]
            print(f"[UniRigApplySkinningML] Running cached inference (in-process)...")

            request_data = {
                "seed": 123,
                "input": input_glb,
                "output": output_fbx,
                "npz_dir": temp_dir,
                "cls": None,  # No cls data when loading from FBX
                "data_name": "predict_skeleton.npz",
                "config_overrides": config_overrides,
            }

            result = model_cache.run_inference(cache_key, request_data)
            if "error" in result:
                error_msg = f"Skinning inference failed: {result['error']}"
                if "traceback" in result:
                    error_msg += f"\n\nTraceback:\n{result['traceback']}"
                raise RuntimeError(error_msg)

            inference_time = time.time() - step_start
            print(f"[UniRigApplySkinningML] ✓ Inference completed in {inference_time:.2f}s")

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

        print(f"[UniRigApplySkinningML] Found output FBX: {fbx_path}")
        print(f"[UniRigApplySkinningML] FBX file size: {os.path.getsize(fbx_path)} bytes")

        # Create texture preview output (placeholder for now - texture data not in FBX)
        print(f"[UniRigApplySkinningML] Creating placeholder texture preview")
        texture_preview = create_placeholder_texture()

        print(f"[UniRigApplySkinningMLNew] Skinning application complete!")

        return (fbx_path, texture_preview)


class UniRigApplySkinningML:
    """
    Apply skinning weights using ML - Original implementation.

    Simple subprocess-based approach for skinning weight prediction.
    Uses direct calls to run.py for reliable inference.
    Takes skeleton dict and mesh, prepares data and runs ML inference via subprocess.

    This is the original working implementation that uses subprocess for each inference.
    For faster inference with model caching, use UniRigApplySkinningMLNew.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "normalized_mesh": ("TRIMESH", {
                    "tooltip": "Normalized mesh from skeleton extraction"
                }),
                "skeleton": ("SKELETON", {
                    "tooltip": "Skeleton dict from UniRigExtractSkeleton"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("rigged_fbx_path", "texture_preview")
    FUNCTION = "apply_skinning"
    CATEGORY = "UniRig"

    def apply_skinning(self, normalized_mesh, skeleton):
        """Apply skinning weights via subprocess (original implementation)."""
        total_start = time.time()
        print(f"[UniRigApplySkinningML] Starting ML skinning (original implementation)...")

        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix="unirig_skinning_")
        predict_skeleton_dir = os.path.join(temp_dir, "input")
        os.makedirs(predict_skeleton_dir, exist_ok=True)

        try:
            # Prepare skeleton NPZ from dict
            predict_skeleton_path = os.path.join(predict_skeleton_dir, "predict_skeleton.npz")

            # Ensure bones are ordered with root first
            joints = skeleton['joints']
            names = skeleton['names']
            parents = skeleton['parents']

            # Debug: Print parent values
            print(f"[UniRigApplySkinningML] Parent array: {parents[:10]}... (showing first 10)")
            print(f"[UniRigApplySkinningML] Parent array dtype: {parents.dtype}")
            print(f"[UniRigApplySkinningML] First parent value: {parents[0]} (type: {type(parents[0])})")

            # Find root bone (parent = -1 or None)
            root_idx = None
            for i, parent in enumerate(parents):
                # Check for None, -1, or any negative value
                if parent is None or (hasattr(parent, '__len__') and len(parent) == 0) or (isinstance(parent, (int, np.integer)) and parent < 0):
                    root_idx = i
                    print(f"[UniRigApplySkinningML] Found root bone at index {i}: name='{names[i]}', parent={parent}")
                    break

            if root_idx is None:
                print(f"[UniRigApplySkinningML] WARNING: No root bone found (parent=-1), using first bone")
                root_idx = 0
            elif root_idx != 0:
                print(f"[UniRigApplySkinningML] Reordering bones to put root bone first (was at index {root_idx})")
                # Create a mapping of old index -> new index
                old_to_new = {}
                old_to_new[root_idx] = 0
                new_idx = 1
                for i in range(len(names)):
                    if i != root_idx:
                        old_to_new[i] = new_idx
                        new_idx += 1

                # Reorder arrays
                new_joints = np.zeros_like(joints)
                new_names = np.empty_like(names)
                new_parents = np.zeros_like(parents)

                for old_idx, new_idx in old_to_new.items():
                    new_joints[new_idx] = joints[old_idx]
                    new_names[new_idx] = names[old_idx]
                    # Update parent indices
                    old_parent = parents[old_idx]
                    if old_parent < 0:
                        new_parents[new_idx] = -1
                    else:
                        new_parents[new_idx] = old_to_new[old_parent]

                joints = new_joints
                names = new_names
                parents = new_parents
                print(f"[UniRigApplySkinningML] Root bone is now: {names[0]}")

                # Also reorder tails if present
                if skeleton.get('tails') is not None:
                    old_tails = skeleton['tails']
                    new_tails = np.zeros_like(old_tails)
                    for old_idx, new_idx in old_to_new.items():
                        new_tails[new_idx] = old_tails[old_idx]
                    tails = new_tails
                else:
                    tails = None
            else:
                tails = skeleton.get('tails')

            # Convert parent values: UniRig expects None for root bone, not -1
            # Replace -1 with None in parents array
            parents_for_save = []
            for p in parents:
                if isinstance(p, (int, np.integer)) and p < 0:
                    parents_for_save.append(None)
                else:
                    parents_for_save.append(int(p) if isinstance(p, (int, np.integer)) else p)

            print(f"[UniRigApplySkinningML] Converted parents: {parents_for_save[:10]}... (first 10)")

            save_data = {
                'joints': joints,
                'names': names,
                'parents': np.array(parents_for_save, dtype=object),  # Use object dtype to allow None
                'tails': tails,
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
                print(f"[UniRigApplySkinningML] UV data included: {len(skeleton['uv_coords'])} UVs")
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
                print(f"[UniRigApplySkinningML] Texture data included: {skeleton['texture_width']}x{skeleton['texture_height']} {skeleton['texture_format']}")
            else:
                save_data['texture_data_base64'] = ""
                save_data['texture_format'] = ""
                save_data['texture_width'] = 0
                save_data['texture_height'] = 0
                save_data['material_name'] = skeleton.get('material_name', '')

            np.savez(predict_skeleton_path, **save_data)
            print(f"[UniRigApplySkinningML] Prepared skeleton NPZ: {predict_skeleton_path}")

            # Export mesh to GLB
            input_glb = os.path.join(temp_dir, "input.glb")
            normalized_mesh.export(input_glb)
            print(f"[UniRigApplySkinningML] Exported mesh: {normalized_mesh.vertices.shape[0]} vertices, {normalized_mesh.faces.shape[0]} faces")

            # Run skinning inference via subprocess
            step_start = time.time()
            print(f"[UniRigApplySkinningML] Running skinning inference via subprocess...")

            python_exe = sys.executable
            run_script = os.path.join(UNIRIG_PATH, "run.py")
            config_path = os.path.join(UNIRIG_PATH, "configs", "task", "quick_inference_unirig_skin.yaml")
            output_fbx = os.path.join(temp_dir, "rigged.fbx")

            cmd = [
                python_exe, run_script,
                "--task", config_path,
                "--input", input_glb,
                "--output", output_fbx,
                "--npz_dir", temp_dir,
                "--seed", "123"
            ]

            print(f"[UniRigApplySkinningML] Running: {' '.join(cmd)}")

            # Set BLENDER_EXE environment variable for FBX export
            env = setup_subprocess_env()

            result = subprocess.run(
                cmd,
                cwd=UNIRIG_PATH,
                capture_output=True,
                text=True,
                timeout=INFERENCE_TIMEOUT,
                env=env
            )

            if result.stdout:
                print(f"[UniRigApplySkinningML] Skinning stdout:\n{result.stdout}")
            if result.stderr:
                print(f"[UniRigApplySkinningML] Skinning stderr:\n{result.stderr}")

            if result.returncode != 0:
                print(f"[UniRigApplySkinningML] Skinning failed with return code: {result.returncode}")
                raise RuntimeError(f"Skinning generation failed with exit code {result.returncode}")

            inference_time = time.time() - step_start
            print(f"[UniRigApplySkinningML] Skinning completed in {inference_time:.2f}s")

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

            print(f"[UniRigApplySkinningML] Found output FBX: {fbx_path}")
            print(f"[UniRigApplySkinningML] FBX file size: {os.path.getsize(fbx_path)} bytes")

            # Copy to persistent location
            persistent_fbx = os.path.join(folder_paths.get_temp_directory(), f"rigged_{int(time.time())}.fbx")
            shutil.copy(fbx_path, persistent_fbx)
            print(f"[UniRigApplySkinningML] Saved persistent FBX: {persistent_fbx}")

            # Create texture preview output
            texture_preview = None
            if skeleton.get('texture_data_base64'):
                texture_preview = decode_texture_to_comfy_image(skeleton['texture_data_base64'])
                if texture_preview is not None:
                    print(f"[UniRigApplySkinningML] Texture preview created: {skeleton['texture_width']}x{skeleton['texture_height']}")

            if texture_preview is None:
                texture_preview = create_placeholder_texture()

            total_time = time.time() - total_start
            print(f"[UniRigApplySkinningML] Skinning application complete!")
            print(f"[UniRigApplySkinningML] TOTAL TIME: {total_time:.2f}s")

            return (persistent_fbx, texture_preview)

        finally:
            # Clean up temp directory
            # Note: We keep it for debugging in original implementation
            pass
