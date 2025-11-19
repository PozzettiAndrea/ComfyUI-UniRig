"""
Skeleton extraction nodes for UniRig.
"""

import os
import sys
import subprocess
import tempfile
import numpy as np
from trimesh import Trimesh
import time
import shutil
import folder_paths

# Support both relative imports (ComfyUI) and absolute imports (testing)
try:
    from ..constants import BLENDER_TIMEOUT, INFERENCE_TIMEOUT, PARSE_TIMEOUT, TARGET_FACE_COUNT
except ImportError:
    from constants import BLENDER_TIMEOUT, INFERENCE_TIMEOUT, PARSE_TIMEOUT, TARGET_FACE_COUNT

try:
    from .base import (
        UNIRIG_PATH,
        BLENDER_EXE,
        BLENDER_SCRIPT,
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
        BLENDER_SCRIPT,
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



class UniRigExtractSkeletonNew:
    """
    Extract skeleton from mesh using UniRig (SIGGRAPH 2025) - CACHED MODEL ONLY.

    Uses ML-based approach for high-quality semantic skeleton extraction.
    Works on any mesh type: humans, animals, objects, cameras, etc.

    This version uses ONLY in-process GPU cached models for faster inference.
    Requires pre-loaded model from UniRigLoadSkeletonModel.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "skeleton_model": ("UNIRIG_SKELETON_MODEL", {
                    "tooltip": "Pre-loaded skeleton model (from UniRigLoadSkeletonModel) - REQUIRED"
                }),
                "seed": ("INT", {"default": 42, "min": 0, "max": 4294967295,
                               "tooltip": "Random seed for skeleton generation variation"}),
            },
            "optional": {
                "target_face_count": ("INT", {
                    "default": 50000,
                    "min": 10000,
                    "max": 500000,
                    "step": 10000,
                    "tooltip": "Target face count for mesh decimation. Higher = preserve more detail, slower. Default: 50000"
                }),
            }
        }

    RETURN_TYPES = ("TRIMESH", "SKELETON", "IMAGE")
    RETURN_NAMES = ("normalized_mesh", "skeleton", "texture_preview")
    FUNCTION = "extract"
    CATEGORY = "UniRig"

    def extract(self, trimesh, skeleton_model, seed, target_face_count=None):
        """Extract skeleton using UniRig with cached model only."""
        total_start = time.time()
        print(f"[UniRigExtractSkeletonNew] Starting skeleton extraction (cached model only)...")

        # Validate model is provided
        if skeleton_model is None:
            raise RuntimeError(
                "skeleton_model is required for UniRigExtractSkeletonNew. "
                "Please connect a UniRigLoadSkeletonModel node."
            )

        # Validate model has cache key
        if not skeleton_model.get("model_cache_key"):
            raise RuntimeError(
                "skeleton_model does not have a cached model. "
                "Ensure UniRigLoadSkeletonModel has 'cache_to_gpu' enabled."
            )

        print(f"[UniRigExtractSkeletonNew] Using pre-loaded cached model")
        task_config_path = skeleton_model.get("task_config_path")

        # Check if Blender is available
        if not BLENDER_EXE or not os.path.exists(BLENDER_EXE):
            raise RuntimeError(
                f"Blender not found. Please run install_blender.py or install manually."
            )

        # Check if UniRig is available
        if not os.path.exists(UNIRIG_PATH):
            raise RuntimeError(
                f"UniRig code not found at {UNIRIG_PATH}. "
                "The lib/unirig directory should contain the UniRig source code."
            )

        # Create temp files
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.glb")
            npz_dir = os.path.join(tmpdir, "input")
            npz_path = os.path.join(npz_dir, "raw_data.npz")
            output_path = os.path.join(tmpdir, "skeleton.fbx")

            os.makedirs(npz_dir, exist_ok=True)

            # Export mesh to GLB
            step_start = time.time()
            print(f"[UniRigExtractSkeletonNew] Exporting mesh to {input_path}")
            print(f"[UniRigExtractSkeletonNew] Mesh has {len(trimesh.vertices)} vertices, {len(trimesh.faces)} faces")
            trimesh.export(input_path)
            export_time = time.time() - step_start
            print(f"[UniRigExtractSkeletonNew] Mesh exported in {export_time:.2f}s")

            # Step 1: Extract/preprocess mesh with Blender
            step_start = time.time()
            print(f"[UniRigExtractSkeletonNew] Step 1: Preprocessing mesh with Blender...")
            actual_face_count = target_face_count if target_face_count is not None else TARGET_FACE_COUNT
            print(f"[UniRigExtractSkeletonNew] Using target face count: {actual_face_count}")
            blender_cmd = [
                BLENDER_EXE,
                "--background",
                "--python", BLENDER_SCRIPT,
                "--",
                input_path,
                npz_path,
                str(actual_face_count)
            ]

            try:
                result = subprocess.run(
                    blender_cmd,
                    capture_output=True,
                    text=True,
                    timeout=BLENDER_TIMEOUT
                )
                if result.stdout:
                    print(f"[UniRigExtractSkeletonNew] Blender output:\n{result.stdout}")
                if result.stderr:
                    stderr_lines = result.stderr.split('\n')
                    important_lines = [l for l in stderr_lines if 'error' in l.lower() or 'fail' in l.lower()]
                    if important_lines:
                        print(f"[UniRigExtractSkeletonNew] Blender warnings:\n" + '\n'.join(important_lines))

                if not os.path.exists(npz_path):
                    raise RuntimeError(f"Blender extraction failed: {npz_path} not created")

                blender_time = time.time() - step_start
                print(f"[UniRigExtractSkeletonNew] Mesh preprocessed in {blender_time:.2f}s: {npz_path}")

            except subprocess.TimeoutExpired:
                raise RuntimeError(f"Blender extraction timed out (>{BLENDER_TIMEOUT}s)")
            except Exception as e:
                print(f"[UniRigExtractSkeletonNew] Blender error: {e}")
                raise

            # Step 2: Run skeleton inference with CACHED MODEL ONLY
            step_start = time.time()
            print(f"[UniRigExtractSkeletonNew] Step 2: Running skeleton inference with cached model...")

            model_cache = _get_model_cache()
            if not model_cache:
                raise RuntimeError(
                    "Model cache module not available. "
                    "Cannot run cached inference."
                )

            cache_key = skeleton_model["model_cache_key"]
            print(f"[UniRigExtractSkeletonNew] Using cached model: {cache_key}")

            request_data = {
                "seed": seed,
                "input": input_path,
                "output": output_path,
                "npz_dir": tmpdir,
                "cls": None,
                "data_name": "raw_data.npz",
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
                        f"If you need fallback support, use UniRigExtractSkeleton instead."
                    )

                inference_time = time.time() - step_start
                print(f"[UniRigExtractSkeletonNew] ✓ Cached inference completed in {inference_time:.2f}s")

            except Exception as e:
                raise RuntimeError(
                    f"Cached model inference exception: {str(e)}\n\n"
                    f"This node requires a working cached model. "
                    f"If you need fallback support, use UniRigExtractSkeleton instead."
                )

            # Load and parse FBX output
            if not os.path.exists(output_path):
                tmpdir_contents = os.listdir(tmpdir)
                print(f"[UniRigExtractSkeletonNew] Output FBX not found: {output_path}")
                print(f"[UniRigExtractSkeletonNew] Temp directory contents: {tmpdir_contents}")
                raise RuntimeError(
                    f"UniRig did not generate output file: {output_path}\n"
                    f"Temp directory contents: {tmpdir_contents}\n"
                    f"Check stdout/stderr above for details"
                )

            print(f"[UniRigExtractSkeletonNew] Found output FBX: {output_path}")
            fbx_size = os.path.getsize(output_path)
            print(f"[UniRigExtractSkeletonNew] FBX file size: {fbx_size} bytes")

            step_start = time.time()
            print(f"[UniRigExtractSkeletonNew] Step 3: Parsing FBX output with Blender...")
            skeleton_npz = os.path.join(tmpdir, "skeleton_data.npz")

            # Use Blender to parse skeleton from FBX
            parse_cmd = [
                BLENDER_EXE,
                "--background",
                "--python", BLENDER_PARSE_SKELETON,
                "--",
                output_path,
                skeleton_npz,
            ]

            try:
                result = subprocess.run(
                    parse_cmd,
                    capture_output=True,
                    text=True,
                    timeout=PARSE_TIMEOUT
                )
                if result.stdout:
                    print(f"[UniRigExtractSkeletonNew] Blender parse output:\n{result.stdout}")
                if result.stderr:
                    stderr_lines = result.stderr.split('\n')
                    important_lines = [l for l in stderr_lines if 'error' in l.lower() or 'fail' in l.lower()]
                    if important_lines:
                        print(f"[UniRigExtractSkeletonNew] Blender parse warnings:\n" + '\n'.join(important_lines))

                if not os.path.exists(skeleton_npz):
                    print(f"[UniRigExtractSkeletonNew] Skeleton NPZ not found: {skeleton_npz}")
                    raise RuntimeError(f"Skeleton parsing failed: {skeleton_npz} not created")

                parse_time = time.time() - step_start
                print(f"[UniRigExtractSkeletonNew] Skeleton parsed in {parse_time:.2f}s")

            except subprocess.TimeoutExpired:
                raise RuntimeError(f"Skeleton parsing timed out (>{PARSE_TIMEOUT}s)")
            except Exception as e:
                print(f"[UniRigExtractSkeletonNew] Skeleton parse error: {e}")
                raise

            # Load skeleton data
            print(f"[UniRigExtractSkeletonNew] Loading skeleton data from NPZ...")
            skeleton_data = np.load(skeleton_npz, allow_pickle=True)
            print(f"[UniRigExtractSkeletonNew] NPZ contains keys: {list(skeleton_data.keys())}")
            all_joints = skeleton_data['vertices']
            edges = skeleton_data['edges']

            print(f"[UniRigExtractSkeletonNew] Extracted {len(all_joints)} joints, {len(edges)} bones")
            print(f"[UniRigExtractSkeletonNew] Skeleton already normalized by UniRig to range [{all_joints.min():.3f}, {all_joints.max():.3f}]")

            # Load preprocessing data
            preprocess_npz = os.path.join(tmpdir, "input", "raw_data.npz")
            uv_coords = None
            uv_faces = None
            material_name = None
            texture_path = None
            texture_data_base64 = None
            texture_format = None
            texture_width = 0
            texture_height = 0

            if os.path.exists(preprocess_npz):
                preprocess_data = np.load(preprocess_npz, allow_pickle=True)
                mesh_vertices_original = preprocess_data['vertices']
                mesh_faces = preprocess_data['faces']
                vertex_normals = preprocess_data.get('vertex_normals', None)
                face_normals = preprocess_data.get('face_normals', None)

                # Load UV coordinates if available
                if 'uv_coords' in preprocess_data and len(preprocess_data['uv_coords']) > 0:
                    uv_coords = preprocess_data['uv_coords']
                    uv_faces = preprocess_data.get('uv_faces', None)
                    print(f"[UniRigExtractSkeletonNew] Loaded UV coordinates: {len(uv_coords)} UVs")

                if 'material_name' in preprocess_data:
                    material_name = str(preprocess_data['material_name'])
                if 'texture_path' in preprocess_data:
                    texture_path = str(preprocess_data['texture_path'])

                # Load texture data if available
                if 'texture_data_base64' in preprocess_data:
                    tex_data = preprocess_data['texture_data_base64']
                    if tex_data is not None and len(str(tex_data)) > 0:
                        texture_data_base64 = str(tex_data)
                        texture_format = str(preprocess_data.get('texture_format', 'PNG'))
                        texture_width = int(preprocess_data.get('texture_width', 0))
                        texture_height = int(preprocess_data.get('texture_height', 0))
                        print(f"[UniRigExtractSkeletonNew] Loaded texture: {texture_width}x{texture_height} {texture_format} ({len(texture_data_base64) // 1024}KB base64)")
            else:
                # Fallback: use trimesh data
                mesh_vertices_original = np.array(trimesh.vertices, dtype=np.float32)
                mesh_faces = np.array(trimesh.faces, dtype=np.int32)
                vertex_normals = np.array(trimesh.vertex_normals, dtype=np.float32) if hasattr(trimesh, 'vertex_normals') else None
                face_normals = np.array(trimesh.face_normals, dtype=np.float32) if hasattr(trimesh, 'face_normals') else None

            # Normalize mesh to [-1, 1]
            mesh_bounds_min = mesh_vertices_original.min(axis=0)
            mesh_bounds_max = mesh_vertices_original.max(axis=0)
            mesh_center = (mesh_bounds_min + mesh_bounds_max) / 2
            mesh_extents = mesh_bounds_max - mesh_bounds_min
            mesh_scale = mesh_extents.max() / 2

            # Normalize mesh vertices to [-1, 1]
            mesh_vertices = (mesh_vertices_original - mesh_center) / mesh_scale

            print(f"[UniRigExtractSkeletonNew] Original mesh bounds: min={mesh_bounds_min}, max={mesh_bounds_max}")
            print(f"[UniRigExtractSkeletonNew] Mesh scale: {mesh_scale:.4f}, extents: {mesh_extents}")
            print(f"[UniRigExtractSkeletonNew] Normalized mesh bounds: min={mesh_vertices.min(axis=0)}, max={mesh_vertices.max(axis=0)}")

            # Create trimesh object from normalized mesh data
            normalized_mesh = Trimesh(
                vertices=mesh_vertices,
                faces=mesh_faces,
                process=True
            )
            print(f"[UniRigExtractSkeletonNew] Created normalized mesh: {len(mesh_vertices)} vertices, {len(mesh_faces)} faces")

            # Build parents list from bone_parents
            if 'bone_parents' in skeleton_data:
                bone_parents = skeleton_data['bone_parents']
                num_bones = len(bone_parents)
                parents_list = [None if p == -1 else int(p) for p in bone_parents]

                # Get bone names
                bone_names = skeleton_data.get('bone_names', None)
                if bone_names is not None:
                    names_list = [str(n) for n in bone_names]
                else:
                    names_list = [f"bone_{i}" for i in range(num_bones)]

                # Map bones to their head joint positions
                if 'bone_to_head_vertex' in skeleton_data:
                    bone_to_head = skeleton_data['bone_to_head_vertex']
                    bone_joints = np.array([all_joints[bone_to_head[i]] for i in range(num_bones)])
                else:
                    bone_joints = all_joints[:num_bones]

                # Compute tails
                tails = np.zeros((num_bones, 3))
                for i in range(num_bones):
                    children = [j for j, p in enumerate(parents_list) if p == i]
                    if children:
                        tails[i] = np.mean([bone_joints[c] for c in children], axis=0)
                    else:
                        if parents_list[i] is not None:
                            direction = bone_joints[i] - bone_joints[parents_list[i]]
                            tails[i] = bone_joints[i] + direction * 0.3
                        else:
                            tails[i] = bone_joints[i] + np.array([0, 0.1, 0])

            else:
                # No hierarchy - create simple chain
                num_bones = len(all_joints)
                bone_joints = all_joints
                parents_list = [None] + list(range(num_bones-1))
                names_list = [f"bone_{i}" for i in range(num_bones)]

                tails = np.zeros_like(bone_joints)
                for i in range(num_bones):
                    children = [j for j, p in enumerate(parents_list) if p == i]
                    if children:
                        tails[i] = np.mean([bone_joints[c] for c in children], axis=0)
                    else:
                        if parents_list[i] is not None:
                            direction = bone_joints[i] - bone_joints[parents_list[i]]
                            tails[i] = bone_joints[i] + direction * 0.3
                        else:
                            tails[i] = bone_joints[i] + np.array([0, 0.1, 0])

            # Save as RawData NPZ for skinning phase
            persistent_npz = os.path.join(folder_paths.get_temp_directory(), f"skeleton_{seed}.npz")
            np.savez(
                persistent_npz,
                vertices=mesh_vertices,
                vertex_normals=vertex_normals,
                faces=mesh_faces,
                face_normals=face_normals,
                joints=bone_joints,
                tails=tails,
                parents=np.array(parents_list, dtype=object),
                names=np.array(names_list, dtype=object),
                uv_coords=uv_coords if uv_coords is not None else np.array([], dtype=np.float32),
                uv_faces=uv_faces if uv_faces is not None else np.array([], dtype=np.int32),
                material_name=material_name if material_name else "",
                texture_path=texture_path if texture_path else "",
                mesh_bounds_min=mesh_bounds_min,
                mesh_bounds_max=mesh_bounds_max,
                mesh_center=mesh_center,
                mesh_scale=mesh_scale,
                skin=None,
                no_skin=None,
                matrix_local=None,
                path=None,
                cls=None
            )
            print(f"[UniRigExtractSkeletonNew] Saved skeleton NPZ to: {persistent_npz}")

            # Build skeleton dict with ALL data
            skeleton = {
                "vertices": all_joints,
                "edges": edges,
                "joints": bone_joints,
                "tails": tails,
                "names": names_list,
                "parents": parents_list,
                "mesh_vertices": mesh_vertices,
                "mesh_faces": mesh_faces,
                "mesh_vertex_normals": vertex_normals,
                "mesh_face_normals": face_normals,
                "uv_coords": uv_coords,
                "uv_faces": uv_faces,
                "material_name": material_name,
                "texture_path": texture_path,
                "texture_data_base64": texture_data_base64,
                "texture_format": texture_format,
                "texture_width": texture_width,
                "texture_height": texture_height,
                "mesh_bounds_min": mesh_bounds_min,
                "mesh_bounds_max": mesh_bounds_max,
                "mesh_center": mesh_center,
                "mesh_scale": mesh_scale,
                "is_normalized": True,
                "skeleton_npz_path": persistent_npz,
                "bone_names": names_list,
                "bone_parents": parents_list,
            }

            if 'bone_to_head_vertex' in skeleton_data:
                skeleton['bone_to_head_vertex'] = skeleton_data['bone_to_head_vertex'].tolist()

            print(f"[UniRigExtractSkeletonNew] Included hierarchy: {len(names_list)} bones with parent relationships")

            # Create texture preview output
            if texture_data_base64:
                texture_preview, tex_w, tex_h = decode_texture_to_comfy_image(texture_data_base64)
                if texture_preview is not None:
                    print(f"[UniRigExtractSkeletonNew] Texture preview created: {tex_w}x{tex_h}")
                else:
                    print(f"[UniRigExtractSkeletonNew] Warning: Could not decode texture for preview")
                    texture_preview = create_placeholder_texture()
            else:
                print(f"[UniRigExtractSkeletonNew] No texture available for preview")
                texture_preview = create_placeholder_texture()

            total_time = time.time() - total_start
            print(f"[UniRigExtractSkeletonNew] Skeleton extraction complete!")
            print(f"[UniRigExtractSkeletonNew] TOTAL TIME: {total_time:.2f}s")
            return (normalized_mesh, skeleton, texture_preview)
