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

from ..constants import BLENDER_TIMEOUT, INFERENCE_TIMEOUT, PARSE_TIMEOUT, TARGET_FACE_COUNT
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


class UniRigExtractSkeleton:
    """
    Extract skeleton from mesh using UniRig (SIGGRAPH 2025).

    Uses ML-based approach for high-quality semantic skeleton extraction.
    Works on any mesh type: humans, animals, objects, cameras, etc.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "seed": ("INT", {"default": 42, "min": 0, "max": 4294967295,
                               "tooltip": "Random seed for skeleton generation variation"}),
                "skeleton_model": ("UNIRIG_SKELETON_MODEL", {
                    "tooltip": "Pre-loaded skeleton model (from UniRigLoadSkeletonModel). Required for inference."
                }),
            },
        }

    RETURN_TYPES = ("TRIMESH", "SKELETON", "IMAGE")
    RETURN_NAMES = ("normalized_mesh", "skeleton", "texture_preview")
    FUNCTION = "extract"
    CATEGORY = "UniRig"

    def extract(self, trimesh, seed, skeleton_model):
        """Extract skeleton using UniRig."""
        total_start = time.time()
        print(f"[UniRigExtractSkeleton] Starting ({len(trimesh.vertices)} verts, {len(trimesh.faces)} faces)")

        # Validate skeleton_model
        if skeleton_model is None:
            raise ValueError("skeleton_model is required. Use UniRigLoadSkeletonModel to load the model first.")

        task_config_path = skeleton_model.get("task_config_path")
        if not task_config_path:
            raise ValueError("skeleton_model missing 'task_config_path'. Model may be corrupted.")

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
            trimesh.export(input_path)

            # Step 1: Extract/preprocess mesh with Blender
            step_start = time.time()
            print(f"[UniRigExtractSkeleton] Step 1: Preprocessing mesh with Blender...")
            blender_cmd = [
                BLENDER_EXE,
                "--background",
                "--python", BLENDER_SCRIPT,
                "--",
                input_path,
                npz_path,
                str(TARGET_FACE_COUNT)
            ]

            try:
                result = subprocess.run(
                    blender_cmd,
                    capture_output=True,
                    text=True,
                    timeout=BLENDER_TIMEOUT
                )
                # Only show errors, not full output
                if result.stderr:
                    stderr_lines = result.stderr.split('\n')
                    error_lines = [l for l in stderr_lines if 'error' in l.lower() or 'fail' in l.lower()]
                    if error_lines:
                        print(f"[UniRigExtractSkeleton] Blender errors:\n" + '\n'.join(error_lines))

                if not os.path.exists(npz_path):
                    # Show full output on failure for debugging
                    if result.stdout:
                        print(f"[UniRigExtractSkeleton] Blender output:\n{result.stdout}")
                    raise RuntimeError(f"Blender extraction failed: {npz_path} not created")

                blender_time = time.time() - step_start
                print(f"[UniRigExtractSkeleton] Mesh preprocessed in {blender_time:.2f}s")

            except subprocess.TimeoutExpired:
                raise RuntimeError(f"Blender extraction timed out (>{BLENDER_TIMEOUT}s)")
            except Exception as e:
                print(f"[UniRigExtractSkeleton] Blender error: {e}")
                raise

            # Step 2: Run skeleton inference
            step_start = time.time()
            print(f"[UniRigExtractSkeleton] Step 2: Running skeleton inference...")

            # Check if we can use the in-process cached model
            model_cache_key = skeleton_model.get("model_cache_key")
            cache_to_gpu = skeleton_model.get("cache_to_gpu", True)
            use_cache = False

            if model_cache_key and cache_to_gpu:
                model_cache = _get_model_cache()
                if model_cache and model_cache.is_model_loaded(model_cache_key):
                    use_cache = True
                    print(f"[UniRigExtractSkeleton] Using cached GPU model for inference")

            if use_cache:
                # Use in-process cached model for inference
                try:
                    response = model_cache.run_inference(
                        cache_key=model_cache_key,
                        request_data={
                            "input": input_path,
                            "output": output_path,
                            "npz_dir": tmpdir,
                            "seed": seed,
                        }
                    )

                    if response.get("error"):
                        print(f"[UniRigExtractSkeleton] Cached model inference error: {response.get('error')}")
                        if response.get("traceback"):
                            print(response.get("traceback"))
                        raise RuntimeError(f"Cached model inference failed: {response.get('error')}")

                    inference_time = time.time() - step_start
                    print(f"[UniRigExtractSkeleton] Inference completed in {inference_time:.2f}s (cached)")

                except Exception as e:
                    print(f"[UniRigExtractSkeleton] Cached model inference error: {e}")
                    raise
            else:
                # Fallback to subprocess
                run_cmd = [
                    sys.executable, os.path.join(UNIRIG_PATH, "run.py"),
                    "--task", task_config_path,
                    "--seed", str(seed),
                    "--input", input_path,
                    "--output", output_path,
                    "--npz_dir", tmpdir,
                ]

                env = setup_subprocess_env()

                # Pass GPU caching preference to subprocess
                env["UNIRIG_CACHE_TO_GPU"] = "1" if cache_to_gpu else "0"

                try:
                    result = subprocess.run(
                        run_cmd,
                        cwd=UNIRIG_PATH,
                        env=env,
                        capture_output=True,
                        text=True,
                        timeout=INFERENCE_TIMEOUT
                    )
                    # Filter output to reduce noise
                    if result.returncode != 0:
                        # Show full output on failure
                        if result.stdout:
                            print(f"[UniRigExtractSkeleton] Inference stdout:\n{result.stdout}")
                        if result.stderr:
                            print(f"[UniRigExtractSkeleton] Inference stderr:\n{result.stderr}")
                        raise RuntimeError(f"Inference failed with exit code {result.returncode}")
                    else:
                        # On success, only show errors from stderr (filter out warnings)
                        if result.stderr:
                            stderr_lines = result.stderr.split('\n')
                            error_lines = [l for l in stderr_lines if 'error' in l.lower() and 'warning' not in l.lower()]
                            if error_lines:
                                print(f"[UniRigExtractSkeleton] Inference warnings:\n" + '\n'.join(error_lines[:5]))

                    inference_time = time.time() - step_start
                    print(f"[UniRigExtractSkeleton] Inference completed in {inference_time:.2f}s")

                except subprocess.TimeoutExpired:
                    raise RuntimeError(f"Inference timed out (>{INFERENCE_TIMEOUT}s)")
                except Exception as e:
                    print(f"[UniRigExtractSkeleton] Inference error: {e}")
                    raise

            # Load and parse FBX output
            if not os.path.exists(output_path):
                tmpdir_contents = os.listdir(tmpdir)
                raise RuntimeError(
                    f"UniRig did not generate output file: {output_path}\n"
                    f"Temp directory contents: {tmpdir_contents}\n"
                    f"Check stdout/stderr above for details"
                )

            step_start = time.time()
            print(f"[UniRigExtractSkeleton] Step 3: Parsing FBX output...")
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
                # Only show errors
                if result.stderr:
                    stderr_lines = result.stderr.split('\n')
                    error_lines = [l for l in stderr_lines if 'error' in l.lower() or 'fail' in l.lower()]
                    if error_lines:
                        print(f"[UniRigExtractSkeleton] Blender parse errors:\n" + '\n'.join(error_lines))

                if not os.path.exists(skeleton_npz):
                    # Show full output on failure
                    if result.stdout:
                        print(f"[UniRigExtractSkeleton] Blender parse output:\n{result.stdout}")
                    raise RuntimeError(f"Skeleton parsing failed: {skeleton_npz} not created")

                parse_time = time.time() - step_start
                print(f"[UniRigExtractSkeleton] Skeleton parsed in {parse_time:.2f}s")

            except subprocess.TimeoutExpired:
                raise RuntimeError(f"Skeleton parsing timed out (>{PARSE_TIMEOUT}s)")
            except Exception as e:
                print(f"[UniRigExtractSkeleton] Skeleton parse error: {e}")
                raise

            # Load skeleton data
            skeleton_data = np.load(skeleton_npz, allow_pickle=True)
            all_joints = skeleton_data['vertices']
            edges = skeleton_data['edges']

            print(f"[UniRigExtractSkeleton] Extracted {len(all_joints)} joints, {len(edges)} bones")

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

            # Create trimesh object from normalized mesh data
            normalized_mesh = Trimesh(
                vertices=mesh_vertices,
                faces=mesh_faces,
                process=True
            )
            print(f"[UniRigExtractSkeleton] Normalized mesh: {len(mesh_vertices)} vertices, {len(mesh_faces)} faces")

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

            # Create texture preview output
            if texture_data_base64:
                texture_preview, tex_w, tex_h = decode_texture_to_comfy_image(texture_data_base64)
                if texture_preview is None:
                    texture_preview = create_placeholder_texture()
            else:
                texture_preview = create_placeholder_texture()

            total_time = time.time() - total_start
            print(f"[UniRigExtractSkeleton] Complete! {len(names_list)} bones, {total_time:.2f}s total")
            return (normalized_mesh, skeleton, texture_preview)
