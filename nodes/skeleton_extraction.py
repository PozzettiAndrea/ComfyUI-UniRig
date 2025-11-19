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
    Extract skeleton from mesh using UniRig (SIGGRAPH 2025) - New implementation with model caching.

    Uses ML-based approach for high-quality semantic skeleton extraction with in-process model caching.
    Works on any mesh type: humans, animals, objects, cameras, etc.

    Note: For the original simple subprocess approach, use UniRigExtractSkeleton (without 'New' suffix).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "seed": ("INT", {"default": 42, "min": 0, "max": 4294967295,
                               "tooltip": "Random seed for skeleton generation variation"}),
            },
            "optional": {
                "skeleton_model": ("UNIRIG_SKELETON_MODEL", {
                    "tooltip": "Pre-loaded skeleton model (from UniRigLoadSkeletonModel)"
                }),
                "checkpoint": ("STRING", {
                    "default": "VAST-AI/UniRig",
                    "tooltip": "HuggingFace model ID or local path (ignored if skeleton_model provided)"
                }),
                "target_face_count": ("INT", {
                    "default": 50000,
                    "min": 10000,
                    "max": 500000,
                    "step": 10000,
                    "tooltip": "Target face count for mesh decimation. Higher = preserve more detail, slower. Default: 50000"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "TRIMESH", "SKELETON", "IMAGE")
    RETURN_NAMES = ("skeleton_fbx_path", "skeleton_npz_path", "normalized_mesh", "skeleton", "texture_preview")
    FUNCTION = "extract"
    CATEGORY = "UniRig"

    def extract(self, trimesh, seed, skeleton_model=None, checkpoint="VAST-AI/UniRig", target_face_count=None):
        """Extract skeleton using UniRig."""
        total_start = time.time()
        print(f"[UniRigExtractSkeleton] Starting skeleton extraction...")

        # Use pre-loaded model if available
        if skeleton_model is not None:
            print(f"[UniRigExtractSkeleton] Using pre-loaded model configuration")
            if skeleton_model.get("cached", False):
                print(f"[UniRigExtractSkeleton] Model weights already downloaded and cached")
            task_config_path = skeleton_model.get("task_config_path")
        else:
            task_config_path = os.path.join(UNIRIG_PATH, "configs/task/quick_inference_skeleton_articulationxl_ar_256.yaml")

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
            print(f"[UniRigExtractSkeleton] Exporting mesh to {input_path}")
            print(f"[UniRigExtractSkeleton] Mesh has {len(trimesh.vertices)} vertices, {len(trimesh.faces)} faces")
            trimesh.export(input_path)
            export_time = time.time() - step_start
            print(f"[UniRigExtractSkeleton] Mesh exported in {export_time:.2f}s")

            # Step 1: Extract/preprocess mesh with Blender
            step_start = time.time()
            print(f"[UniRigExtractSkeleton] Step 1: Preprocessing mesh with Blender...")
            actual_face_count = target_face_count if target_face_count is not None else TARGET_FACE_COUNT
            print(f"[UniRigExtractSkeleton] Using target face count: {actual_face_count}")
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
                    print(f"[UniRigExtractSkeleton] Blender output:\n{result.stdout}")
                if result.stderr:
                    stderr_lines = result.stderr.split('\n')
                    important_lines = [l for l in stderr_lines if 'error' in l.lower() or 'fail' in l.lower()]
                    if important_lines:
                        print(f"[UniRigExtractSkeleton] Blender warnings:\n" + '\n'.join(important_lines))

                if not os.path.exists(npz_path):
                    raise RuntimeError(f"Blender extraction failed: {npz_path} not created")

                blender_time = time.time() - step_start
                print(f"[UniRigExtractSkeleton] Mesh preprocessed in {blender_time:.2f}s: {npz_path}")

            except subprocess.TimeoutExpired:
                raise RuntimeError(f"Blender extraction timed out (>{BLENDER_TIMEOUT}s)")
            except Exception as e:
                print(f"[UniRigExtractSkeleton] Blender error: {e}")
                raise

            # Step 2: Run skeleton inference
            step_start = time.time()
            print(f"[UniRigExtractSkeleton] Step 2: Running skeleton inference...")

            # Try in-process cached model first
            used_cache = False
            if skeleton_model is not None and skeleton_model.get("model_cache_key"):
                model_cache = _get_model_cache()
                if model_cache:
                    cache_key = skeleton_model["model_cache_key"]
                    print(f"[UniRigExtractSkeleton] Using cached model (in-process inference)...")

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
                            print(f"[UniRigExtractSkeleton] Cache inference failed: {result['error']}")
                            if "traceback" in result:
                                print(f"[UniRigExtractSkeleton] Traceback:\n{result['traceback']}")
                            print(f"[UniRigExtractSkeleton] Falling back to subprocess inference...")
                        else:
                            used_cache = True
                            inference_time = time.time() - step_start
                            print(f"[UniRigExtractSkeleton] ✓ In-process inference completed in {inference_time:.2f}s")
                    except Exception as e:
                        print(f"[UniRigExtractSkeleton] Cache inference exception: {e}")
                        print(f"[UniRigExtractSkeleton] Falling back to subprocess inference...")

            # Fall back to subprocess if cache not used or failed
            if not used_cache:
                run_cmd = [
                    sys.executable, os.path.join(UNIRIG_PATH, "run.py"),
                    "--task", task_config_path,
                    "--seed", str(seed),
                    "--input", input_path,
                    "--output", output_path,
                    "--npz_dir", tmpdir,
                ]

                print(f"[UniRigExtractSkeleton] Running subprocess: {' '.join(run_cmd)}")
                print(f"[UniRigExtractSkeleton] Using Blender: {BLENDER_EXE}")
                print(f"[UniRigExtractSkeleton] Task config: {task_config_path}")

                env = setup_subprocess_env()
                print(f"[UniRigExtractSkeleton] Set BLENDER_EXE environment variable for FBX export")

                try:
                    result = subprocess.run(
                        run_cmd,
                        cwd=UNIRIG_PATH,
                        env=env,
                        capture_output=True,
                        text=True,
                        timeout=INFERENCE_TIMEOUT
                    )
                    if result.stdout:
                        print(f"[UniRigExtractSkeleton] Inference stdout:\n{result.stdout}")
                    if result.stderr:
                        print(f"[UniRigExtractSkeleton] Inference stderr:\n{result.stderr}")

                    if result.returncode != 0:
                        print(f"[UniRigExtractSkeleton] Inference failed with exit code {result.returncode}")
                        raise RuntimeError(f"Inference failed with exit code {result.returncode}")

                    inference_time = time.time() - step_start
                    print(f"[UniRigExtractSkeleton] Subprocess inference completed in {inference_time:.2f}s")

                except subprocess.TimeoutExpired:
                    raise RuntimeError(f"Inference timed out (>{INFERENCE_TIMEOUT}s)")
                except Exception as e:
                    print(f"[UniRigExtractSkeleton] Inference error: {e}")
                    raise

            # Load and parse FBX output
            if not os.path.exists(output_path):
                tmpdir_contents = os.listdir(tmpdir)
                print(f"[UniRigExtractSkeleton] Output FBX not found: {output_path}")
                print(f"[UniRigExtractSkeleton] Temp directory contents: {tmpdir_contents}")
                raise RuntimeError(
                    f"UniRig did not generate output file: {output_path}\n"
                    f"Temp directory contents: {tmpdir_contents}\n"
                    f"Check stdout/stderr above for details"
                )

            print(f"[UniRigExtractSkeleton] Found output FBX: {output_path}")
            fbx_size = os.path.getsize(output_path)
            print(f"[UniRigExtractSkeleton] FBX file size: {fbx_size} bytes")

            # Save skeleton FBX to persistent location
            persistent_fbx_path = os.path.join(folder_paths.get_temp_directory(), f"skeleton_{seed}.fbx")
            shutil.copy2(output_path, persistent_fbx_path)
            print(f"[UniRigExtractSkeleton] Saved persistent FBX: {persistent_fbx_path}")

            step_start = time.time()
            print(f"[UniRigExtractSkeleton] Step 3: Parsing FBX output with Blender...")
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
                    print(f"[UniRigExtractSkeleton] Blender parse output:\n{result.stdout}")
                if result.stderr:
                    stderr_lines = result.stderr.split('\n')
                    important_lines = [l for l in stderr_lines if 'error' in l.lower() or 'fail' in l.lower()]
                    if important_lines:
                        print(f"[UniRigExtractSkeleton] Blender parse warnings:\n" + '\n'.join(important_lines))

                if not os.path.exists(skeleton_npz):
                    print(f"[UniRigExtractSkeleton] Skeleton NPZ not found: {skeleton_npz}")
                    raise RuntimeError(f"Skeleton parsing failed: {skeleton_npz} not created")

                parse_time = time.time() - step_start
                print(f"[UniRigExtractSkeleton] Skeleton parsed in {parse_time:.2f}s")

            except subprocess.TimeoutExpired:
                raise RuntimeError(f"Skeleton parsing timed out (>{PARSE_TIMEOUT}s)")
            except Exception as e:
                print(f"[UniRigExtractSkeleton] Skeleton parse error: {e}")
                raise

            # Load skeleton data
            print(f"[UniRigExtractSkeleton] Loading skeleton data from NPZ...")
            skeleton_data = np.load(skeleton_npz, allow_pickle=True)
            print(f"[UniRigExtractSkeleton] NPZ contains keys: {list(skeleton_data.keys())}")
            all_joints = skeleton_data['vertices']
            edges = skeleton_data['edges']

            print(f"[UniRigExtractSkeleton] Extracted {len(all_joints)} joints, {len(edges)} bones")
            print(f"[UniRigExtractSkeleton] Skeleton already normalized by UniRig to range [{all_joints.min():.3f}, {all_joints.max():.3f}]")

            # Load FBX mesh data if available (ground truth from ML model)
            mesh_vertices_fbx = None
            mesh_faces_fbx = None
            mesh_vertex_normals_fbx = None
            mesh_face_normals_fbx = None
            if 'mesh_vertices_fbx' in skeleton_data:
                mesh_vertices_fbx = skeleton_data['mesh_vertices_fbx']
                mesh_faces_fbx = skeleton_data.get('mesh_faces_fbx', None)
                mesh_vertex_normals_fbx = skeleton_data.get('mesh_vertex_normals_fbx', None)
                mesh_face_normals_fbx = skeleton_data.get('mesh_face_normals_fbx', None)
                print(f"[UniRigExtractSkeleton] Found FBX mesh: {len(mesh_vertices_fbx)} vertices, {len(mesh_faces_fbx) if mesh_faces_fbx is not None else 0} faces")
                if mesh_vertex_normals_fbx is not None:
                    print(f"[UniRigExtractSkeleton] Found FBX normals: {len(mesh_vertex_normals_fbx)} vertex normals, {len(mesh_face_normals_fbx) if mesh_face_normals_fbx is not None else 0} face normals")

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
                    print(f"[UniRigExtractSkeleton] Loaded UV coordinates: {len(uv_coords)} UVs")

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
                        print(f"[UniRigExtractSkeleton] Loaded texture: {texture_width}x{texture_height} {texture_format} ({len(texture_data_base64) // 1024}KB base64)")
            else:
                print("[UniRigExtractSkeleton] Warning: No preprocessing data found - this should not happen in normal workflow")

            # Use FBX mesh as the single source of truth (already normalized by ML model)
            if mesh_vertices_fbx is None or mesh_faces_fbx is None:
                raise RuntimeError("FBX mesh data not found in skeleton.fbx - cannot proceed without ground truth mesh")

            print(f"[UniRigExtractSkeleton] Using FBX mesh as single source of truth")
            print(f"[UniRigExtractSkeleton] FBX mesh bounds: min={mesh_vertices_fbx.min(axis=0)}, max={mesh_vertices_fbx.max(axis=0)}")

            # Create trimesh object from FBX mesh data (already normalized by ML model)
            normalized_mesh = Trimesh(
                vertices=mesh_vertices_fbx,
                faces=mesh_faces_fbx,
                vertex_normals=mesh_vertex_normals_fbx,
                process=False  # Don't reprocess - use data as-is from FBX
            )
            print(f"[UniRigExtractSkeleton] Created normalized mesh from FBX: {len(mesh_vertices_fbx)} vertices, {len(mesh_faces_fbx)} faces")

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

            # Save as RawData NPZ for skinning phase (using FBX mesh only)
            persistent_npz = os.path.join(folder_paths.get_temp_directory(), f"skeleton_{seed}.npz")
            np.savez(
                persistent_npz,
                vertices=mesh_vertices_fbx,
                vertex_normals=mesh_vertex_normals_fbx,
                faces=mesh_faces_fbx,
                face_normals=mesh_face_normals_fbx,
                joints=bone_joints,
                tails=tails,
                parents=np.array(parents_list, dtype=object),
                names=np.array(names_list, dtype=object),
                uv_coords=uv_coords if uv_coords is not None else np.array([], dtype=np.float32),
                uv_faces=uv_faces if uv_faces is not None else np.array([], dtype=np.int32),
                material_name=material_name if material_name else "",
                texture_path=texture_path if texture_path else "",
                texture_data_base64=texture_data_base64 if texture_data_base64 else "",
                texture_format=texture_format if texture_format else "",
                texture_width=texture_width,
                texture_height=texture_height,
                skin=None,
                no_skin=None,
                matrix_local=None,
                path=None,
                cls=None
            )
            print(f"[UniRigExtractSkeleton] Saved skeleton NPZ to: {persistent_npz}")

            # Build skeleton dict with FBX mesh only (single source of truth)
            skeleton = {
                # Skeleton structure
                "vertices": all_joints,
                "edges": edges,
                "joints": bone_joints,
                "tails": tails,
                "names": names_list,
                "parents": parents_list,
                "bone_names": names_list,
                "bone_parents": parents_list,
                # FBX mesh data (ground truth from ML model, already normalized)
                "mesh_vertices_fbx": mesh_vertices_fbx,
                "mesh_faces_fbx": mesh_faces_fbx,
                "mesh_vertex_normals_fbx": mesh_vertex_normals_fbx,
                "mesh_face_normals_fbx": mesh_face_normals_fbx,
                # Texture data
                "uv_coords": uv_coords,
                "uv_faces": uv_faces,
                "material_name": material_name,
                "texture_path": texture_path,
                "texture_data_base64": texture_data_base64,
                "texture_format": texture_format,
                "texture_width": texture_width,
                "texture_height": texture_height,
                # Metadata
                "skeleton_npz_path": persistent_npz,
            }

            if 'bone_to_head_vertex' in skeleton_data:
                skeleton['bone_to_head_vertex'] = skeleton_data['bone_to_head_vertex'].tolist()

            print(f"[UniRigExtractSkeleton] Built skeleton dict with FBX mesh: {len(mesh_vertices_fbx)} vertices, {len(mesh_faces_fbx)} faces")
            print(f"[UniRigExtractSkeleton] FBX mesh includes: {len(mesh_vertex_normals_fbx) if mesh_vertex_normals_fbx is not None else 0} vertex normals, {len(mesh_face_normals_fbx) if mesh_face_normals_fbx is not None else 0} face normals")

            print(f"[UniRigExtractSkeleton] Included hierarchy: {len(names_list)} bones with parent relationships")

            # Create texture preview output
            if texture_data_base64:
                texture_preview, tex_w, tex_h = decode_texture_to_comfy_image(texture_data_base64)
                if texture_preview is not None:
                    print(f"[UniRigExtractSkeleton] Texture preview created: {tex_w}x{tex_h}")
                else:
                    print(f"[UniRigExtractSkeleton] Warning: Could not decode texture for preview")
                    texture_preview = create_placeholder_texture()
            else:
                print(f"[UniRigExtractSkeleton] No texture available for preview")
                texture_preview = create_placeholder_texture()

            total_time = time.time() - total_start
            print(f"[UniRigExtractSkeleton] Skeleton extraction complete!")
            print(f"[UniRigExtractSkeleton] TOTAL TIME: {total_time:.2f}s")
            return (persistent_fbx_path, persistent_npz, normalized_mesh, skeleton, texture_preview)


class UniRigExtractRig:
    """
    Extract full rig (skeleton + skinning weights) using UniRig.

    This node runs both skeleton and skinning prediction.
    Output includes skinning weights for animation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "seed": ("INT", {"default": 42, "min": 0, "max": 4294967295}),
            },
            "optional": {
                "checkpoint": ("STRING", {
                    "default": "VAST-AI/UniRig",
                    "tooltip": "HuggingFace model ID or local path"
                }),
            }
        }

    RETURN_TYPES = ("RIGGED_MESH",)
    RETURN_NAMES = ("rigged_mesh",)
    FUNCTION = "extract"
    CATEGORY = "UniRig"

    def extract(self, trimesh, seed, checkpoint="VAST-AI/UniRig"):
        """Extract full rig with skinning weights."""
        total_start = time.time()
        print(f"[UniRigExtractRig] Starting full rig extraction...")

        # Check if Blender is available
        if not BLENDER_EXE or not os.path.exists(BLENDER_EXE):
            raise RuntimeError(f"Blender not found. Please run install_blender.py or install manually.")

        if not os.path.exists(UNIRIG_PATH):
            raise RuntimeError(f"UniRig not found at {UNIRIG_PATH}")

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.glb")
            npz_dir = os.path.join(tmpdir, "input")
            npz_path = os.path.join(npz_dir, "raw_data.npz")
            skeleton_npz_path = os.path.join(tmpdir, "predict_skeleton.npz")
            output_path = os.path.join(tmpdir, "result_fbx.fbx")

            os.makedirs(npz_dir, exist_ok=True)

            # Export mesh to GLB
            step_start = time.time()
            print(f"[UniRigExtractRig] Exporting mesh: {len(trimesh.vertices)} vertices, {len(trimesh.faces)} faces")
            trimesh.export(input_path)
            export_time = time.time() - step_start
            print(f"[UniRigExtractRig] Mesh exported in {export_time:.2f}s")

            # Step 1: Preprocess mesh with Blender
            step_start = time.time()
            print(f"[UniRigExtractRig] Step 1: Preprocessing mesh with Blender...")
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
                result = subprocess.run(blender_cmd, capture_output=True, text=True, timeout=BLENDER_TIMEOUT)
                if result.stdout:
                    print(f"[UniRigExtractRig] Blender output:\n{result.stdout}")

                if not os.path.exists(npz_path):
                    raise RuntimeError(f"Blender extraction failed: {npz_path} not created")

                blender_time = time.time() - step_start
                print(f"[UniRigExtractRig] Mesh preprocessed in {blender_time:.2f}s")

            except subprocess.TimeoutExpired:
                raise RuntimeError(f"Blender extraction timed out (>{BLENDER_TIMEOUT}s)")
            except Exception as e:
                print(f"[UniRigExtractRig] Blender error: {e}")
                raise

            # Step 2: Generate skeleton
            step_start = time.time()
            print(f"[UniRigExtractRig] Step 2: Generating skeleton...")

            skeleton_fbx_path = os.path.join(tmpdir, "skeleton.fbx")

            skeleton_cmd = [
                sys.executable, os.path.join(UNIRIG_PATH, "run.py"),
                "--task", os.path.join(UNIRIG_PATH, "configs/task/quick_inference_skeleton_articulationxl_ar_256.yaml"),
                "--seed", str(seed),
                "--input", input_path,
                "--output", skeleton_fbx_path,
                "--npz_dir", tmpdir,
            ]

            env = setup_subprocess_env()

            try:
                result = subprocess.run(skeleton_cmd, cwd=UNIRIG_PATH, env=env, capture_output=True, text=True, timeout=INFERENCE_TIMEOUT)
                if result.stdout:
                    print(f"[UniRigExtractRig] Skeleton stdout:\n{result.stdout}")
                if result.stderr:
                    print(f"[UniRigExtractRig] Skeleton stderr:\n{result.stderr}")

                if result.returncode != 0:
                    raise RuntimeError(f"Skeleton generation failed with exit code {result.returncode}")

                if not os.path.exists(skeleton_fbx_path):
                    raise RuntimeError(f"Skeleton FBX not created: {skeleton_fbx_path}")

                print(f"[UniRigExtractRig] Skeleton FBX created: {skeleton_fbx_path}")

                # Parse the FBX to create predict_skeleton.npz
                print(f"[UniRigExtractRig] Creating skeleton NPZ from FBX...")
                parse_cmd = [
                    BLENDER_EXE,
                    "--background",
                    "--python", BLENDER_PARSE_SKELETON,
                    "--",
                    skeleton_fbx_path,
                    skeleton_npz_path,
                ]

                try:
                    result = subprocess.run(parse_cmd, capture_output=True, text=True, timeout=PARSE_TIMEOUT)
                    if result.stdout:
                        print(f"[UniRigExtractRig] Blender parse output:\n{result.stdout}")

                    if not os.path.exists(skeleton_npz_path):
                        raise RuntimeError(f"Failed to create skeleton NPZ: {skeleton_npz_path}")

                    print(f"[UniRigExtractRig] Skeleton NPZ created: {skeleton_npz_path}")

                except subprocess.TimeoutExpired:
                    raise RuntimeError(f"Skeleton NPZ creation timed out (>{PARSE_TIMEOUT}s)")
                except Exception as e:
                    print(f"[UniRigExtractRig] Skeleton NPZ creation error: {e}")
                    raise

                skeleton_time = time.time() - step_start
                print(f"[UniRigExtractRig] Skeleton generated in {skeleton_time:.2f}s")

            except subprocess.TimeoutExpired:
                raise RuntimeError(f"Skeleton generation timed out (>{INFERENCE_TIMEOUT}s)")
            except Exception as e:
                print(f"[UniRigExtractRig] Skeleton error: {e}")
                raise

            # Step 3: Generate skinning weights
            step_start = time.time()
            print(f"[UniRigExtractRig] Step 3: Generating skinning weights...")
            skin_cmd = [
                sys.executable, os.path.join(UNIRIG_PATH, "run.py"),
                "--task", os.path.join(UNIRIG_PATH, "configs/task/quick_inference_unirig_skin.yaml"),
                "--input", input_path,
                "--output", output_path,
                "--npz_dir", tmpdir,
            ]

            try:
                result = subprocess.run(skin_cmd, cwd=UNIRIG_PATH, env=env, capture_output=True, text=True, timeout=INFERENCE_TIMEOUT)
                if result.stdout:
                    print(f"[UniRigExtractRig] Skinning stdout:\n{result.stdout}")
                if result.stderr:
                    print(f"[UniRigExtractRig] Skinning stderr:\n{result.stderr}")

                if result.returncode != 0:
                    raise RuntimeError(f"Skinning generation failed with exit code {result.returncode}")

                # Look for the output FBX
                if not os.path.exists(output_path):
                    alt_paths = [
                        os.path.join(tmpdir, "results", "result_fbx.fbx"),
                        os.path.join(tmpdir, "input", "result_fbx.fbx"),
                    ]
                    for alt_path in alt_paths:
                        if os.path.exists(alt_path):
                            shutil.copy(alt_path, output_path)
                            break
                    else:
                        raise RuntimeError(f"Skinned FBX not found: {output_path}")

                skinning_time = time.time() - step_start
                print(f"[UniRigExtractRig] Skinning generated in {skinning_time:.2f}s")

            except subprocess.TimeoutExpired:
                raise RuntimeError(f"Skinning generation timed out (>{INFERENCE_TIMEOUT}s)")
            except Exception as e:
                print(f"[UniRigExtractRig] Skinning error: {e}")
                raise

            # Load the rigged mesh
            print(f"[UniRigExtractRig] Loading rigged mesh from {output_path}...")

            rigged_mesh = {
                "mesh": trimesh,
                "fbx_path": output_path,
                "has_skinning": True,
                "has_skeleton": True,
            }

            # Copy to a persistent location
            persistent_fbx = os.path.join(folder_paths.get_temp_directory(), f"rigged_mesh_{seed}.fbx")
            shutil.copy(output_path, persistent_fbx)
            rigged_mesh["fbx_path"] = persistent_fbx

            total_time = time.time() - total_start
            print(f"[UniRigExtractRig] Rig extraction complete!")
            print(f"[UniRigExtractRig] TOTAL TIME: {total_time:.2f}s")

            return (rigged_mesh,)


class UniRigExtractSkeleton:
    """
    Extract skeleton from mesh using UniRig (SIGGRAPH 2025) - Original implementation.

    Simple subprocess-based approach for skeleton extraction.
    Uses direct calls to run.py for reliable inference.
    Works on any mesh type: humans, animals, objects, cameras, etc.

    This is the original working implementation that uses subprocess for each inference.
    For faster inference with model caching, use UniRigExtractSkeletonNew.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH", {
                    "tooltip": "Input mesh to extract skeleton from"
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 4294967295,
                    "tooltip": "Random seed for skeleton generation variation"
                }),
            }
        }

    RETURN_TYPES = ("TRIMESH", "SKELETON", "IMAGE")
    RETURN_NAMES = ("normalized_mesh", "skeleton", "texture_preview")
    FUNCTION = "extract"
    CATEGORY = "UniRig"

    def extract(self, trimesh, seed):
        """Extract skeleton using UniRig via subprocess (original implementation)."""
        total_start = time.time()
        print(f"[UniRigExtractSkeleton] Starting skeleton extraction (original implementation)...")

        # Check if Blender is available
        if not BLENDER_EXE or not os.path.exists(BLENDER_EXE):
            raise RuntimeError(
                f"Blender not found. Please run install or install manually."
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
            print(f"[UniRigExtractSkeleton] Exporting mesh to {input_path}")
            print(f"[UniRigExtractSkeleton] Mesh has {len(trimesh.vertices)} vertices, {len(trimesh.faces)} faces")
            trimesh.export(input_path)
            export_time = time.time() - step_start
            print(f"[UniRigExtractSkeleton] Mesh exported in {export_time:.2f}s")

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
                "50000"  # target face count
            ]

            try:
                result = subprocess.run(
                    blender_cmd,
                    capture_output=True,
                    text=True,
                    timeout=BLENDER_TIMEOUT
                )
                if result.stdout:
                    print(f"[UniRigExtractSkeleton] Blender output:\n{result.stdout}")
                if result.stderr:
                    # Filter out Blender noise
                    stderr_lines = result.stderr.split('\n')
                    important_lines = [l for l in stderr_lines if 'error' in l.lower() or 'fail' in l.lower()]
                    if important_lines:
                        print(f"[UniRigExtractSkeleton] Blender warnings:\n" + '\n'.join(important_lines))

                # Check if NPZ was created
                if not os.path.exists(npz_path):
                    raise RuntimeError(f"Blender extraction failed: {npz_path} not created")

                blender_time = time.time() - step_start
                print(f"[UniRigExtractSkeleton] Mesh preprocessed in {blender_time:.2f}s: {npz_path}")

            except subprocess.TimeoutExpired:
                raise RuntimeError(f"Blender extraction timed out (>{BLENDER_TIMEOUT}s)")
            except Exception as e:
                print(f"[UniRigExtractSkeleton] Blender error: {e}")
                raise

            # Step 2: Run skeleton inference via subprocess
            step_start = time.time()
            print(f"[UniRigExtractSkeleton] Step 2: Running skeleton inference via subprocess...")
            run_cmd = [
                sys.executable, os.path.join(UNIRIG_PATH, "run.py"),
                "--task", os.path.join(UNIRIG_PATH, "configs/task/quick_inference_skeleton_articulationxl_ar_256.yaml"),
                "--seed", str(seed),
                "--input", input_path,
                "--output", output_path,
                "--npz_dir", tmpdir,
            ]

            print(f"[UniRigExtractSkeleton] Running: {' '.join(run_cmd)}")

            # Set up environment
            env = setup_subprocess_env()

            try:
                result = subprocess.run(
                    run_cmd,
                    cwd=UNIRIG_PATH,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=INFERENCE_TIMEOUT
                )
                if result.stdout:
                    print(f"[UniRigExtractSkeleton] Inference stdout:\n{result.stdout}")
                if result.stderr:
                    print(f"[UniRigExtractSkeleton] Inference stderr:\n{result.stderr}")

                if result.returncode != 0:
                    print(f"[UniRigExtractSkeleton] Inference failed with exit code {result.returncode}")
                    raise RuntimeError(f"Inference failed with exit code {result.returncode}")

                inference_time = time.time() - step_start
                print(f"[UniRigExtractSkeleton] Inference completed in {inference_time:.2f}s")

            except subprocess.TimeoutExpired:
                raise RuntimeError(f"Inference timed out (>{INFERENCE_TIMEOUT}s)")
            except Exception as e:
                print(f"[UniRigExtractSkeleton] Inference error: {e}")
                raise

            # Check output exists
            if not os.path.exists(output_path):
                tmpdir_contents = os.listdir(tmpdir)
                print(f"[UniRigExtractSkeleton] Output FBX not found: {output_path}")
                print(f"[UniRigExtractSkeleton] Temp directory contents: {tmpdir_contents}")
                raise RuntimeError(
                    f"UniRig did not generate output file: {output_path}\n"
                    f"Temp directory contents: {tmpdir_contents}\n"
                    f"Check stdout/stderr above for details"
                )

            print(f"[UniRigExtractSkeleton] Found output FBX: {output_path}")
            fbx_size = os.path.getsize(output_path)
            print(f"[UniRigExtractSkeleton] FBX file size: {fbx_size} bytes")

            # Save persistent copy
            persistent_fbx = os.path.join(folder_paths.get_temp_directory(), f"skeleton_{seed}.fbx")
            shutil.copy(output_path, persistent_fbx)
            print(f"[UniRigExtractSkeleton] Saved persistent FBX: {persistent_fbx}")

            # Step 3: Parse FBX output with Blender
            step_start = time.time()
            print(f"[UniRigExtractSkeleton] Step 3: Parsing FBX output with Blender...")

            skeleton_npz_path = os.path.join(tmpdir, "skeleton_data.npz")
            parse_cmd = [
                BLENDER_EXE,
                "--background",
                "--python", BLENDER_PARSE_SKELETON,
                "--",
                output_path,
                skeleton_npz_path
            ]

            try:
                result = subprocess.run(
                    parse_cmd,
                    capture_output=True,
                    text=True,
                    timeout=PARSE_TIMEOUT
                )
                if result.stdout:
                    print(f"[UniRigExtractSkeleton] Blender parse output:\n{result.stdout}")
                if result.stderr:
                    stderr_lines = result.stderr.split('\n')
                    important_lines = [l for l in stderr_lines if 'error' in l.lower() or 'fail' in l.lower()]
                    if important_lines:
                        print(f"[UniRigExtractSkeleton] Blender parse warnings:\n" + '\n'.join(important_lines))

                if not os.path.exists(skeleton_npz_path):
                    raise RuntimeError(f"Blender FBX parsing failed: {skeleton_npz_path} not created")

                parse_time = time.time() - step_start
                print(f"[UniRigExtractSkeleton] Skeleton parsed in {parse_time:.2f}s")

            except subprocess.TimeoutExpired:
                raise RuntimeError(f"Blender parsing timed out (>{PARSE_TIMEOUT}s)")
            except Exception as e:
                print(f"[UniRigExtractSkeleton] Blender parse error: {e}")
                raise

            # Load skeleton data from NPZ
            print(f"[UniRigExtractSkeleton] Loading skeleton data from NPZ...")
            skeleton_data = np.load(skeleton_npz_path, allow_pickle=True)
            print(f"[UniRigExtractSkeleton] NPZ contains keys: {list(skeleton_data.keys())}")

            # Extract skeleton components
            vertices = skeleton_data['vertices']
            edges = skeleton_data['edges']
            bone_names = skeleton_data['bone_names']
            bone_parents = skeleton_data['bone_parents']
            bone_to_head_vertex = skeleton_data['bone_to_head_vertex']

            print(f"[UniRigExtractSkeleton] Extracted {len(vertices)} joints, {len(bone_names)} bones")

            # Extract FBX mesh data
            mesh_vertices_fbx = skeleton_data.get('mesh_vertices_fbx', [])
            mesh_faces_fbx = skeleton_data.get('mesh_faces_fbx', [])
            mesh_vertex_normals_fbx = skeleton_data.get('mesh_vertex_normals_fbx')
            mesh_face_normals_fbx = skeleton_data.get('mesh_face_normals_fbx')

            if len(mesh_vertices_fbx) > 0:
                print(f"[UniRigExtractSkeleton] Found FBX mesh: {len(mesh_vertices_fbx)} vertices, {len(mesh_faces_fbx)} faces")

            # Load UV and texture data
            uv_coords = skeleton_data.get('uv_coords', np.array([]))
            uv_faces = skeleton_data.get('uv_faces', np.array([]))
            if len(uv_coords) > 0:
                print(f"[UniRigExtractSkeleton] Loaded UV coordinates: {len(uv_coords)} UVs")

            texture_data_base64 = str(skeleton_data.get('texture_data_base64', ''))
            texture_format = str(skeleton_data.get('texture_format', ''))
            texture_width = int(skeleton_data.get('texture_width', 0))
            texture_height = int(skeleton_data.get('texture_height', 0))
            material_name = str(skeleton_data.get('material_name', ''))

            if texture_data_base64:
                print(f"[UniRigExtractSkeleton] Loaded texture: {texture_width}x{texture_height} {texture_format} ({len(texture_data_base64)//1024}KB base64)")

            # Create normalized mesh from FBX data
            if len(mesh_vertices_fbx) > 0:
                print(f"[UniRigExtractSkeleton] Using FBX mesh as single source of truth")
                print(f"[UniRigExtractSkeleton] FBX mesh bounds: min={mesh_vertices_fbx.min(axis=0)}, max={mesh_vertices_fbx.max(axis=0)}")

                normalized_mesh = Trimesh(
                    vertices=mesh_vertices_fbx,
                    faces=mesh_faces_fbx,
                    vertex_normals=mesh_vertex_normals_fbx,
                    process=False
                )
                print(f"[UniRigExtractSkeleton] Created normalized mesh from FBX: {len(mesh_vertices_fbx)} vertices, {len(mesh_faces_fbx)} faces")
            else:
                print(f"[UniRigExtractSkeleton] No FBX mesh data, using original trimesh")
                normalized_mesh = trimesh

            # Save persistent NPZ
            persistent_npz = os.path.join(folder_paths.get_temp_directory(), f"skeleton_{seed}.npz")
            shutil.copy(skeleton_npz_path, persistent_npz)
            print(f"[UniRigExtractSkeleton] Saved skeleton NPZ to: {persistent_npz}")

            # Build skeleton dict
            skeleton = {
                "joints": vertices,
                "edges": edges,
                "names": bone_names,
                "parents": bone_parents,
                "bone_to_head_vertex": bone_to_head_vertex,
                "tails": None,  # Will be computed from edges if needed
                "mesh_vertices": mesh_vertices_fbx if len(mesh_vertices_fbx) > 0 else None,
                "mesh_faces": mesh_faces_fbx if len(mesh_faces_fbx) > 0 else None,
                "mesh_vertex_normals": mesh_vertex_normals_fbx,
                "mesh_face_normals": mesh_face_normals_fbx,
                "uv_coords": uv_coords if len(uv_coords) > 0 else None,
                "uv_faces": uv_faces if len(uv_faces) > 0 else None,
                "texture_data_base64": texture_data_base64 if texture_data_base64 else None,
                "texture_format": texture_format,
                "texture_width": texture_width,
                "texture_height": texture_height,
                "material_name": material_name,
                "fbx_path": persistent_fbx,
                "npz_path": persistent_npz,
            }

            print(f"[UniRigExtractSkeleton] Built skeleton dict with FBX mesh: {len(mesh_vertices_fbx)} vertices, {len(mesh_faces_fbx)} faces")
            if mesh_vertex_normals_fbx is not None:
                print(f"[UniRigExtractSkeleton] FBX mesh includes: {len(mesh_vertex_normals_fbx)} vertex normals, {len(mesh_face_normals_fbx)} face normals")
            print(f"[UniRigExtractSkeleton] Included hierarchy: {len(bone_names)} bones with parent relationships")

            # Create texture preview
            texture_preview = None
            if texture_data_base64:
                texture_preview = decode_texture_to_comfy_image(texture_data_base64)
                if texture_preview is not None:
                    print(f"[UniRigExtractSkeleton] Texture preview created: {texture_width}x{texture_height}")

            if texture_preview is None:
                texture_preview = create_placeholder_texture()

            total_time = time.time() - total_start
            print(f"[UniRigExtractSkeleton] Skeleton extraction complete!")
            print(f"[UniRigExtractSkeleton] TOTAL TIME: {total_time:.2f}s")

            return (normalized_mesh, skeleton, texture_preview)
