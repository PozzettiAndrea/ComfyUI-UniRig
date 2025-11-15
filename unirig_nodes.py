"""
UniRig nodes for ComfyUI

Provides state-of-the-art skeleton extraction and rigging using the UniRig framework.
Self-contained with bundled Blender and UniRig code.
"""

import os
import sys
import subprocess
import tempfile
import numpy as np
import trimesh
from trimesh import Trimesh
from pathlib import Path
import folder_paths
import time
import shutil
import glob


# Get paths relative to this file
NODE_DIR = Path(__file__).parent.absolute()
LIB_DIR = NODE_DIR / "lib"
UNIRIG_PATH = str(LIB_DIR / "unirig")
BLENDER_SCRIPT = str(LIB_DIR / "blender_extract.py")
BLENDER_PARSE_SKELETON = str(LIB_DIR / "blender_parse_skeleton.py")
BLENDER_EXTRACT_MESH_INFO = str(LIB_DIR / "blender_extract_mesh_info.py")

# Set up UniRig models directory in ComfyUI's models folder
# IMPORTANT: This must happen BEFORE any HuggingFace imports
UNIRIG_MODELS_DIR = Path(folder_paths.models_dir) / "unirig"
UNIRIG_MODELS_DIR.mkdir(parents=True, exist_ok=True)
(UNIRIG_MODELS_DIR / "hub").mkdir(parents=True, exist_ok=True)

# Set HuggingFace cache to use ComfyUI's models folder FIRST
os.environ['HF_HOME'] = str(UNIRIG_MODELS_DIR)
os.environ['TRANSFORMERS_CACHE'] = str(UNIRIG_MODELS_DIR / "transformers")
os.environ['HF_HUB_CACHE'] = str(UNIRIG_MODELS_DIR / "hub")

# Check if models exist in old HuggingFace cache and move them
try:
    old_hf_hub = Path.home() / ".cache" / "huggingface" / "hub"
    models_to_move = [
        ("models--VAST-AI--UniRig", "UniRig models (1.4GB)"),
        ("models--facebook--opt-350m", "OPT-350M transformer"),
    ]

    for model_dir, description in models_to_move:
        old_cache = old_hf_hub / model_dir
        new_cache = UNIRIG_MODELS_DIR / "hub" / model_dir

        if old_cache.exists() and not new_cache.exists():
            print(f"[UniRig] Found {description} in old cache: {old_cache}")
            print(f"[UniRig] Moving to ComfyUI models folder...")
            try:
                import shutil
                shutil.move(str(old_cache), str(new_cache))
                print(f"[UniRig] ✓ Moved {description}")
            except Exception as move_error:
                print(f"[UniRig] Warning: Could not move {description}: {move_error}")
                print(f"[UniRig] Manual move: mv '{old_cache}' '{new_cache}'")

    print(f"[UniRig] Models cache location: {UNIRIG_MODELS_DIR}")
except Exception as e:
    print(f"[UniRig] Warning during model setup: {e}")
    import traceback
    traceback.print_exc()

# Find Blender executable
BLENDER_DIR = LIB_DIR / "blender"
BLENDER_EXE = None
if BLENDER_DIR.exists():
    blender_bins = list(BLENDER_DIR.rglob("blender"))
    if blender_bins:
        BLENDER_EXE = str(blender_bins[0])
        print(f"[UniRig] Found Blender: {BLENDER_EXE}")

# Install Blender if not found
if not BLENDER_EXE:
    print("[UniRig] Blender not found, installing...")
    try:
        from .install_blender import install_blender
        BLENDER_EXE = install_blender(target_dir=BLENDER_DIR)
        if BLENDER_EXE:
            print(f"[UniRig] Blender installed: {BLENDER_EXE}")
        else:
            print("[UniRig] Warning: Blender installation failed")
    except Exception as e:
        print(f"[UniRig] Warning: Could not install Blender: {e}")

# Add local UniRig to path
if UNIRIG_PATH not in sys.path:
    sys.path.insert(0, UNIRIG_PATH)


def normalize_skeleton(vertices: np.ndarray) -> tuple:
    """
    Normalize skeleton vertices to [-1, 1] range.

    Returns:
        tuple: (normalized_vertices, normalization_params)
            normalization_params contains 'center' and 'scale' for denormalization
    """
    min_coords = vertices.min(axis=0)
    max_coords = vertices.max(axis=0)
    center = (min_coords + max_coords) / 2
    vertices_centered = vertices - center
    scale = (max_coords - min_coords).max() / 2

    if scale > 0:
        vertices_normalized = vertices_centered / scale
    else:
        vertices_normalized = vertices_centered

    normalization_params = {
        'center': center,
        'scale': scale,
        'min_coords': min_coords,
        'max_coords': max_coords
    }

    return vertices_normalized, normalization_params


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
                "seed": ("INT", {"default": 42, "min": 0, "max": 4294967295,  # numpy's max seed (2^32-1)
                               "tooltip": "Random seed for skeleton generation variation"}),
            },
            "optional": {
                "checkpoint": ("STRING", {
                    "default": "VAST-AI/UniRig",
                    "tooltip": "HuggingFace model ID or local path"
                }),
            }
        }

    RETURN_TYPES = ("TRIMESH", "SKELETON")
    RETURN_NAMES = ("normalized_mesh", "skeleton")
    FUNCTION = "extract"
    CATEGORY = "UniRig"

    def extract(self, trimesh, seed, checkpoint="VAST-AI/UniRig"):
        """Extract skeleton using UniRig."""
        total_start = time.time()
        print(f"[UniRigExtractSkeleton] ⏱️  Starting skeleton extraction...")

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
            # UniRig expects NPZ at: {npz_dir}/{basename}/raw_data.npz
            # Since input is "input.glb", basename is "input", so we need npz_dir to be tmpdir
            # and the NPZ will be at {tmpdir}/input/raw_data.npz
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
            print(f"[UniRigExtractSkeleton] ⏱️  Mesh exported in {export_time:.2f}s")

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
                    timeout=120
                )
                if result.stdout:
                    print(f"[UniRigExtractSkeleton] Blender output:\n{result.stdout}")
                if result.stderr:
                    # Blender always outputs some stuff to stderr, filter out noise
                    stderr_lines = result.stderr.split('\n')
                    important_lines = [l for l in stderr_lines if 'error' in l.lower() or 'fail' in l.lower()]
                    if important_lines:
                        print(f"[UniRigExtractSkeleton] Blender warnings:\n" + '\n'.join(important_lines))

                # Check if NPZ was created (ignore return code, Blender might segfault after saving)
                if not os.path.exists(npz_path):
                    raise RuntimeError(f"Blender extraction failed: {npz_path} not created")

                blender_time = time.time() - step_start
                print(f"[UniRigExtractSkeleton] ⏱️  Mesh preprocessed in {blender_time:.2f}s: {npz_path}")

            except subprocess.TimeoutExpired:
                raise RuntimeError("Blender extraction timed out (>2 minutes)")
            except Exception as e:
                print(f"[UniRigExtractSkeleton] Blender error: {e}")
                raise

            # Step 2: Run skeleton inference
            step_start = time.time()
            print(f"[UniRigExtractSkeleton] Step 2: Running skeleton inference...")
            run_cmd = [
                sys.executable, os.path.join(UNIRIG_PATH, "run.py"),
                "--task", os.path.join(UNIRIG_PATH, "configs/task/quick_inference_skeleton_articulationxl_ar_256.yaml"),
                "--seed", str(seed),
                "--input", input_path,
                "--output", output_path,
                "--npz_dir", tmpdir,  # This should match where NPZ is: tmpdir/input/raw_data.npz
            ]

            print(f"[UniRigExtractSkeleton] Running: {' '.join(run_cmd)}")
            print(f"[UniRigExtractSkeleton] Using Blender: {BLENDER_EXE}")

            # Set up environment with Blender path for internal FBX export
            env = os.environ.copy()
            env['BLENDER_EXE'] = BLENDER_EXE
            # Set PyOpenGL to use OSMesa for headless rendering (no EGL/X11 needed)
            env['PYOPENGL_PLATFORM'] = 'osmesa'
            # Ensure HuggingFace cache is set for subprocess
            if UNIRIG_MODELS_DIR:
                env['HF_HOME'] = str(UNIRIG_MODELS_DIR)
                env['TRANSFORMERS_CACHE'] = str(UNIRIG_MODELS_DIR / "transformers")
                env['HF_HUB_CACHE'] = str(UNIRIG_MODELS_DIR / "hub")
            print(f"[UniRigExtractSkeleton] Set BLENDER_EXE environment variable for FBX export")

            try:
                result = subprocess.run(
                    run_cmd,
                    cwd=UNIRIG_PATH,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=600  # 10 minutes
                )
                if result.stdout:
                    print(f"[UniRigExtractSkeleton] Inference stdout:\n{result.stdout}")
                if result.stderr:
                    print(f"[UniRigExtractSkeleton] Inference stderr:\n{result.stderr}")

                if result.returncode != 0:
                    print(f"[UniRigExtractSkeleton] ✗ Inference failed with exit code {result.returncode}")
                    raise RuntimeError(f"Inference failed with exit code {result.returncode}")

                inference_time = time.time() - step_start
                print(f"[UniRigExtractSkeleton] ⏱️  Inference completed in {inference_time:.2f}s")

            except subprocess.TimeoutExpired:
                raise RuntimeError("Inference timed out (>10 minutes)")
            except Exception as e:
                print(f"[UniRigExtractSkeleton] Inference error: {e}")
                raise

            # Load and parse FBX output
            if not os.path.exists(output_path):
                tmpdir_contents = os.listdir(tmpdir)
                print(f"[UniRigExtractSkeleton] ✗ Output FBX not found: {output_path}")
                print(f"[UniRigExtractSkeleton] Temp directory contents: {tmpdir_contents}")
                raise RuntimeError(
                    f"UniRig did not generate output file: {output_path}\n"
                    f"Temp directory contents: {tmpdir_contents}\n"
                    f"Check stdout/stderr above for details"
                )

            print(f"[UniRigExtractSkeleton] ✓ Found output FBX: {output_path}")
            fbx_size = os.path.getsize(output_path)
            print(f"[UniRigExtractSkeleton] FBX file size: {fbx_size} bytes")

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
                    timeout=60
                )
                if result.stdout:
                    print(f"[UniRigExtractSkeleton] Blender parse output:\n{result.stdout}")
                if result.stderr:
                    stderr_lines = result.stderr.split('\n')
                    important_lines = [l for l in stderr_lines if 'error' in l.lower() or 'fail' in l.lower()]
                    if important_lines:
                        print(f"[UniRigExtractSkeleton] Blender parse warnings:\n" + '\n'.join(important_lines))

                if not os.path.exists(skeleton_npz):
                    print(f"[UniRigExtractSkeleton] ✗ Skeleton NPZ not found: {skeleton_npz}")
                    raise RuntimeError(f"Skeleton parsing failed: {skeleton_npz} not created")

                parse_time = time.time() - step_start
                print(f"[UniRigExtractSkeleton] ⏱️  Skeleton parsed in {parse_time:.2f}s")

            except subprocess.TimeoutExpired:
                raise RuntimeError("Skeleton parsing timed out (>1 minute)")
            except Exception as e:
                print(f"[UniRigExtractSkeleton] Skeleton parse error: {e}")
                raise

            # Load skeleton data
            print(f"[UniRigExtractSkeleton] Loading skeleton data from NPZ...")
            skeleton_data = np.load(skeleton_npz, allow_pickle=True)
            print(f"[UniRigExtractSkeleton] NPZ contains keys: {list(skeleton_data.keys())}")
            all_joints = skeleton_data['vertices']  # All joint positions (already normalized by UniRig)
            edges = skeleton_data['edges']

            print(f"[UniRigExtractSkeleton] Extracted {len(all_joints)} joints, {len(edges)} bones")
            print(f"[UniRigExtractSkeleton] Skeleton already normalized by UniRig to range [{all_joints.min():.3f}, {all_joints.max():.3f}]")

            # Transform skeleton data to RawData format for skinning compatibility
            # Load the preprocessing data which contains mesh vertices/faces/normals
            preprocess_npz = os.path.join(tmpdir, "input", "raw_data.npz")
            if os.path.exists(preprocess_npz):
                preprocess_data = np.load(preprocess_npz, allow_pickle=True)
                mesh_vertices_original = preprocess_data['vertices']
                mesh_faces = preprocess_data['faces']
                vertex_normals = preprocess_data.get('vertex_normals', None)
                face_normals = preprocess_data.get('face_normals', None)
            else:
                # Fallback: use trimesh data
                mesh_vertices_original = np.array(trimesh.vertices, dtype=np.float32)
                mesh_faces = np.array(trimesh.faces, dtype=np.int32)
                vertex_normals = np.array(trimesh.vertex_normals, dtype=np.float32) if hasattr(trimesh, 'vertex_normals') else None
                face_normals = np.array(trimesh.face_normals, dtype=np.float32) if hasattr(trimesh, 'face_normals') else None

            # Normalize mesh to [-1, 1] using mesh bounds (matching what UniRig did during inference)
            # The skeleton from UniRig is already normalized using this same transform
            # This ensures mesh and skeleton share the same coordinate space
            mesh_bounds_min = mesh_vertices_original.min(axis=0)
            mesh_bounds_max = mesh_vertices_original.max(axis=0)
            mesh_center = (mesh_bounds_min + mesh_bounds_max) / 2
            mesh_extents = mesh_bounds_max - mesh_bounds_min
            mesh_scale = mesh_extents.max() / 2  # Max extent over 2 gives [-1, 1] range

            # Normalize mesh vertices to [-1, 1]
            mesh_vertices = (mesh_vertices_original - mesh_center) / mesh_scale

            print(f"[UniRigExtractSkeleton] Original mesh bounds: min={mesh_bounds_min}, max={mesh_bounds_max}")
            print(f"[UniRigExtractSkeleton] Mesh scale: {mesh_scale:.4f}, extents: {mesh_extents}")
            print(f"[UniRigExtractSkeleton] Normalized mesh bounds: min={mesh_vertices.min(axis=0)}, max={mesh_vertices.max(axis=0)}")

            # Create trimesh object from normalized mesh data
            # This is the preprocessed/decimated mesh, normalized to [-1, 1] (matching skeleton space)
            normalized_mesh = Trimesh(
                vertices=mesh_vertices,
                faces=mesh_faces,
                process=True
            )
            print(f"[UniRigExtractSkeleton] Created normalized mesh (matching skeleton normalization): {len(mesh_vertices)} vertices, {len(mesh_faces)} faces")

            # Build parents list from bone_parents (convert -1 to None)
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
                # bone_to_head_vertex tells us which vertex is the head of each bone
                if 'bone_to_head_vertex' in skeleton_data:
                    bone_to_head = skeleton_data['bone_to_head_vertex']
                    # Extract joint positions for each bone's head from all joints
                    bone_joints = np.array([all_joints[bone_to_head[i]] for i in range(num_bones)])
                else:
                    # Fallback: use first num_bones joints
                    bone_joints = all_joints[:num_bones]

                # Compute tails (end points of bones)
                tails = np.zeros((num_bones, 3))
                for i in range(num_bones):
                    # Find children bones
                    children = [j for j, p in enumerate(parents_list) if p == i]
                    if children:
                        # Tail is average of children bone head positions
                        tails[i] = np.mean([bone_joints[c] for c in children], axis=0)
                    else:
                        # Leaf bone: extend tail along parent direction
                        if parents_list[i] is not None:
                            direction = bone_joints[i] - bone_joints[parents_list[i]]
                            tails[i] = bone_joints[i] + direction * 0.3
                        else:
                            # Root bone with no children: extend upward
                            tails[i] = bone_joints[i] + np.array([0, 0.1, 0])

            else:
                # No hierarchy - create simple chain using all joints
                num_bones = len(all_joints)
                bone_joints = all_joints
                parents_list = [None] + list(range(num_bones-1))
                names_list = [f"bone_{i}" for i in range(num_bones)]

                # Compute tails for simple chain
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

            # Save as RawData NPZ for skinning phase (using bone joints, not all joints)
            persistent_npz = os.path.join(folder_paths.get_temp_directory(), f"skeleton_{seed}.npz")
            np.savez(
                persistent_npz,
                vertices=mesh_vertices,
                vertex_normals=vertex_normals,
                faces=mesh_faces,
                face_normals=face_normals,
                joints=bone_joints,  # Only bone head positions (normalized) for skinning
                tails=tails,
                parents=np.array(parents_list, dtype=object),  # Use object dtype to allow None values
                names=np.array(names_list, dtype=object),
                # Mesh bounds for denormalization
                mesh_bounds_min=mesh_bounds_min,
                mesh_bounds_max=mesh_bounds_max,
                mesh_center=mesh_center,
                mesh_scale=mesh_scale,
                # Legacy fields (for compatibility)
                skin=None,
                no_skin=None,
                matrix_local=None,
                path=None,
                cls=None
            )
            print(f"[UniRigExtractSkeleton] Saved skeleton NPZ to: {persistent_npz}")
            print(f"[UniRigExtractSkeleton] Bone data: {len(bone_joints)} joints, {len(tails)} tails")
            print(f"[UniRigExtractSkeleton] DEBUG - Mesh vertices (NORMALIZED) bounds: {mesh_vertices.min(axis=0)} to {mesh_vertices.max(axis=0)}")
            print(f"[UniRigExtractSkeleton] DEBUG - Skeleton joints (NORMALIZED) bounds: {bone_joints.min(axis=0)} to {bone_joints.max(axis=0)}")
            print(f"[UniRigExtractSkeleton] DEBUG - Skeleton tails (NORMALIZED) bounds: {tails.min(axis=0)} to {tails.max(axis=0)}")

            # Debug: Y position of vertex with highest Z value
            max_z_idx = mesh_vertices[:, 2].argmax()
            max_z_vertex = mesh_vertices[max_z_idx]
            print(f"[UniRigExtractSkeleton] DEBUG - Mesh vertex with max Z: position={max_z_vertex}, Z={max_z_vertex[2]:.6f}, Y={max_z_vertex[1]:.6f}")

            max_z_joint_idx = bone_joints[:, 2].argmax()
            max_z_joint = bone_joints[max_z_joint_idx]
            print(f"[UniRigExtractSkeleton] DEBUG - Skeleton joint with max Z: position={max_z_joint}, Z={max_z_joint[2]:.6f}, Y={max_z_joint[1]:.6f}")

            # Build skeleton dict with ALL data (no need to load NPZ later)
            skeleton = {
                # Visualization data (all joints for SkeletonToMesh)
                "vertices": all_joints,  # All joint positions for visualization
                "edges": edges,  # Edges reference all_joints indices

                # Core skeleton data (for skinning)
                "joints": bone_joints,  # Bone head positions (normalized)
                "tails": tails,  # Bone tail positions (normalized)
                "names": names_list,  # Bone names
                "parents": parents_list,  # Parent bone indices

                # Mesh data (for skinning)
                "mesh_vertices": mesh_vertices,  # Normalized mesh vertices
                "mesh_faces": mesh_faces,
                "mesh_vertex_normals": vertex_normals,
                "mesh_face_normals": face_normals,

                # Normalization metadata
                "mesh_bounds_min": mesh_bounds_min,
                "mesh_bounds_max": mesh_bounds_max,
                "mesh_center": mesh_center,
                "mesh_scale": mesh_scale,
                "is_normalized": True,  # Flag indicating data is normalized

                # Legacy/compatibility
                "skeleton_npz_path": persistent_npz,  # For backward compat
                "bone_names": names_list,  # Legacy field name
                "bone_parents": parents_list,  # Legacy field name
            }

            if 'bone_to_head_vertex' in skeleton_data:
                skeleton['bone_to_head_vertex'] = skeleton_data['bone_to_head_vertex'].tolist()

            print(f"[UniRigExtractSkeleton] Included hierarchy: {len(names_list)} bones with parent relationships")
            print(f"[UniRigExtractSkeleton] Skeleton dict contains all data (no NPZ loading needed)")

            total_time = time.time() - total_start
            print(f"[UniRigExtractSkeleton] ✓✓✓ Skeleton extraction complete! ✓✓✓")
            print(f"[UniRigExtractSkeleton] ⏱️  TOTAL TIME: {total_time:.2f}s")
            return (normalized_mesh, skeleton)

    def _extract_bones_from_fbx(self, fbx_mesh):
        """
        Extract bone structure from FBX.

        FBX armature structure is complex. For now, we extract:
        - Joint positions from mesh vertices
        - Bone connections from edge structure
        """
        # If the FBX has a scene graph with bones, extract from there
        # For now, simplified: use mesh structure as proxy

        if hasattr(fbx_mesh, 'vertices'):
            vertices = np.array(fbx_mesh.vertices)

            # Try to extract edges if available
            if hasattr(fbx_mesh, 'edges'):
                edges = np.array(fbx_mesh.edges)
            elif hasattr(fbx_mesh, 'faces') and len(fbx_mesh.faces) > 0:
                # Extract edges from faces
                faces = fbx_mesh.faces
                edges_set = set()
                for face in faces:
                    for i in range(len(face)):
                        edge = tuple(sorted([face[i], face[(i+1) % len(face)]]))
                        edges_set.add(edge)
                edges = np.array(list(edges_set))
            else:
                # Create minimal spanning tree from vertices
                from scipy.spatial import cKDTree
                tree = cKDTree(vertices)
                edges = []
                for i in range(len(vertices) - 1):
                    # Connect to nearest unconnected neighbor
                    dists, indices = tree.query(vertices[i], k=2)
                    edges.append([i, indices[1]])
                edges = np.array(edges)
        else:
            raise ValueError("Cannot extract bones from FBX: no vertices found")

        return vertices, edges


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
        print(f"[UniRigExtractRig] ⏱️  Starting full rig extraction...")

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
            print(f"[UniRigExtractRig] ⏱️  Mesh exported in {export_time:.2f}s")

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
                "50000"  # target face count
            ]

            try:
                result = subprocess.run(blender_cmd, capture_output=True, text=True, timeout=120)
                if result.stdout:
                    print(f"[UniRigExtractRig] Blender output:\n{result.stdout}")

                if not os.path.exists(npz_path):
                    raise RuntimeError(f"Blender extraction failed: {npz_path} not created")

                blender_time = time.time() - step_start
                print(f"[UniRigExtractRig] ⏱️  Mesh preprocessed in {blender_time:.2f}s")

            except subprocess.TimeoutExpired:
                raise RuntimeError("Blender extraction timed out (>2 minutes)")
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
                "--output", skeleton_fbx_path,  # FBX output path (required)
                "--npz_dir", tmpdir,  # NPZ will be written to {npz_dir}/input/predict_skeleton.npz
            ]

            env = os.environ.copy()
            env['BLENDER_EXE'] = BLENDER_EXE
            # Set PyOpenGL to use OSMesa for headless rendering (no EGL/X11 needed)
            env['PYOPENGL_PLATFORM'] = 'osmesa'
            # Ensure HuggingFace cache is set for subprocess
            if UNIRIG_MODELS_DIR:
                env['HF_HOME'] = str(UNIRIG_MODELS_DIR)
                env['TRANSFORMERS_CACHE'] = str(UNIRIG_MODELS_DIR / "transformers")
                env['HF_HUB_CACHE'] = str(UNIRIG_MODELS_DIR / "hub")

            try:
                result = subprocess.run(skeleton_cmd, cwd=UNIRIG_PATH, env=env, capture_output=True, text=True, timeout=600)  # 10 minutes
                if result.stdout:
                    print(f"[UniRigExtractRig] Skeleton stdout:\n{result.stdout}")
                if result.stderr:
                    print(f"[UniRigExtractRig] Skeleton stderr:\n{result.stderr}")

                if result.returncode != 0:
                    raise RuntimeError(f"Skeleton generation failed with exit code {result.returncode}")

                # The skeleton NPZ might not be auto-generated, so create it from the FBX
                # Check if FBX was created
                if not os.path.exists(skeleton_fbx_path):
                    raise RuntimeError(f"Skeleton FBX not created: {skeleton_fbx_path}")

                print(f"[UniRigExtractRig] ✓ Skeleton FBX created: {skeleton_fbx_path}")

                # Parse the FBX to create predict_skeleton.npz using Blender
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
                    result = subprocess.run(parse_cmd, capture_output=True, text=True, timeout=60)
                    if result.stdout:
                        print(f"[UniRigExtractRig] Blender parse output:\n{result.stdout}")

                    if not os.path.exists(skeleton_npz_path):
                        raise RuntimeError(f"Failed to create skeleton NPZ: {skeleton_npz_path}")

                    print(f"[UniRigExtractRig] ✓ Skeleton NPZ created: {skeleton_npz_path}")

                except subprocess.TimeoutExpired:
                    raise RuntimeError("Skeleton NPZ creation timed out (>1 minute)")
                except Exception as e:
                    print(f"[UniRigExtractRig] Skeleton NPZ creation error: {e}")
                    raise

                skeleton_time = time.time() - step_start
                print(f"[UniRigExtractRig] ⏱️  Skeleton generated in {skeleton_time:.2f}s")

            except subprocess.TimeoutExpired:
                raise RuntimeError("Skeleton generation timed out (>10 minutes)")
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
                result = subprocess.run(skin_cmd, cwd=UNIRIG_PATH, env=env, capture_output=True, text=True, timeout=600)  # 10 minutes
                if result.stdout:
                    print(f"[UniRigExtractRig] Skinning stdout:\n{result.stdout}")
                if result.stderr:
                    print(f"[UniRigExtractRig] Skinning stderr:\n{result.stderr}")

                if result.returncode != 0:
                    raise RuntimeError(f"Skinning generation failed with exit code {result.returncode}")

                # Look for the output FBX in results directory or tmpdir
                if not os.path.exists(output_path):
                    alt_paths = [
                        os.path.join(tmpdir, "results", "result_fbx.fbx"),
                        os.path.join(tmpdir, "input", "result_fbx.fbx"),
                    ]
                    for alt_path in alt_paths:
                        if os.path.exists(alt_path):
                            import shutil
                            shutil.copy(alt_path, output_path)
                            break
                    else:
                        raise RuntimeError(f"Skinned FBX not found: {output_path}")

                skinning_time = time.time() - step_start
                print(f"[UniRigExtractRig] ⏱️  Skinning generated in {skinning_time:.2f}s")

            except subprocess.TimeoutExpired:
                raise RuntimeError("Skinning generation timed out (>10 minutes)")
            except Exception as e:
                print(f"[UniRigExtractRig] Skinning error: {e}")
                raise

            # Load the rigged mesh (FBX with skeleton and skinning)
            print(f"[UniRigExtractRig] Loading rigged mesh from {output_path}...")

            # Return as a rigged mesh dict containing the FBX path and the original mesh
            rigged_mesh = {
                "mesh": trimesh,
                "fbx_path": output_path,
                "has_skinning": True,
                "has_skeleton": True,
            }

            # Copy to a persistent location in the temp directory so it doesn't get deleted
            persistent_fbx = os.path.join(folder_paths.get_temp_directory(), f"rigged_mesh_{seed}.fbx")
            import shutil
            shutil.copy(output_path, persistent_fbx)
            rigged_mesh["fbx_path"] = persistent_fbx

            total_time = time.time() - total_start
            print(f"[UniRigExtractRig] ✓✓✓ Rig extraction complete! ✓✓✓")
            print(f"[UniRigExtractRig] ⏱️  TOTAL TIME: {total_time:.2f}s")

            return (rigged_mesh,)


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
        print(f"[UniRigApplySkinning] ⏱️  Starting skinning application...")

        # Check if Blender is available
        if not BLENDER_EXE or not os.path.exists(BLENDER_EXE):
            raise RuntimeError(f"Blender not found. Please run install_blender.py or install manually.")

        if not os.path.exists(UNIRIG_PATH):
            raise RuntimeError(f"UniRig not found at {UNIRIG_PATH}")

        # Get skeleton NPZ path
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
            print(f"[UniRigApplySkinning] ⏱️  Mesh exported in {export_time:.2f}s")

            # Denormalize skeleton and save to expected location
            # Skinning expects predict_skeleton.npz in npz_dir/input/ subdirectory
            # (where "input" matches the input filename without extension)
            input_subdir = os.path.join(tmpdir, "input")
            os.makedirs(input_subdir, exist_ok=True)
            predict_skeleton_path = os.path.join(input_subdir, "predict_skeleton.npz")

            # Load normalized skeleton
            print(f"[UniRigApplySkinning] Loading and denormalizing skeleton...")
            skeleton_data = np.load(skeleton_npz_path, allow_pickle=True)

            # Get mesh bounds for denormalization
            mesh_center = skeleton.get("mesh_center")
            mesh_scale = skeleton.get("mesh_scale")

            if mesh_center is None or mesh_scale is None:
                print(f"[UniRigApplySkinning] WARNING: Missing mesh bounds in skeleton, using normalized coordinates")
                import shutil
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

            env = os.environ.copy()
            env['BLENDER_EXE'] = BLENDER_EXE
            # Set PyOpenGL to use OSMesa for headless rendering (no EGL/X11 needed)
            env['PYOPENGL_PLATFORM'] = 'osmesa'
            # Ensure HuggingFace cache is set for subprocess
            if UNIRIG_MODELS_DIR:
                env['HF_HOME'] = str(UNIRIG_MODELS_DIR)
                env['TRANSFORMERS_CACHE'] = str(UNIRIG_MODELS_DIR / "transformers")
                env['HF_HUB_CACHE'] = str(UNIRIG_MODELS_DIR / "hub")

            try:
                result = subprocess.run(skin_cmd, cwd=UNIRIG_PATH, env=env, capture_output=True, text=True, timeout=600)  # 10 minutes
                if result.stdout:
                    print(f"[UniRigApplySkinning] Skinning stdout:\n{result.stdout}")
                if result.stderr:
                    print(f"[UniRigApplySkinning] Skinning stderr:\n{result.stderr}")

                if result.returncode != 0:
                    print(f"[UniRigApplySkinning] ✗ Skinning failed with return code: {result.returncode}")
                    raise RuntimeError(f"Skinning generation failed with exit code {result.returncode}")

                print(f"[UniRigApplySkinning] ✓ Skinning inference completed successfully")

                # Look for the output FBX in results directory or tmpdir
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
                            print(f"[UniRigApplySkinning] ✓ Found FBX at: {alt_path}")
                            shutil.copy(alt_path, output_path)
                            found = True
                            break

                    if not found:
                        print(f"[UniRigApplySkinning] ✗ FBX not found in any expected location")
                        raise RuntimeError(f"Skinned FBX not found: {output_path}")
                else:
                    print(f"[UniRigApplySkinning] ✓ Found FBX at primary location: {output_path}")

                skinning_time = time.time() - step_start
                print(f"[UniRigApplySkinning] ⏱️  Skinning applied in {skinning_time:.2f}s")

            except subprocess.TimeoutExpired:
                raise RuntimeError("Skinning generation timed out (>10 minutes)")
            except Exception as e:
                print(f"[UniRigApplySkinning] Skinning error: {e}")
                raise

            # Load the rigged mesh (FBX with skeleton and skinning)
            print(f"[UniRigApplySkinning] Loading rigged mesh from {output_path}...")

            # Return as a rigged mesh dict containing the FBX path and the original mesh
            rigged_mesh = {
                "mesh": trimesh,
                "fbx_path": output_path,
                "has_skinning": True,
                "has_skeleton": True,
            }

            # Copy to a persistent location in the temp directory so it doesn't get deleted
            persistent_fbx = os.path.join(folder_paths.get_temp_directory(), f"rigged_mesh_skinning_{int(time.time())}.fbx")
            shutil.copy(output_path, persistent_fbx)
            rigged_mesh["fbx_path"] = persistent_fbx

            total_time = time.time() - total_start
            print(f"[UniRigApplySkinning] ✓✓✓ Skinning application complete! ✓✓✓")
            print(f"[UniRigApplySkinning] ⏱️  TOTAL TIME: {total_time:.2f}s")

            return (rigged_mesh,)


class UniRigSaveSkeleton:
    """
    Save skeleton to file in various formats.

    Supports:
    - OBJ, PLY: Simple line mesh visualization
    - FBX, GLB: Full hierarchy (if available) for animation
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "skeleton": ("SKELETON",),
                "filename": ("STRING", {"default": "skeleton.fbx"}),
                "format": (["fbx", "glb", "obj", "ply"], {"default": "fbx"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "UniRig"

    def save(self, skeleton, filename, format):
        """Save skeleton to file."""
        print(f"[UniRigSaveSkeleton] Saving skeleton to {filename} as {format.upper()}...")

        # Get ComfyUI output directory
        output_dir = folder_paths.get_output_directory()
        filepath = os.path.join(output_dir, filename)

        # Ensure filename has correct extension
        if not filepath.endswith(f'.{format}'):
            filepath = os.path.splitext(filepath)[0] + f'.{format}'

        vertices = skeleton['vertices']
        edges = skeleton['edges']

        has_hierarchy = 'bone_names' in skeleton and 'bone_parents' in skeleton

        if format in ['fbx', 'glb']:
            if not has_hierarchy:
                print(f"[UniRigSaveSkeleton] Warning: Skeleton has no hierarchy data. FBX/GLB will only contain line mesh.")
                print(f"[UniRigSaveSkeleton] For full animation support, ensure the skeleton was extracted with UniRig.")
                self._save_line_mesh(vertices, edges, filepath, format)
            else:
                self._save_fbx_with_hierarchy(skeleton, filepath, format)
        else:
            # OBJ/PLY: save as line mesh
            self._save_line_mesh(vertices, edges, filepath, format)

        print(f"[UniRigSaveSkeleton] ✓ Saved to: {filepath}")
        return {}

    def _save_line_mesh(self, vertices, edges, filepath, format):
        """Save skeleton as a simple line mesh (OBJ or PLY)."""
        import trimesh

        # Create line segments from edges
        # For trimesh, we create a Path3D object
        entities = []
        for edge in edges:
            entities.append(trimesh.path.entities.Line([edge[0], edge[1]]))

        path = trimesh.path.Path3D(
            vertices=vertices,
            entities=entities
        )

        # Export
        path.export(filepath, file_type=format)

    def _save_fbx_with_hierarchy(self, skeleton, filepath, format):
        """Save skeleton with full hierarchy using Blender."""
        import pickle
        import tempfile

        vertices = skeleton['vertices']
        edges = skeleton['edges']
        bone_names = skeleton['bone_names']
        bone_parents = skeleton['bone_parents']
        bone_to_head_vertex = skeleton['bone_to_head_vertex']

        # Denormalize vertices (they're in [-1, 1] range)
        # Use mesh bounds from skeleton to restore original scale
        mesh_center = skeleton.get('mesh_center')
        mesh_scale = skeleton.get('mesh_scale')

        if mesh_center is not None and mesh_scale is not None:
            vertices_denorm = vertices * mesh_scale + mesh_center
            print(f"[UniRigSaveSkeleton] Denormalizing skeleton:")
            print(f"  Mesh center: {mesh_center}")
            print(f"  Mesh scale: {mesh_scale}")
            print(f"  Vertices before: {vertices.min(axis=0)} to {vertices.max(axis=0)}")
            print(f"  Vertices after: {vertices_denorm.min(axis=0)} to {vertices_denorm.max(axis=0)}")
        else:
            print(f"[UniRigSaveSkeleton] WARNING: Missing mesh bounds, skeleton may be incorrectly scaled")
            vertices_denorm = vertices.copy()

        # Reconstruct joint positions for each bone
        # bone_to_head_vertex maps bone index to the vertex index of its head
        joints = []
        for bone_idx in range(len(bone_names)):
            head_vertex_idx = bone_to_head_vertex[bone_idx]
            joints.append(vertices_denorm[head_vertex_idx])

        joints = np.array(joints, dtype=np.float32)

        # Prepare data for Blender export (plain Python types for pickle)
        data = {
            'joints': joints.tolist(),
            'parents': bone_parents,
            'names': bone_names,
            'vertices': None,  # No mesh
            'faces': None,
            'skin': None,
            'tails': None,  # Will be auto-calculated
            'group_per_vertex': -1,
            'do_not_normalize': True,
        }

        # Save to temporary pickle file
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
            pickle_path = f.name
            pickle.dump(data, f)

        try:
            # Find Blender executable and wrapper script
            current_dir = os.path.dirname(os.path.abspath(__file__))
            wrapper_script = os.path.join(current_dir, 'lib', 'blender_export_fbx.py')

            if not os.path.exists(wrapper_script):
                raise RuntimeError(f"Blender export script not found: {wrapper_script}")

            if not os.path.exists(BLENDER_EXE):
                raise RuntimeError(f"Blender executable not found: {BLENDER_EXE}")

            # Determine output format
            if format == 'glb':
                # For GLB, first export to FBX, then convert
                temp_fbx = filepath.replace('.glb', '_temp.fbx')
                output_path = temp_fbx
            else:
                output_path = filepath

            # Build Blender command
            cmd = [
                BLENDER_EXE,
                '--background',
                '--python', wrapper_script,
                '--',
                pickle_path,
                output_path,
                '--extrude_size=0.03',
            ]

            # Run Blender
            print(f"[UniRigSaveSkeleton] Running Blender to export {format.upper()}...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            if result.returncode != 0:
                print(f"[UniRigSaveSkeleton] Blender error: {result.stderr}")
                raise RuntimeError(f"FBX export failed with return code {result.returncode}")

            if not os.path.exists(output_path):
                raise RuntimeError(f"Export completed but output file not found: {output_path}")

            # Convert FBX to GLB if needed
            if format == 'glb':
                print(f"[UniRigSaveSkeleton] Converting FBX to GLB...")
                import trimesh
                mesh = trimesh.load(temp_fbx)
                mesh.export(filepath)
                os.remove(temp_fbx)

        finally:
            # Clean up pickle file
            if os.path.exists(pickle_path):
                os.unlink(pickle_path)


class UniRigSaveRiggedMesh:
    """
    Save rigged mesh (with skeleton and skinning weights) to file.

    Supports FBX and GLB formats for animation software.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rigged_mesh": ("RIGGED_MESH",),
                "filename": ("STRING", {"default": "rigged_mesh.fbx"}),
                "format": (["fbx", "glb"], {"default": "fbx"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "UniRig"

    def save(self, rigged_mesh, filename, format):
        """Save rigged mesh to file."""
        print(f"[UniRigSaveRiggedMesh] Saving rigged mesh to {filename} as {format.upper()}...")

        # Get ComfyUI output directory
        output_dir = folder_paths.get_output_directory()
        filepath = os.path.join(output_dir, filename)

        # Ensure filename has correct extension
        if not filepath.endswith(f'.{format}'):
            filepath = os.path.splitext(filepath)[0] + f'.{format}'

        # Get the FBX file path from the rigged mesh
        source_fbx = rigged_mesh.get("fbx_path")
        if not source_fbx or not os.path.exists(source_fbx):
            raise RuntimeError(f"Rigged mesh FBX not found: {source_fbx}")

        # Copy or convert the file
        if format == "fbx":
            # Direct copy for FBX
            import shutil
            shutil.copy(source_fbx, filepath)
            print(f"[UniRigSaveRiggedMesh] ✓ Saved FBX to: {filepath}")
        elif format == "glb":
            # Convert FBX to GLB using trimesh
            print(f"[UniRigSaveRiggedMesh] Converting FBX to GLB...")
            try:
                import trimesh
                scene = trimesh.load(source_fbx)
                scene.export(filepath)
                print(f"[UniRigSaveRiggedMesh] ✓ Saved GLB to: {filepath}")
            except Exception as e:
                print(f"[UniRigSaveRiggedMesh] Warning: GLB conversion failed, saving as FBX: {e}")
                import shutil
                shutil.copy(source_fbx, filepath.replace('.glb', '.fbx'))
                filepath = filepath.replace('.glb', '.fbx')
                print(f"[UniRigSaveRiggedMesh] ✓ Saved FBX to: {filepath}")

        file_size = os.path.getsize(filepath)
        print(f"[UniRigSaveRiggedMesh] File size: {file_size / 1024:.2f} KB")
        print(f"[UniRigSaveRiggedMesh] Has skinning: {rigged_mesh.get('has_skinning', False)}")
        print(f"[UniRigSaveRiggedMesh] Has skeleton: {rigged_mesh.get('has_skeleton', False)}")

        return {}


class UniRigLoadRiggedMesh:
    """
    Load a rigged FBX file from disk.

    Loads existing FBX files with rigging/skeleton data, allowing you to
    preview and work with pre-rigged models.
    """

    @classmethod
    def INPUT_TYPES(cls):
        # Get list of FBX files from input directory
        input_dir = folder_paths.get_input_directory()
        fbx_files = []

        # Search for FBX files in input directory and subdirectories
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith('.fbx'):
                    # Get relative path from input directory
                    rel_path = os.path.relpath(os.path.join(root, file), input_dir)
                    fbx_files.append(rel_path)

        # Sort alphabetically
        fbx_files.sort()

        if not fbx_files:
            fbx_files = ["No FBX files found in input directory"]

        return {
            "required": {
                "fbx_file": (fbx_files,),
            },
        }

    RETURN_TYPES = ("RIGGED_MESH", "STRING")
    RETURN_NAMES = ("rigged_mesh", "info")
    FUNCTION = "load"
    CATEGORY = "unirig"

    def load(self, fbx_file):
        """
        Load an FBX file and return it as a RIGGED_MESH.

        Args:
            fbx_file: Filename of FBX file in input directory

        Returns:
            tuple: (rigged_mesh, info_string)
        """
        print(f"[UniRigLoadRiggedMesh] Loading FBX file: {fbx_file}")

        # Handle case where no FBX files exist
        if fbx_file == "No FBX files found in input directory":
            raise RuntimeError("No FBX files found in ComfyUI/input directory. Please add an FBX file first.")

        # Get full path
        input_dir = folder_paths.get_input_directory()
        fbx_path = os.path.join(input_dir, fbx_file)

        if not os.path.exists(fbx_path):
            raise RuntimeError(f"FBX file not found: {fbx_path}")

        # Copy to temp directory with unique name to avoid conflicts
        temp_dir = folder_paths.get_temp_directory()
        temp_fbx = os.path.join(temp_dir, f"loaded_rigged_{int(time.time())}_{os.path.basename(fbx_file)}")
        shutil.copy(fbx_path, temp_fbx)

        print(f"[UniRigLoadRiggedMesh] Copied to temp: {temp_fbx}")

        # Use Blender to extract mesh info (trimesh doesn't support FBX)
        mesh_info = {}
        try:
            # Use the global BLENDER_EXE variable or environment variable
            blender_exe = os.environ.get('UNIRIG_BLENDER_EXECUTABLE')
            if not blender_exe:
                blender_exe = BLENDER_EXE

            if blender_exe and os.path.exists(blender_exe):
                # Create temp output for mesh data
                mesh_npz = os.path.join(temp_dir, f"mesh_info_{int(time.time())}.npz")

                cmd = [
                    blender_exe,
                    "--background",
                    "--python", BLENDER_EXTRACT_MESH_INFO,
                    "--",
                    fbx_path,
                    mesh_npz
                ]

                print(f"[UniRigLoadRiggedMesh] Extracting mesh info with Blender...")
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

                if os.path.exists(mesh_npz):
                    data = np.load(mesh_npz, allow_pickle=True)

                    total_vertices = int(data.get('total_vertices', 0))
                    total_faces = int(data.get('total_faces', 0))
                    mesh_count = int(data.get('mesh_count', 0))
                    bbox_min = data.get('bbox_min', np.array([0, 0, 0]))
                    bbox_max = data.get('bbox_max', np.array([0, 0, 0]))
                    extents = data.get('extents', np.array([0, 0, 0]))

                    mesh_info = {
                        "type": "Scene" if mesh_count > 1 else "Mesh",
                        "mesh_count": mesh_count,
                        "total_vertices": total_vertices,
                        "total_faces": total_faces,
                        "bbox_min": bbox_min.tolist(),
                        "bbox_max": bbox_max.tolist(),
                        "extents": extents.tolist()
                    }

                    # Clean up temp file
                    os.remove(mesh_npz)

                    print(f"[UniRigLoadRiggedMesh] Mesh: {mesh_count} objects, {total_vertices} verts, {total_faces} faces")
                    print(f"[UniRigLoadRiggedMesh] Extents: {extents.tolist()}")
                else:
                    print(f"[UniRigLoadRiggedMesh] Mesh info extraction failed")
                    mesh_info = {"type": "Unknown", "note": "Extraction failed"}
            else:
                print(f"[UniRigLoadRiggedMesh] Blender not available for mesh info")
                mesh_info = {"type": "Unknown", "note": "Blender not available"}

        except Exception as e:
            print(f"[UniRigLoadRiggedMesh] Could not parse mesh geometry: {e}")
            mesh_info = {"type": "Unknown", "error": str(e)}

        # Parse FBX with Blender to get skeleton info (if available)
        skeleton_info = {}
        try:
            # Use the global BLENDER_EXE variable or environment variable
            blender_exe = os.environ.get('UNIRIG_BLENDER_EXECUTABLE')
            if not blender_exe:
                blender_exe = BLENDER_EXE

            if blender_exe and os.path.exists(blender_exe):
                # Create temp output for skeleton data
                skeleton_npz = os.path.join(temp_dir, f"skeleton_info_{int(time.time())}.npz")

                cmd = [
                    blender_exe,
                    "--background",
                    "--python", BLENDER_PARSE_SKELETON,
                    "--",
                    fbx_path,
                    skeleton_npz
                ]

                print(f"[UniRigLoadRiggedMesh] Parsing skeleton with Blender...")
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

                if os.path.exists(skeleton_npz):
                    data = np.load(skeleton_npz, allow_pickle=True)

                    # Get bone data (key is 'bone_names' not 'names')
                    num_bones = len(data.get('bone_names', []))
                    bone_names = [str(name) for name in data.get('bone_names', [])]

                    # Get vertices for skeleton extents
                    vertices = data.get('vertices', np.array([]))
                    skeleton_extents = None
                    if len(vertices) > 0:
                        min_coords = vertices.min(axis=0)
                        max_coords = vertices.max(axis=0)
                        skeleton_extents = (max_coords - min_coords).tolist()

                    skeleton_info = {
                        "num_bones": num_bones,
                        "bone_names": bone_names[:10],  # First 10 bones
                        "has_skeleton": num_bones > 0,
                        "skeleton_extents": skeleton_extents
                    }

                    # Clean up temp file
                    os.remove(skeleton_npz)

                    print(f"[UniRigLoadRiggedMesh] Found {num_bones} bones")
                    if skeleton_extents:
                        print(f"[UniRigLoadRiggedMesh] Skeleton extents: {skeleton_extents}")
                else:
                    skeleton_info = {"has_skeleton": False, "note": "No armature found"}
                    print(f"[UniRigLoadRiggedMesh] No skeleton data found")
            else:
                skeleton_info = {"has_skeleton": "unknown", "note": "Blender not available"}

        except Exception as e:
            print(f"[UniRigLoadRiggedMesh] Could not parse skeleton: {e}")
            skeleton_info = {"has_skeleton": "unknown", "error": str(e)}

        # Create rigged mesh structure
        rigged_mesh = {
            "mesh": None,  # Mesh data not needed (viewer loads FBX directly)
            "fbx_path": temp_fbx,
            "has_skinning": skeleton_info.get("has_skeleton", False),
            "has_skeleton": skeleton_info.get("has_skeleton", False),
        }

        # Create info string
        file_size = os.path.getsize(fbx_path)
        info_lines = [
            f"File: {os.path.basename(fbx_file)}",
            f"Size: {file_size / 1024:.1f} KB",
            "",
            "Mesh Info:",
            f"  Type: {mesh_info.get('type', 'Unknown')}",
            f"  Meshes: {mesh_info.get('mesh_count', 'Unknown')}",
            f"  Vertices: {mesh_info.get('total_vertices', 'Unknown'):,}" if isinstance(mesh_info.get('total_vertices'), int) else f"  Vertices: Unknown",
            f"  Faces: {mesh_info.get('total_faces', 'Unknown'):,}" if isinstance(mesh_info.get('total_faces'), int) else f"  Faces: Unknown",
        ]

        # Add bounding box and extents if available
        if 'extents' in mesh_info and mesh_info['extents']:
            extents = mesh_info['extents']
            info_lines.append(f"  Mesh Size: [{extents[0]:.3f}, {extents[1]:.3f}, {extents[2]:.3f}]")

        if 'bbox_min' in mesh_info and 'bbox_max' in mesh_info:
            bbox_min = mesh_info['bbox_min']
            bbox_max = mesh_info['bbox_max']
            info_lines.append(f"  Bounding Box:")
            info_lines.append(f"    Min: [{bbox_min[0]:.3f}, {bbox_min[1]:.3f}, {bbox_min[2]:.3f}]")
            info_lines.append(f"    Max: [{bbox_max[0]:.3f}, {bbox_max[1]:.3f}, {bbox_max[2]:.3f}]")

        info_lines.append("")
        info_lines.append("Skeleton Info:")

        if skeleton_info.get("has_skeleton"):
            info_lines.append(f"  Bones: {skeleton_info.get('num_bones', 0)}")
            if skeleton_info.get("skeleton_extents"):
                extents = skeleton_info['skeleton_extents']
                info_lines.append(f"  Skeleton Size: [{extents[0]:.3f}, {extents[1]:.3f}, {extents[2]:.3f}]")
            if skeleton_info.get("bone_names"):
                sample_bones = skeleton_info['bone_names'][:5]
                info_lines.append(f"  Sample bones: {', '.join(sample_bones)}")
        else:
            info_lines.append(f"  Status: {skeleton_info.get('note', 'No skeleton detected')}")

        info_string = "\n".join(info_lines)

        print(f"[UniRigLoadRiggedMesh] ✓ Loaded successfully")
        print(info_string)

        return (rigged_mesh, info_string)


class UniRigPreviewRiggedMesh:
    """
    Preview rigged mesh with interactive FBX viewer.

    Displays the rigged FBX in a Three.js viewer with skeleton visualization
    and interactive bone manipulation controls.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rigged_mesh": ("RIGGED_MESH",),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "preview"
    CATEGORY = "unirig"

    def preview(self, rigged_mesh):
        """
        Preview the rigged mesh in an interactive FBX viewer.

        Args:
            rigged_mesh: RIGGED_MESH dictionary with fbx_path

        Returns:
            dict: UI data for frontend widget
        """
        print(f"[UniRigPreviewRiggedMesh] Preparing preview...")

        # Get the FBX file path
        fbx_path = rigged_mesh.get("fbx_path")
        if not fbx_path or not os.path.exists(fbx_path):
            raise RuntimeError(f"Rigged mesh FBX not found: {fbx_path}")

        print(f"[UniRigPreviewRiggedMesh] FBX path: {fbx_path}")

        # Copy FBX to ComfyUI's output directory so it can be served via /view endpoint
        output_dir = folder_paths.get_output_directory()
        filename = f"rigged_preview_{int(time.time())}.fbx"
        output_fbx_path = os.path.join(output_dir, filename)

        shutil.copy2(fbx_path, output_fbx_path)
        print(f"[UniRigPreviewRiggedMesh] Copied FBX to output: {output_fbx_path}")

        # Get mesh info if available
        has_skinning = rigged_mesh.get("has_skinning", False)
        has_skeleton = rigged_mesh.get("has_skeleton", False)

        print(f"[UniRigPreviewRiggedMesh] Has skinning: {has_skinning}")
        print(f"[UniRigPreviewRiggedMesh] Has skeleton: {has_skeleton}")

        # Return UI data only (no pass-through output)
        return {
            "ui": {
                "fbx_file": [filename],
                "has_skinning": [bool(has_skinning)],
                "has_skeleton": [bool(has_skeleton)],
            }
        }


class UniRigDenormalizeSkeleton:
    """
    Denormalize skeleton from [-1, 1] range back to original mesh scale.
    This makes the transformation explicit and debuggable.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "skeleton": ("SKELETON",),
            }
        }

    RETURN_TYPES = ("SKELETON",)
    RETURN_NAMES = ("denormalized_skeleton",)
    FUNCTION = "denormalize"
    CATEGORY = "UniRig/Utils"

    def denormalize(self, skeleton):
        print(f"[UniRigDenormalizeSkeleton] ⏱️  Starting denormalization...")

        # Read data directly from skeleton dict (no NPZ loading needed!)
        mesh_center = skeleton.get('mesh_center')
        mesh_scale = skeleton.get('mesh_scale')

        if mesh_center is None or mesh_scale is None:
            print(f"[UniRigDenormalizeSkeleton] WARNING: Missing mesh bounds, skeleton is not normalized")
            # Return skeleton as-is
            return (skeleton,)

        # Get normalized data from skeleton dict
        joints_normalized = np.array(skeleton['joints'])
        tails_normalized = np.array(skeleton['tails'])

        # Denormalize joint and tail positions
        joints_denormalized = joints_normalized * mesh_scale + mesh_center
        tails_denormalized = tails_normalized * mesh_scale + mesh_center

        print(f"[UniRigDenormalizeSkeleton] Denormalization:")
        print(f"  Mesh center: {mesh_center}")
        print(f"  Mesh scale: {mesh_scale}")
        print(f"  Joint extents before: {joints_normalized.min(axis=0)} to {joints_normalized.max(axis=0)}")
        print(f"  Joint extents after: {joints_denormalized.min(axis=0)} to {joints_denormalized.max(axis=0)}")
        print(f"  Tail extents before: {tails_normalized.min(axis=0)} to {tails_normalized.max(axis=0)}")
        print(f"  Tail extents after: {tails_denormalized.min(axis=0)} to {tails_denormalized.max(axis=0)}")

        # DEBUG: Check mesh vertices if present
        if 'mesh_vertices' in skeleton and skeleton['mesh_vertices'] is not None:
            mesh_verts = np.array(skeleton['mesh_vertices'])
            print(f"  DEBUG - Mesh vertices in skeleton (NORMALIZED, should match skeleton normalization): {mesh_verts.min(axis=0)} to {mesh_verts.max(axis=0)}")

        # Create denormalized skeleton dict (update existing dict)
        denormalized_skeleton = {
            **skeleton,  # Copy all fields from original skeleton
            'joints': joints_denormalized,
            'tails': tails_denormalized,
            'is_normalized': False,  # Update flag
        }

        print(f"[UniRigDenormalizeSkeleton] ✓ Denormalization complete (no file I/O needed)")

        return (denormalized_skeleton,)


class UniRigValidateSkeleton:
    """
    Validate skeleton quality and data integrity.
    Provides warnings if issues are detected.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "skeleton": ("SKELETON",),
            }
        }

    RETURN_TYPES = ("SKELETON", "STRING")
    RETURN_NAMES = ("skeleton", "validation_report")
    FUNCTION = "validate"
    CATEGORY = "UniRig/Utils"

    def validate(self, skeleton):
        print(f"[UniRigValidateSkeleton] Validating skeleton...")

        issues = []
        warnings = []

        # Check for required fields
        required_fields = ['joints', 'names', 'parents']
        for field in required_fields:
            if field not in skeleton:
                issues.append(f"Missing required field: {field}")

        if issues:
            report = "VALIDATION FAILED:\n" + "\n".join(f"- {issue}" for issue in issues)
            print(f"[UniRigValidateSkeleton] ✗ {report}")
            return (skeleton, report)

        # Get data
        joints = skeleton.get('joints')
        names = skeleton.get('names')
        parents = skeleton.get('parents')
        is_normalized = skeleton.get('is_normalized', None)

        # Check counts match
        num_joints = len(joints) if isinstance(joints, (list, np.ndarray)) else 0
        num_names = len(names) if isinstance(names, (list, np.ndarray)) else 0
        num_parents = len(parents) if isinstance(parents, (list, np.ndarray)) else 0

        if not (num_joints == num_names == num_parents):
            issues.append(f"Count mismatch: {num_joints} joints, {num_names} names, {num_parents} parents")

        # Check normalization status
        if is_normalized is None:
            warnings.append("Normalization status unknown")
        elif is_normalized:
            # Check if joints are in expected range for normalized skeleton
            joints_array = np.array(joints)
            min_val = joints_array.min()
            max_val = joints_array.max()
            if min_val < -1.5 or max_val > 1.5:
                warnings.append(f"Normalized skeleton has values outside [-1, 1]: [{min_val:.2f}, {max_val:.2f}]")
        else:
            # Denormalized - check if values are reasonable
            joints_array = np.array(joints)
            min_val = joints_array.min()
            max_val = joints_array.max()
            if abs(min_val) > 1000 or abs(max_val) > 1000:
                warnings.append(f"Denormalized skeleton has very large values: [{min_val:.2f}, {max_val:.2f}]")

        # Build report
        if issues:
            report = "VALIDATION FAILED:\n" + "\n".join(f"- {issue}" for issue in issues)
            if warnings:
                report += "\n\nWARNINGS:\n" + "\n".join(f"- {warning}" for warning in warnings)
            print(f"[UniRigValidateSkeleton] ✗ {report}")
        elif warnings:
            report = "VALIDATION PASSED WITH WARNINGS:\n" + "\n".join(f"- {warning}" for warning in warnings)
            print(f"[UniRigValidateSkeleton] ⚠ {report}")
        else:
            report = f"VALIDATION PASSED:\n- {num_joints} joints\n- Hierarchy valid\n- Normalization: {'normalized' if is_normalized else 'denormalized'}"
            print(f"[UniRigValidateSkeleton] ✓ {report}")

        return (skeleton, report)


class UniRigPrepareSkeletonForSkinning:
    """
    Prepare skeleton data in the exact format required by the skinning model.
    Saves predict_skeleton.npz with correct field names.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "skeleton": ("SKELETON",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("skeleton_npz_path",)
    FUNCTION = "prepare"
    CATEGORY = "UniRig/Utils"

    def prepare(self, skeleton):
        print(f"[UniRigPrepareSkeletonForSkinning] Preparing skeleton for skinning...")

        # Create temporary directory for predict_skeleton.npz
        temp_dir = tempfile.mkdtemp(prefix="unirig_skinning_")
        predict_skeleton_dir = os.path.join(temp_dir, "input")
        os.makedirs(predict_skeleton_dir, exist_ok=True)
        predict_skeleton_path = os.path.join(predict_skeleton_dir, "predict_skeleton.npz")

        # Read ALL data from skeleton dict (no NPZ loading!)
        # Build save_data dict with CORRECT field names for RawData
        save_data = {
            # Core skeleton data (MUST use these exact names for RawData)
            'joints': skeleton['joints'],
            'names': skeleton['names'],
            'parents': skeleton['parents'],
            'tails': skeleton['tails'],
        }

        # Add mesh data (use prefixed keys from ExtractSkeleton)
        mesh_data_mapping = {
            'mesh_vertices': 'vertices',
            'mesh_faces': 'faces',
            'mesh_vertex_normals': 'vertex_normals',
            'mesh_face_normals': 'face_normals',
        }
        for skel_key, npz_key in mesh_data_mapping.items():
            if skel_key in skeleton:
                save_data[npz_key] = skeleton[skel_key]

        # Add optional fields that RawData expects
        save_data['skin'] = None  # Will be filled by skinning
        save_data['no_skin'] = None
        save_data['matrix_local'] = skeleton.get('matrix_local')
        save_data['path'] = None
        save_data['cls'] = skeleton.get('cls')

        # Save NPZ (only place where we save to disk for ML process)
        np.savez(predict_skeleton_path, **save_data)

        print(f"[UniRigPrepareSkeletonForSkinning] ✓ Saved skeleton to: {predict_skeleton_path}")
        print(f"[UniRigPrepareSkeletonForSkinning] Fields saved: {list(save_data.keys())}")
        print(f"[UniRigPrepareSkeletonForSkinning] Read all data from skeleton dict (no NPZ loading needed)")

        return (predict_skeleton_path,)


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
            }
        }

    RETURN_TYPES = ("RIGGED_MESH",)
    FUNCTION = "apply_skinning"
    CATEGORY = "UniRig"

    def apply_skinning(self, normalized_mesh, skeleton):
        print(f"[UniRigApplySkinningML] ⏱️  Starting ML skinning...")

        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix="unirig_skinning_")
        predict_skeleton_dir = os.path.join(temp_dir, "input")
        os.makedirs(predict_skeleton_dir, exist_ok=True)

        # Prepare skeleton NPZ from dict (absorb PrepareSkeletonForSkinning logic)
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

        np.savez(predict_skeleton_path, **save_data)
        print(f"[UniRigApplySkinningML] Prepared skeleton NPZ: {predict_skeleton_path}")

        # DEBUG: Show what we're sending to ML
        print(f"[UniRigApplySkinningML] DEBUG - NPZ joints bounds: {save_data['joints'].min(axis=0)} to {save_data['joints'].max(axis=0)}")
        print(f"[UniRigApplySkinningML] DEBUG - NPZ tails bounds: {save_data['tails'].min(axis=0)} to {save_data['tails'].max(axis=0)}")
        if 'vertices' in save_data and save_data['vertices'] is not None:
            print(f"[UniRigApplySkinningML] DEBUG - NPZ mesh vertices bounds: {np.array(save_data['vertices']).min(axis=0)} to {np.array(save_data['vertices']).max(axis=0)}")

        # Export mesh to GLB
        input_glb = os.path.join(temp_dir, "input.glb")

        normalized_mesh.export(input_glb)
        print(f"[UniRigApplySkinningML] Exported mesh: {normalized_mesh.vertices.shape[0]} vertices, {normalized_mesh.faces.shape[0]} faces")
        print(f"[UniRigApplySkinningML] DEBUG - GLB mesh bounds: {normalized_mesh.vertices.min(axis=0)} to {normalized_mesh.vertices.max(axis=0)}")

        # Run skinning inference
        print(f"[UniRigApplySkinningML] Running skinning inference...")

        python_exe = sys.executable  # Use current Python (conda env)
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

        # Set BLENDER_EXE environment variable for FBX export
        env = os.environ.copy()
        if BLENDER_EXE:
            env['BLENDER_EXE'] = BLENDER_EXE

        result = subprocess.run(
            cmd,
            cwd=UNIRIG_PATH,
            capture_output=True,
            text=True,
            timeout=600,
            env=env
        )

        if result.stdout:
            print(f"[UniRigApplySkinningML] Skinning stdout:\n{result.stdout}")
        if result.stderr:
            print(f"[UniRigApplySkinningML] Skinning stderr:\n{result.stderr}")

        if result.returncode != 0:
            print(f"[UniRigApplySkinningML] ✗ Skinning failed with return code: {result.returncode}")
            raise RuntimeError(f"Skinning generation failed with exit code {result.returncode}")

        print(f"[UniRigApplySkinningML] ⏱️  Skinning completed")

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

        print(f"[UniRigApplySkinningML] ✓ Found output FBX: {fbx_path}")
        print(f"[UniRigApplySkinningML] FBX file size: {os.path.getsize(fbx_path)} bytes")

        # Create rigged mesh dict
        rigged_mesh = {
            "fbx_path": fbx_path,
            "has_skinning": True,
            "has_skeleton": True,
        }

        print(f"[UniRigApplySkinningML] ✓✓✓ Skinning application complete! ✓✓✓")

        return (rigged_mesh,)


NODE_CLASS_MAPPINGS = {
    "UniRigExtractSkeleton": UniRigExtractSkeleton,
    "UniRigApplySkinning": UniRigApplySkinning,
    "UniRigExtractRig": UniRigExtractRig,
    "UniRigSaveSkeleton": UniRigSaveSkeleton,
    "UniRigSaveRiggedMesh": UniRigSaveRiggedMesh,
    "UniRigLoadRiggedMesh": UniRigLoadRiggedMesh,
    "UniRigPreviewRiggedMesh": UniRigPreviewRiggedMesh,
    "UniRigDenormalizeSkeleton": UniRigDenormalizeSkeleton,
    "UniRigValidateSkeleton": UniRigValidateSkeleton,
    "UniRigPrepareSkeletonForSkinning": UniRigPrepareSkeletonForSkinning,
    "UniRigApplySkinningML": UniRigApplySkinningML,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UniRigExtractSkeleton": "UniRig: Extract Skeleton",
    "UniRigApplySkinning": "UniRig: Apply Skinning (Legacy)",
    "UniRigExtractRig": "UniRig: Extract Full Rig (All-in-One)",
    "UniRigSaveSkeleton": "UniRig: Save Skeleton",
    "UniRigSaveRiggedMesh": "UniRig: Save Rigged Mesh",
    "UniRigLoadRiggedMesh": "UniRig: Load Rigged Mesh",
    "UniRigPreviewRiggedMesh": "UniRig: Preview Rigged Mesh",
    "UniRigDenormalizeSkeleton": "UniRig: Denormalize Skeleton",
    "UniRigValidateSkeleton": "UniRig: Validate Skeleton",
    "UniRigPrepareSkeletonForSkinning": "UniRig: Prepare Skeleton for Skinning",
    "UniRigApplySkinningML": "UniRig: Apply Skinning (ML Only)",
}
