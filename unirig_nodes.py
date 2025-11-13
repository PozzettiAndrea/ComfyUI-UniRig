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
from pathlib import Path
import folder_paths
import time


# Get paths relative to this file
NODE_DIR = Path(__file__).parent.absolute()
LIB_DIR = NODE_DIR / "lib"
UNIRIG_PATH = str(LIB_DIR / "unirig")
BLENDER_SCRIPT = str(LIB_DIR / "blender_extract.py")
BLENDER_PARSE_SKELETON = str(LIB_DIR / "blender_parse_skeleton.py")

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


def normalize_skeleton(vertices: np.ndarray) -> np.ndarray:
    """Normalize skeleton vertices to [-1, 1] range."""
    min_coords = vertices.min(axis=0)
    max_coords = vertices.max(axis=0)
    center = (min_coords + max_coords) / 2
    vertices = vertices - center
    scale = (max_coords - min_coords).max() / 2
    if scale > 0:
        vertices = vertices / scale
    return vertices


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

    RETURN_TYPES = ("SKELETON",)
    RETURN_NAMES = ("skeleton",)
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
                    timeout=180
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
                raise RuntimeError("Inference timed out (>3 minutes)")
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
            vertices = skeleton_data['vertices']
            edges = skeleton_data['edges']

            print(f"[UniRigExtractSkeleton] Extracted {len(vertices)} joints, {len(edges)} bones")

            # Normalize to [-1, 1]
            vertices = normalize_skeleton(vertices)
            print(f"[UniRigExtractSkeleton] Normalized to range [{vertices.min():.3f}, {vertices.max():.3f}]")

            # Build skeleton dict with basic data
            skeleton = {
                "vertices": vertices,
                "edges": edges,
            }

            # Add hierarchy data if available (for animation-ready export)
            if 'bone_names' in skeleton_data:
                skeleton['bone_names'] = skeleton_data['bone_names'].tolist()
                skeleton['bone_parents'] = skeleton_data['bone_parents'].tolist()
                skeleton['bone_to_head_vertex'] = skeleton_data['bone_to_head_vertex'].tolist()
                print(f"[UniRigExtractSkeleton] Included hierarchy: {len(skeleton['bone_names'])} bones with parent relationships")
            else:
                print(f"[UniRigExtractSkeleton] No hierarchy data in skeleton (edges-only mode)")

            total_time = time.time() - total_start
            print(f"[UniRigExtractSkeleton] ✓✓✓ Skeleton extraction complete! ✓✓✓")
            print(f"[UniRigExtractSkeleton] ⏱️  TOTAL TIME: {total_time:.2f}s")
            return (skeleton,)

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
            # Ensure HuggingFace cache is set for subprocess
            if UNIRIG_MODELS_DIR:
                env['HF_HOME'] = str(UNIRIG_MODELS_DIR)
                env['TRANSFORMERS_CACHE'] = str(UNIRIG_MODELS_DIR / "transformers")
                env['HF_HUB_CACHE'] = str(UNIRIG_MODELS_DIR / "hub")

            try:
                result = subprocess.run(skeleton_cmd, cwd=UNIRIG_PATH, env=env, capture_output=True, text=True, timeout=180)
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
                raise RuntimeError("Skeleton generation timed out (>3 minutes)")
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
                result = subprocess.run(skin_cmd, cwd=UNIRIG_PATH, env=env, capture_output=True, text=True, timeout=180)
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
                raise RuntimeError("Skinning generation timed out (>3 minutes)")
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
        # Scale up to reasonable size (e.g., 1 meter = 1 unit)
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


NODE_CLASS_MAPPINGS = {
    "UniRigExtractSkeleton": UniRigExtractSkeleton,
    "UniRigExtractRig": UniRigExtractRig,
    "UniRigSaveSkeleton": UniRigSaveSkeleton,
    "UniRigSaveRiggedMesh": UniRigSaveRiggedMesh,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UniRigExtractSkeleton": "UniRig: Extract Skeleton",
    "UniRigExtractRig": "UniRig: Extract Full Rig",
    "UniRigSaveSkeleton": "UniRig: Save Skeleton",
    "UniRigSaveRiggedMesh": "UniRig: Save Rigged Mesh",
}
