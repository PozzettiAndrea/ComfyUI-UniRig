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


# Get paths relative to this file
NODE_DIR = Path(__file__).parent.absolute()
LIB_DIR = NODE_DIR / "lib"
UNIRIG_PATH = str(LIB_DIR / "unirig")
BLENDER_SCRIPT = str(LIB_DIR / "blender_extract.py")
BLENDER_PARSE_SKELETON = str(LIB_DIR / "blender_parse_skeleton.py")

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
        print(f"[UniRigExtractSkeleton] Starting skeleton extraction...")

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
            print(f"[UniRigExtractSkeleton] Exporting mesh to {input_path}")
            trimesh.export(input_path)

            # Step 1: Extract/preprocess mesh with Blender
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

                print(f"[UniRigExtractSkeleton] ✓ Mesh preprocessed: {npz_path}")

            except subprocess.TimeoutExpired:
                raise RuntimeError("Blender extraction timed out (>2 minutes)")
            except Exception as e:
                print(f"[UniRigExtractSkeleton] Blender error: {e}")
                raise

            # Step 2: Run skeleton inference
            print(f"[UniRigExtractSkeleton] Step 2: Running skeleton inference...")
            run_cmd = [
                sys.executable, os.path.join(UNIRIG_PATH, "run.py"),
                "--task", os.path.join(UNIRIG_PATH, "configs/task/quick_inference_skeleton_articulationxl_ar_256.yaml"),
                "--seed", str(seed),
                "--input", input_path,
                "--output", output_path,
                "--npz_dir", tmpdir,  # This should match where NPZ is: tmpdir/input/raw_data.npz
                "--blender_exe", BLENDER_EXE,
            ]

            print(f"[UniRigExtractSkeleton] Running: {' '.join(run_cmd)}")

            try:
                result = subprocess.run(
                    run_cmd,
                    cwd=UNIRIG_PATH,
                    capture_output=True,
                    text=True,
                    timeout=180
                )
                if result.stdout:
                    print(f"[UniRigExtractSkeleton] Inference stdout:\n{result.stdout}")
                if result.stderr:
                    print(f"[UniRigExtractSkeleton] Inference stderr:\n{result.stderr}")

                if result.returncode != 0:
                    raise RuntimeError(f"Inference failed with exit code {result.returncode}")

            except subprocess.TimeoutExpired:
                raise RuntimeError("Inference timed out (>3 minutes)")
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

            print(f"[UniRigExtractSkeleton] Parsing FBX output with Blender...")
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
                    raise RuntimeError(f"Skeleton parsing failed: {skeleton_npz} not created")

            except subprocess.TimeoutExpired:
                raise RuntimeError("Skeleton parsing timed out (>1 minute)")
            except Exception as e:
                print(f"[UniRigExtractSkeleton] Skeleton parse error: {e}")
                raise

            # Load skeleton data
            skeleton_data = np.load(skeleton_npz)
            vertices = skeleton_data['vertices']
            edges = skeleton_data['edges']

            print(f"[UniRigExtractSkeleton] Extracted {len(vertices)} joints, {len(edges)} bones")

            # Normalize to [-1, 1]
            vertices = normalize_skeleton(vertices)
            print(f"[UniRigExtractSkeleton] Normalized to range [{vertices.min():.3f}, {vertices.max():.3f}]")

            skeleton = {
                "vertices": vertices,
                "edges": edges,
            }

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
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("rigged_mesh",)
    FUNCTION = "extract"
    CATEGORY = "UniRig"

    def extract(self, trimesh, seed):
        """Extract full rig with skinning weights."""
        print(f"[UniRigExtractRig] Starting full rig extraction...")

        if not os.path.exists(UNIRIG_PATH):
            raise RuntimeError(f"UniRig not found at {UNIRIG_PATH}")

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.glb")
            skeleton_path = os.path.join(tmpdir, "skeleton.fbx")
            rigged_path = os.path.join(tmpdir, "rigged.fbx")

            # Export mesh
            trimesh.export(input_path)

            # Step 1: Generate skeleton
            print(f"[UniRigExtractRig] Step 1: Generating skeleton...")
            subprocess.run([
                "bash",
                os.path.join(UNIRIG_PATH, "launch", "inference", "generate_skeleton.sh"),
                "--input", input_path,
                "--output", skeleton_path,
                "--seed", str(seed),
            ], cwd=UNIRIG_PATH, check=True, timeout=300)

            if not os.path.exists(skeleton_path):
                raise RuntimeError(f"Skeleton generation failed: {skeleton_path} not created")

            # Step 2: Generate skinning
            print(f"[UniRigExtractRig] Step 2: Generating skinning weights...")
            subprocess.run([
                "bash",
                os.path.join(UNIRIG_PATH, "launch", "inference", "generate_skin.sh"),
                "--input", skeleton_path,
                "--output", rigged_path,
            ], cwd=UNIRIG_PATH, check=True, timeout=300)

            if not os.path.exists(rigged_path):
                raise RuntimeError(f"Skinning generation failed: {rigged_path} not created")

            # Load rigged mesh
            rigged_mesh = trimesh.load(rigged_path)

            print(f"[UniRigExtractRig] Rig extraction complete!")

            return (rigged_mesh,)


NODE_CLASS_MAPPINGS = {
    "UniRigExtractSkeleton": UniRigExtractSkeleton,
    "UniRigExtractRig": UniRigExtractRig,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UniRigExtractSkeleton": "UniRig: Extract Skeleton",
    "UniRigExtractRig": "UniRig: Extract Full Rig",
}
