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

# VRoid to Mixamo bone name mapping (52 bones, 1:1 correspondence)
VROID_TO_MIXAMO_BONE_MAP = {
    # Body (22 bones)
    "J_Bip_C_Hips": "mixamorig:Hips",
    "J_Bip_C_Spine": "mixamorig:Spine",
    "J_Bip_C_Chest": "mixamorig:Spine1",
    "J_Bip_C_UpperChest": "mixamorig:Spine2",
    "J_Bip_C_Neck": "mixamorig:Neck",
    "J_Bip_C_Head": "mixamorig:Head",
    "J_Bip_L_Shoulder": "mixamorig:LeftShoulder",
    "J_Bip_L_UpperArm": "mixamorig:LeftArm",
    "J_Bip_L_LowerArm": "mixamorig:LeftForeArm",
    "J_Bip_L_Hand": "mixamorig:LeftHand",
    "J_Bip_R_Shoulder": "mixamorig:RightShoulder",
    "J_Bip_R_UpperArm": "mixamorig:RightArm",
    "J_Bip_R_LowerArm": "mixamorig:RightForeArm",
    "J_Bip_R_Hand": "mixamorig:RightHand",
    "J_Bip_L_UpperLeg": "mixamorig:LeftUpLeg",
    "J_Bip_L_LowerLeg": "mixamorig:LeftLeg",
    "J_Bip_L_Foot": "mixamorig:LeftFoot",
    "J_Bip_L_ToeBase": "mixamorig:LeftToeBase",
    "J_Bip_R_UpperLeg": "mixamorig:RightUpLeg",
    "J_Bip_R_LowerLeg": "mixamorig:RightLeg",
    "J_Bip_R_Foot": "mixamorig:RightFoot",
    "J_Bip_R_ToeBase": "mixamorig:RightToeBase",
    # Left Hand (15 bones)
    "J_Bip_L_Thumb1": "mixamorig:LeftHandThumb1",
    "J_Bip_L_Thumb2": "mixamorig:LeftHandThumb2",
    "J_Bip_L_Thumb3": "mixamorig:LeftHandThumb3",
    "J_Bip_L_Index1": "mixamorig:LeftHandIndex1",
    "J_Bip_L_Index2": "mixamorig:LeftHandIndex2",
    "J_Bip_L_Index3": "mixamorig:LeftHandIndex3",
    "J_Bip_L_Middle1": "mixamorig:LeftHandMiddle1",
    "J_Bip_L_Middle2": "mixamorig:LeftHandMiddle2",
    "J_Bip_L_Middle3": "mixamorig:LeftHandMiddle3",
    "J_Bip_L_Ring1": "mixamorig:LeftHandRing1",
    "J_Bip_L_Ring2": "mixamorig:LeftHandRing2",
    "J_Bip_L_Ring3": "mixamorig:LeftHandRing3",
    "J_Bip_L_Little1": "mixamorig:LeftHandPinky1",
    "J_Bip_L_Little2": "mixamorig:LeftHandPinky2",
    "J_Bip_L_Little3": "mixamorig:LeftHandPinky3",
    # Right Hand (15 bones)
    "J_Bip_R_Thumb1": "mixamorig:RightHandThumb1",
    "J_Bip_R_Thumb2": "mixamorig:RightHandThumb2",
    "J_Bip_R_Thumb3": "mixamorig:RightHandThumb3",
    "J_Bip_R_Index1": "mixamorig:RightHandIndex1",
    "J_Bip_R_Index2": "mixamorig:RightHandIndex2",
    "J_Bip_R_Index3": "mixamorig:RightHandIndex3",
    "J_Bip_R_Middle1": "mixamorig:RightHandMiddle1",
    "J_Bip_R_Middle2": "mixamorig:RightHandMiddle2",
    "J_Bip_R_Middle3": "mixamorig:RightHandMiddle3",
    "J_Bip_R_Ring1": "mixamorig:RightHandRing1",
    "J_Bip_R_Ring2": "mixamorig:RightHandRing2",
    "J_Bip_R_Ring3": "mixamorig:RightHandRing3",
    "J_Bip_R_Little1": "mixamorig:RightHandPinky1",
    "J_Bip_R_Little2": "mixamorig:RightHandPinky2",
    "J_Bip_R_Little3": "mixamorig:RightHandPinky3",
}

# VRoid to SMPL bone mapping (22 joints - maps VRoid bones to SMPL joint names)
VROID_TO_SMPL_BONE_MAP = {
    "J_Bip_C_Hips": "Pelvis",           # 0
    "J_Bip_L_UpperLeg": "L_Hip",         # 1
    "J_Bip_R_UpperLeg": "R_Hip",         # 2
    "J_Bip_C_Spine": "Spine1",           # 3
    "J_Bip_L_LowerLeg": "L_Knee",        # 4
    "J_Bip_R_LowerLeg": "R_Knee",        # 5
    "J_Bip_C_Chest": "Spine2",           # 6
    "J_Bip_L_Foot": "L_Ankle",           # 7
    "J_Bip_R_Foot": "R_Ankle",           # 8
    "J_Bip_C_UpperChest": "Spine3",      # 9
    "J_Bip_L_ToeBase": "L_Foot",         # 10
    "J_Bip_R_ToeBase": "R_Foot",         # 11
    "J_Bip_C_Neck": "Neck",              # 12
    "J_Bip_L_Shoulder": "L_Collar",      # 13
    "J_Bip_R_Shoulder": "R_Collar",      # 14
    "J_Bip_C_Head": "Head",              # 15
    "J_Bip_L_UpperArm": "L_Shoulder",    # 16
    "J_Bip_R_UpperArm": "R_Shoulder",    # 17
    "J_Bip_L_LowerArm": "L_Elbow",       # 18
    "J_Bip_R_LowerArm": "R_Elbow",       # 19
    "J_Bip_L_Hand": "L_Wrist",           # 20
    "J_Bip_R_Hand": "R_Wrist",           # 21
}

# SMPL joint names in order (22 joints)
SMPL_JOINT_NAMES = [
    'Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee',
    'Spine2', 'L_Ankle', 'R_Ankle', 'Spine3', 'L_Foot', 'R_Foot',
    'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder',
    'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist'
]

# SMPL parent hierarchy (22 joints) - index of parent for each joint
SMPL_PARENTS = [
    -1,  # 0: Pelvis (root)
    0,   # 1: L_Hip -> Pelvis
    0,   # 2: R_Hip -> Pelvis
    0,   # 3: Spine1 -> Pelvis
    1,   # 4: L_Knee -> L_Hip
    2,   # 5: R_Knee -> R_Hip
    3,   # 6: Spine2 -> Spine1
    4,   # 7: L_Ankle -> L_Knee
    5,   # 8: R_Ankle -> R_Knee
    6,   # 9: Spine3 -> Spine2
    7,   # 10: L_Foot -> L_Ankle
    8,   # 11: R_Foot -> R_Ankle
    9,   # 12: Neck -> Spine3
    9,   # 13: L_Collar -> Spine3
    9,   # 14: R_Collar -> Spine3
    12,  # 15: Head -> Neck
    13,  # 16: L_Shoulder -> L_Collar
    14,  # 17: R_Shoulder -> R_Collar
    16,  # 18: L_Elbow -> L_Shoulder
    17,  # 19: R_Elbow -> R_Shoulder
    18,  # 20: L_Wrist -> L_Elbow
    19,  # 21: R_Wrist -> R_Elbow
]

# SMPL canonical bone directions (unit vectors pointing from head to tail)
# These define how each bone should be oriented in rest pose
# Coordinate system: Blender default (X=right, Y=forward, Z=up)
# These get rotated to SMPL coords (Y-up) when skeleton_template="smpl"
# For symmetric bones, L and R have mirrored X component (left/right)
SMPL_BONE_DIRECTIONS = {
    'Pelvis':     [0, 0, 1],      # Up +Z (toward spine)
    'L_Hip':      [0, 0, -1],     # Down -Z (toward knee)
    'R_Hip':      [0, 0, -1],     # Down -Z (toward knee)
    'Spine1':     [0, 0, 1],      # Up +Z
    'L_Knee':     [0, 0, -1],     # Down -Z (toward ankle)
    'R_Knee':     [0, 0, -1],     # Down -Z (toward ankle)
    'Spine2':     [0, 0, 1],      # Up +Z
    'L_Ankle':    [0, 1, 0],      # Forward +Y (toward toe)
    'R_Ankle':    [0, 1, 0],      # Forward +Y (toward toe)
    'Spine3':     [0, 0, 1],      # Up +Z
    'L_Foot':     [0, 1, 0],      # Forward +Y
    'R_Foot':     [0, 1, 0],      # Forward +Y
    'Neck':       [0, 0, 1],      # Up +Z
    'L_Collar':   [1, 0, 0],      # Left +X (toward shoulder)
    'R_Collar':   [-1, 0, 0],     # Right -X (toward shoulder)
    'Head':       [0, 0, 1],      # Up +Z
    'L_Shoulder': [1, 0, 0],      # Left +X (toward elbow)
    'R_Shoulder': [-1, 0, 0],     # Right -X (toward elbow)
    'L_Elbow':    [1, 0, 0],      # Left +X (toward wrist)
    'R_Elbow':    [-1, 0, 0],     # Right -X (toward wrist)
    'L_Wrist':    [1, 0, 0],      # Left +X (toward hand)
    'R_Wrist':    [-1, 0, 0],     # Right -X (toward hand)
}

# Default bone length for SMPL (used when computing tails)
SMPL_DEFAULT_BONE_LENGTH = 0.1

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
                "skeleton_template": (["auto", "vroid", "mixamo", "smpl", "articulationxl"], {
                    "default": "auto",
                    "tooltip": "Skeleton template: auto (let model decide), vroid (52 bones), mixamo (Mixamo-compatible 52 bones), smpl (22 joints, SMPL-compatible for direct motion application), articulationxl (generic/flexible)"
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

    RETURN_TYPES = ("TRIMESH", "SKELETON", "IMAGE")
    RETURN_NAMES = ("normalized_mesh", "skeleton", "texture_preview")
    FUNCTION = "extract"
    CATEGORY = "UniRig"

    def extract(self, trimesh, skeleton_model, seed, skeleton_template="auto", target_face_count=None):
        """Extract skeleton using UniRig with cached model only."""
        total_start = time.time()
        print(f"[UniRigExtractSkeletonNew] Starting skeleton extraction (cached model only)...")
        print(f"[UniRigExtractSkeletonNew] Skeleton template: {skeleton_template}")

        # Track if we need to remap to mixamo or smpl naming
        remap_to_mixamo = (skeleton_template == "mixamo")
        remap_to_smpl = (skeleton_template == "smpl")

        # If mixamo is requested, use vroid for extraction (model trained on vroid), then remap names
        if skeleton_template == "mixamo":
            skeleton_template = "vroid"
            print(f"[UniRigExtractSkeletonNew] Mixamo requested, using vroid extraction + name remapping")

        # If smpl is requested, use vroid for extraction, then filter to 22 SMPL joints
        if skeleton_template == "smpl":
            skeleton_template = "vroid"
            print(f"[UniRigExtractSkeletonNew] SMPL requested, using vroid extraction + SMPL conversion")

        # Validate model is provided
        if skeleton_model is None:
            raise RuntimeError(
                "skeleton_model is required for UniRigExtractSkeletonNew. "
                "Please connect a UniRigLoadSkeletonModel node."
            )

        # Validate model has cache key
        if not skeleton_model.get("model_cache_key"):
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
                    "skeleton_model does not have a cached model. "
                    "Ensure UniRigLoadSkeletonModel has 'cache_to_gpu' enabled."
                )

        print(f"[UniRigExtractSkeletonNew] Using pre-loaded cached model")
        task_config_path = skeleton_model.get("task_config_path")

        # Check if Blender is available
        if not BLENDER_EXE or not os.path.exists(BLENDER_EXE):
            raise RuntimeError(
                f"Blender not found. Please run blender_install.py or install manually."
            )

        # Check if UniRig is available
        if not os.path.exists(UNIRIG_PATH):
            raise RuntimeError(
                f"UniRig code not found at {UNIRIG_PATH}. "
                "The lib/unirig directory should contain the UniRig source code."
            )

        # Create temp files
        # ignore_cleanup_errors=True prevents Windows errors when npz files are still locked
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            input_path = os.path.join(tmpdir, "input.glb")
            npz_dir = os.path.join(tmpdir, "input")
            npz_path = os.path.join(npz_dir, "raw_data.npz")
            # FBX is saved to npz_dir when user_mode=False (for NPZ export)
            output_path = os.path.join(npz_dir, "skeleton.fbx")

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

            # Map skeleton template to cls token
            cls_value = None  # auto (let model decide)
            if skeleton_template == "vroid":
                cls_value = "vroid"
            elif skeleton_template == "articulationxl":
                cls_value = "articulationxl"

            if cls_value:
                print(f"[UniRigExtractSkeletonNew] Forcing skeleton template: {cls_value}")
            else:
                print(f"[UniRigExtractSkeletonNew] Using auto skeleton detection")

            request_data = {
                "seed": seed,
                "input": input_path,
                "output": output_path,
                "npz_dir": tmpdir,
                "cls": cls_value,
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
            # For mesh/texture: always use raw_data.npz (has texture data)
            # For skeleton: use parsed FBX output (has correct bone names from model)
            preprocessing_npz = os.path.join(tmpdir, "input", "raw_data.npz")

            uv_coords = None
            uv_faces = None
            material_name = None
            texture_path = None
            texture_data_base64 = None
            texture_format = None
            texture_width = 0
            texture_height = 0

            # Load mesh and texture data from preprocessing NPZ (raw_data.npz)
            if os.path.exists(preprocessing_npz):
                print(f"[UniRigExtractSkeletonNew] Loading mesh/texture from: raw_data.npz")
                preprocess_data = np.load(preprocessing_npz, allow_pickle=True)

                # Helper to safely get array field (handles 0-d arrays from None values)
                def safe_get_array(key):
                    if key not in preprocess_data:
                        return None
                    val = preprocess_data[key]
                    if hasattr(val, 'ndim') and val.ndim == 0:
                        # 0-d array (scalar) - treat as None
                        return None
                    return val

                mesh_vertices_original = preprocess_data['vertices']
                mesh_faces = preprocess_data['faces']
                vertex_normals = safe_get_array('vertex_normals')
                face_normals = safe_get_array('face_normals')

                # Load UV coordinates if available
                uv_coords_data = safe_get_array('uv_coords')
                if uv_coords_data is not None and len(uv_coords_data) > 0:
                    uv_coords = uv_coords_data
                    uv_faces = safe_get_array('uv_faces')
                    print(f"[UniRigExtractSkeletonNew] Loaded UV coordinates: {len(uv_coords)} UVs")

                # Load material and texture info if available
                mat_name = safe_get_array('material_name')
                if mat_name is not None:
                    material_name = str(mat_name)
                tex_path = safe_get_array('texture_path')
                if tex_path is not None:
                    texture_path = str(tex_path)

                # Load texture data if available
                # Note: texture fields may be 0-d string scalars, handle them specially
                if 'texture_data_base64' in preprocess_data:
                    tex_data = preprocess_data['texture_data_base64']
                    # Handle both 0-d scalar and regular arrays
                    if hasattr(tex_data, 'item'):
                        tex_str = tex_data.item() if tex_data.ndim == 0 else str(tex_data)
                    else:
                        tex_str = str(tex_data)

                    if len(tex_str) > 0:
                        texture_data_base64 = tex_str

                        # Load texture metadata (also handle 0-d scalars)
                        if 'texture_format' in preprocess_data:
                            fmt = preprocess_data['texture_format']
                            texture_format = fmt.item() if hasattr(fmt, 'item') and fmt.ndim == 0 else str(fmt)
                        if 'texture_width' in preprocess_data:
                            w = preprocess_data['texture_width']
                            texture_width = int(w.item() if hasattr(w, 'item') and w.ndim == 0 else w)
                        if 'texture_height' in preprocess_data:
                            h = preprocess_data['texture_height']
                            texture_height = int(h.item() if hasattr(h, 'item') and h.ndim == 0 else h)

                        print(f"[UniRigExtractSkeletonNew] Loaded texture: {texture_width}x{texture_height} {texture_format} ({len(texture_data_base64) // 1024}KB base64)")

                # Close npz file to release handle (required for Windows temp cleanup)
                preprocess_data.close()
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

                # Get bone names - prioritize model-generated names from predict_skeleton.npz
                # The model generates correct semantic names (e.g., VRoid template names)
                # but Blender parsing may return generic names, so we prefer model names
                model_bone_names = None
                model_output_npz = os.path.join(tmpdir, "input", "predict_skeleton.npz")
                if os.path.exists(model_output_npz):
                    try:
                        model_data = np.load(model_output_npz, allow_pickle=True)
                        if 'names' in model_data and model_data['names'] is not None:
                            raw_names = model_data['names']

                            # Handle different numpy array types
                            if raw_names.ndim == 0:
                                # 0-dimensional array (scalar) - this shouldn't happen with the fix
                                # but kept for backward compatibility with old NPZ files
                                print(f"[UniRigExtractSkeletonNew] Warning: names is 0-d array (old format)")
                                model_bone_names = None  # Skip, use fallback
                            elif raw_names.ndim == 1:
                                # Proper 1-D array (expected format after fix)
                                model_bone_names = [str(name) for name in raw_names]
                                print(f"[UniRigExtractSkeletonNew] Loaded {len(model_bone_names)} model bone names from predict_skeleton.npz")
                            else:
                                print(f"[UniRigExtractSkeletonNew] Warning: names has unexpected shape {raw_names.shape}")
                                model_bone_names = None
                        # Close npz file to release handle (required for Windows temp cleanup)
                        model_data.close()
                    except Exception as e:
                        print(f"[UniRigExtractSkeletonNew] Warning: Could not load model bone names: {e}")
                        import traceback
                        traceback.print_exc()

                if model_bone_names is not None and len(model_bone_names) == num_bones:
                    # Use model-generated names (correct VRoid/template names)
                    names_list = [str(n) for n in model_bone_names]
                    print(f"[UniRigExtractSkeletonNew] ✓ Using {len(names_list)} model-generated bone names")
                else:
                    if model_bone_names is not None:
                        print(f"[UniRigExtractSkeletonNew] Model names count mismatch: {len(model_bone_names)} names vs {num_bones} bones")
                    # Fallback to Blender-parsed names
                    bone_names = skeleton_data.get('bone_names', None)
                    if bone_names is not None:
                        names_list = [str(n) for n in bone_names]
                        print(f"[UniRigExtractSkeletonNew] Using {len(names_list)} Blender-parsed bone names (fallback)")
                    else:
                        names_list = [f"bone_{i}" for i in range(num_bones)]
                        print(f"[UniRigExtractSkeletonNew] Using {len(names_list)} generic bone names (fallback)")

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

            # Remap bone names if mixamo was requested (applies to both branches above)
            if remap_to_mixamo:
                remapped_names = []
                for name in names_list:
                    if name in VROID_TO_MIXAMO_BONE_MAP:
                        remapped_names.append(VROID_TO_MIXAMO_BONE_MAP[name])
                    else:
                        remapped_names.append(name)  # Keep original if not in map
                names_list = remapped_names
                print(f"[UniRigExtractSkeletonNew] Remapped {len(names_list)} bones to Mixamo naming")

            # Convert to SMPL skeleton if requested (filter 52 VRoid bones to 22 SMPL joints)
            if remap_to_smpl:
                print(f"[UniRigExtractSkeletonNew] Converting to SMPL skeleton (22 joints)...")

                # Build VRoid name -> index mapping from current skeleton
                vroid_name_to_idx = {name: i for i, name in enumerate(names_list)}

                # Filter to only SMPL joints (22 out of 52)
                smpl_joints = []
                missing_joints = []

                for smpl_name in SMPL_JOINT_NAMES:
                    # Find corresponding VRoid bone name
                    vroid_name = None
                    for vn, sn in VROID_TO_SMPL_BONE_MAP.items():
                        if sn == smpl_name:
                            vroid_name = vn
                            break

                    if vroid_name and vroid_name in vroid_name_to_idx:
                        idx = vroid_name_to_idx[vroid_name]
                        smpl_joints.append(bone_joints[idx])
                    else:
                        missing_joints.append(smpl_name)
                        # Use zero position as fallback (shouldn't happen)
                        smpl_joints.append(np.array([0, 0, 0]))

                if missing_joints:
                    print(f"[UniRigExtractSkeletonNew] Warning: Missing VRoid bones for SMPL joints: {missing_joints}")

                # Replace with SMPL data
                bone_joints = np.array(smpl_joints)
                names_list = list(SMPL_JOINT_NAMES)
                parents_list = [None if p == -1 else p for p in SMPL_PARENTS]

                # Compute tails using CANONICAL SMPL bone directions (for symmetric rest pose)
                # This ensures left/right bones have mirrored orientations
                num_smpl_joints = len(SMPL_JOINT_NAMES)
                tails = np.zeros((num_smpl_joints, 3))

                for i, joint_name in enumerate(SMPL_JOINT_NAMES):
                    # Get canonical bone direction
                    direction = np.array(SMPL_BONE_DIRECTIONS.get(joint_name, [0, 1, 0]))

                    # Compute bone length from child distance or use default
                    children = [j for j, p in enumerate(parents_list) if p == i]
                    if children:
                        # Use distance to first child as bone length
                        child_idx = children[0]
                        bone_length = np.linalg.norm(bone_joints[child_idx] - bone_joints[i])
                        if bone_length < 0.01:
                            bone_length = SMPL_DEFAULT_BONE_LENGTH
                    else:
                        # Leaf bone - use default length
                        bone_length = SMPL_DEFAULT_BONE_LENGTH

                    # Tail = head + direction * length
                    tails[i] = bone_joints[i] + direction * bone_length

                print(f"[UniRigExtractSkeletonNew] Converted to SMPL: {len(names_list)} joints with canonical bone orientations")

                # === STEP 1: Detect current facing direction and rotate to SMPL standard ===
                # SMPL standard (before Y-up conversion): facing -Y, lateral along X, up along Z
                # We need to detect current orientation and rotate to match

                # Get shoulder positions to determine lateral axis
                l_shoulder_idx = names_list.index('L_Shoulder')
                r_shoulder_idx = names_list.index('R_Shoulder')
                pelvis_idx = names_list.index('Pelvis')
                head_idx = names_list.index('Head') if 'Head' in names_list else names_list.index('Neck')

                l_shoulder = bone_joints[l_shoulder_idx]
                r_shoulder = bone_joints[r_shoulder_idx]
                pelvis = bone_joints[pelvis_idx]
                head = bone_joints[head_idx]

                # Compute current orientation vectors
                shoulder_vec = r_shoulder - l_shoulder  # Left to Right
                spine_vec = head - pelvis  # Up direction

                # Normalize
                shoulder_vec = shoulder_vec / (np.linalg.norm(shoulder_vec) + 1e-8)
                spine_vec = spine_vec / (np.linalg.norm(spine_vec) + 1e-8)

                # Forward = cross(right, up) for right-handed system
                forward_vec = np.cross(shoulder_vec, spine_vec)
                forward_vec = forward_vec / (np.linalg.norm(forward_vec) + 1e-8)

                print(f"[UniRigExtractSkeletonNew] Current orientation - Lateral: {shoulder_vec}, Up: {spine_vec}, Forward: {forward_vec}")

                # Determine which axis is lateral (should be X for SMPL)
                # In Blender Z-up, SMPL standard is: lateral=X, up=Z, forward=-Y
                lateral_axis = np.argmax(np.abs(shoulder_vec))
                up_axis = np.argmax(np.abs(spine_vec))

                # Check if we need to rotate around Z axis to align lateral with X
                if lateral_axis == 0:
                    # Already aligned with X
                    print(f"[UniRigExtractSkeletonNew] Lateral axis already aligned with X")
                    z_rotation_angle = 0
                elif lateral_axis == 1:
                    # Lateral is along Y, need to rotate 90° around Z
                    z_rotation_angle = np.pi / 2 if shoulder_vec[1] > 0 else -np.pi / 2
                    print(f"[UniRigExtractSkeletonNew] Rotating {np.degrees(z_rotation_angle):.0f}° around Z to align lateral with X")
                else:
                    # Lateral is along Z (our current case), need to rotate around up axis
                    # This shouldn't happen in Z-up Blender coords, but handle it
                    z_rotation_angle = 0
                    print(f"[UniRigExtractSkeletonNew] Unusual: Lateral along Z axis")

                # For the current mesh: lateral is along Y (in original coords), up is along Z
                # After Z-up to Y-up conversion, this becomes: lateral along Y, up along Y - wrong!
                # We need to rotate so lateral is along X before the conversion

                # Actually, let's detect more carefully:
                # If shoulders differ mainly in Y, we need 90° rotation around Z
                if abs(shoulder_vec[1]) > abs(shoulder_vec[0]) and abs(shoulder_vec[1]) > 0.5:
                    # Lateral is along Y, rotate 90° around Z
                    cos_a, sin_a = 0, 1  # 90 degrees
                    if shoulder_vec[1] < 0:
                        sin_a = -1  # -90 degrees

                    def rotate_around_z(points):
                        """Rotate points 90° around Z axis"""
                        rotated = np.zeros_like(points)
                        rotated[..., 0] = cos_a * points[..., 0] - sin_a * points[..., 1]
                        rotated[..., 1] = sin_a * points[..., 0] + cos_a * points[..., 1]
                        rotated[..., 2] = points[..., 2]
                        return rotated

                    print(f"[UniRigExtractSkeletonNew] Rotating 90° around Z to align shoulders with X axis")
                    bone_joints = rotate_around_z(bone_joints)
                    tails = rotate_around_z(tails)
                    mesh_vertices = rotate_around_z(mesh_vertices)
                    vertex_normals = rotate_around_z(vertex_normals)
                    face_normals = rotate_around_z(face_normals)

                # === STEP 2: Rotate from Blender Z-up to SMPL Y-up ===
                # This is a -90° rotation around X axis: (x, y, z) -> (x, z, -y)
                # SMPL uses: X=right, Y=up, Z=back
                # Blender uses: X=right, Y=forward, Z=up
                def rotate_to_smpl_coords(points):
                    """Rotate points from Blender coords (Z-up) to SMPL coords (Y-up)"""
                    rotated = np.zeros_like(points)
                    rotated[..., 0] = points[..., 0]   # X stays X
                    rotated[..., 1] = points[..., 2]   # Z becomes Y (up)
                    rotated[..., 2] = -points[..., 1]  # -Y becomes Z (back)
                    return rotated

                # Rotate joints, tails, mesh vertices, and normals
                bone_joints = rotate_to_smpl_coords(bone_joints)
                tails = rotate_to_smpl_coords(tails)
                mesh_vertices = rotate_to_smpl_coords(mesh_vertices)
                vertex_normals = rotate_to_smpl_coords(vertex_normals)
                face_normals = rotate_to_smpl_coords(face_normals)

                # === STEP 3: Ensure correct handedness (L_Shoulder at +X, R_Shoulder at -X) ===
                # After rotation, check if left/right are correct
                l_shoulder_new = bone_joints[l_shoulder_idx]
                r_shoulder_new = bone_joints[r_shoulder_idx]

                # In SMPL, L_Shoulder should have positive X, R_Shoulder negative X
                if l_shoulder_new[0] < r_shoulder_new[0]:
                    # Left/Right are swapped, need to mirror along X
                    print(f"[UniRigExtractSkeletonNew] Mirroring along X to fix left/right")
                    bone_joints[..., 0] = -bone_joints[..., 0]
                    tails[..., 0] = -tails[..., 0]
                    mesh_vertices[..., 0] = -mesh_vertices[..., 0]
                    vertex_normals[..., 0] = -vertex_normals[..., 0]
                    face_normals[..., 0] = -face_normals[..., 0]
                    # Also need to flip face winding
                    mesh_faces = mesh_faces[:, ::-1]

                # Update mesh bounds after rotation
                mesh_bounds_min = mesh_vertices.min(axis=0)
                mesh_bounds_max = mesh_vertices.max(axis=0)
                mesh_center = (mesh_bounds_min + mesh_bounds_max) / 2

                print(f"[UniRigExtractSkeletonNew] Rotated to SMPL Y-up coordinate system")

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
                cls=cls_value
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
                "output_format": "smpl" if remap_to_smpl else ("mixamo" if remap_to_mixamo else "vroid"),
            }

            if 'bone_to_head_vertex' in skeleton_data:
                skeleton['bone_to_head_vertex'] = skeleton_data['bone_to_head_vertex'].tolist()

            # Close npz file to release handle (required for Windows temp cleanup)
            skeleton_data.close()

            # Windows-specific: Force garbage collection to release file handles
            # This helps prevent "file in use by another process" errors during temp cleanup
            if sys.platform == 'win32':
                import gc
                gc.collect()

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
