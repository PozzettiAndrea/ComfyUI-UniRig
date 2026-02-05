"""
Skeleton I/O nodes for UniRig - Save, Load, and Preview operations.
"""

import os
import subprocess
import tempfile
import numpy as np
import time
import shutil
import pickle
import json
import folder_paths

MESH_INFO_TIMEOUT = 30  # seconds

try:
    from .base import (
        BLENDER_PARSE_SKELETON,
        BLENDER_EXTRACT_MESH_INFO,
        NODE_DIR,
        LIB_DIR,
    )
except ImportError:
    from base import (
        BLENDER_PARSE_SKELETON,
        BLENDER_EXTRACT_MESH_INFO,
        NODE_DIR,
        LIB_DIR,
    )

# Blender executable no longer used - set to None for backwards compatibility
BLENDER_EXE = None


# Direct bone debug extraction module (bpy as Python module)
_DIRECT_BONE_DEBUG_MODULE = None


def _get_direct_bone_debug():
    """Get the direct bone debug extraction module for in-process extraction using bpy."""
    global _DIRECT_BONE_DEBUG_MODULE
    if _DIRECT_BONE_DEBUG_MODULE is None:
        debug_path = os.path.join(LIB_DIR, "unirig", "src", "inference", "direct_extract_bone_debug.py")
        if os.path.exists(debug_path):
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location("unirig_direct_bone_debug", debug_path)
                _DIRECT_BONE_DEBUG_MODULE = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(_DIRECT_BONE_DEBUG_MODULE)
                print(f"[UniRig] Loaded direct bone debug module from {debug_path}")
            except ImportError as e:
                print(f"[UniRig] Direct bone debug not available (bpy not installed): {e}")
                _DIRECT_BONE_DEBUG_MODULE = False
            except Exception as e:
                print(f"[UniRig] Warning: Could not load direct bone debug module: {e}")
                _DIRECT_BONE_DEBUG_MODULE = False
        else:
            print(f"[UniRig] Warning: Direct bone debug module not found at {debug_path}")
            _DIRECT_BONE_DEBUG_MODULE = False
    return _DIRECT_BONE_DEBUG_MODULE if _DIRECT_BONE_DEBUG_MODULE else None


class UniRigLoadRiggedMesh:
    """
    Load a rigged FBX file from disk.

    Loads existing FBX files with rigging/skeleton data, allowing you to
    preview and work with pre-rigged models.
    """

    @classmethod
    def INPUT_TYPES(cls):
        input_files = cls.get_fbx_files_from_input()
        output_files = cls.get_fbx_files_from_output()
        all_files = sorted(set(input_files + output_files)) or [""]

        return {
            "required": {
                "fbx_file": (all_files, {"file_upload": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("fbx_output_path", "info")
    FUNCTION = "load"
    CATEGORY = "unirig"

    @classmethod
    def get_fbx_files_from_input(cls):
        """Get list of available FBX files in input folder."""
        fbx_files = []
        input_dir = folder_paths.get_input_directory()

        if input_dir is not None and os.path.exists(input_dir):
            for root, dirs, files in os.walk(input_dir):
                for file in files:
                    if file.lower().endswith('.fbx'):
                        rel_path = os.path.relpath(os.path.join(root, file), input_dir)
                        rel_path = rel_path.replace(os.sep, '/')  # Normalize to forward slashes for cross-platform
                        fbx_files.append(rel_path)

        return sorted(fbx_files)

    @classmethod
    def get_fbx_files_from_output(cls):
        """Get list of available FBX files in output folder."""
        fbx_files = []
        output_dir = folder_paths.get_output_directory()

        if output_dir is not None and os.path.exists(output_dir):
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    if file.lower().endswith('.fbx'):
                        rel_path = os.path.relpath(os.path.join(root, file), output_dir)
                        rel_path = rel_path.replace(os.sep, '/')  # Normalize to forward slashes for cross-platform
                        fbx_files.append(rel_path)

        return sorted(fbx_files)

    def load(self, fbx_file):
        """Load an FBX file and return its filename in output directory."""
        print(f"[UniRigLoadRiggedMesh] Loading FBX file: {fbx_file}")

        if not fbx_file:
            raise RuntimeError("No FBX file specified. Please select or upload an FBX file.")

        # Search in both input and output directories
        input_dir = folder_paths.get_input_directory()
        output_dir = folder_paths.get_output_directory()

        input_path = os.path.join(input_dir, fbx_file)
        output_path = os.path.join(output_dir, fbx_file)

        # Find the file
        if os.path.exists(output_path):
            fbx_path = output_path
            source = "output"
        elif os.path.exists(input_path):
            fbx_path = input_path
            source = "input"
        else:
            raise RuntimeError(f"FBX file not found: {fbx_file}")

        # If loading from input, copy to output directory
        if source == "input":
            output_filename = f"loaded_{int(time.time())}_{os.path.basename(fbx_file)}"
            final_output_path = os.path.join(output_dir, output_filename)
            shutil.copy(fbx_path, final_output_path)
            print(f"[UniRigLoadRiggedMesh] Copied from input to output: {output_filename}")
        else:
            output_filename = fbx_file
            final_output_path = output_path
            print(f"[UniRigLoadRiggedMesh] Using existing file from output: {output_filename}")

        # Extract mesh info with Blender (using original path for analysis)
        temp_dir = folder_paths.get_temp_directory()
        mesh_info = {}
        try:
            blender_exe = os.environ.get('UNIRIG_BLENDER_EXECUTABLE', BLENDER_EXE)

            if blender_exe and os.path.exists(blender_exe):
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
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=MESH_INFO_TIMEOUT, encoding='utf-8', errors='replace')

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

                    # Close npz file before removing (required for Windows)
                    data.close()
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

        # Parse FBX with Blender to get skeleton info
        skeleton_info = {}
        try:
            blender_exe = os.environ.get('UNIRIG_BLENDER_EXECUTABLE', BLENDER_EXE)

            if blender_exe and os.path.exists(blender_exe):
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
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=MESH_INFO_TIMEOUT, encoding='utf-8', errors='replace')

                if os.path.exists(skeleton_npz):
                    data = np.load(skeleton_npz, allow_pickle=True)

                    num_bones = len(data.get('bone_names', []))
                    bone_names = [str(name) for name in data.get('bone_names', [])]

                    vertices = data.get('vertices', np.array([]))
                    skeleton_extents = None
                    if len(vertices) > 0:
                        min_coords = vertices.min(axis=0)
                        max_coords = vertices.max(axis=0)
                        skeleton_extents = (max_coords - min_coords).tolist()

                    skeleton_info = {
                        "num_bones": num_bones,
                        "bone_names": bone_names[:10],
                        "has_skeleton": num_bones > 0,
                        "skeleton_extents": skeleton_extents
                    }

                    # Close npz file before removing (required for Windows)
                    data.close()
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

        # Create info string
        file_size = os.path.getsize(final_output_path)
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

        print(f"[UniRigLoadRiggedMesh] Loaded successfully")
        print(info_string)

        return (final_output_path, info_string)


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
                "fbx_output_path": ("STRING", {
                    "tooltip": "FBX filename from output directory (from UniRigApplySkinningMLNew or UniRigLoadRiggedMesh)"
                }),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "preview"
    CATEGORY = "unirig"

    def preview(self, fbx_output_path):
        """Preview the rigged mesh in an interactive FBX viewer."""
        print(f"[UniRigPreviewRiggedMesh] Preparing preview...")

        # FBX should already be in output directory
        output_dir = folder_paths.get_output_directory()
        fbx_path = os.path.join(output_dir, fbx_output_path)

        if not os.path.exists(fbx_path):
            raise RuntimeError(f"FBX file not found in output directory: {fbx_output_path}")

        print(f"[UniRigPreviewRiggedMesh] FBX path: {fbx_path}")

        # FBX is already in output, so viewer can access it directly
        # Assume all FBX files have skinning and skeleton
        has_skinning = True
        has_skeleton = True

        print(f"[UniRigPreviewRiggedMesh] Has skinning: {has_skinning}")
        print(f"[UniRigPreviewRiggedMesh] Has skeleton: {has_skeleton}")

        return {
            "ui": {
                "fbx_file": [fbx_output_path],
                "has_skinning": [bool(has_skinning)],
                "has_skeleton": [bool(has_skeleton)],
            }
        }


class UniRigExportPosedFBX:
    """
    Export rigged mesh with custom bone pose to FBX.

    Takes a rigged mesh and bone transform data, applies the pose,
    and exports the result as FBX using Blender.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rigged_mesh": ("RIGGED_MESH",),
                "output_filename": ("STRING", {
                    "default": "posed_export.fbx",
                    "tooltip": "Output filename for the posed FBX"
                }),
            },
            "optional": {
                "bone_transforms_json": ("STRING", {
                    "default": "{}",
                    "multiline": True,
                    "tooltip": "JSON string containing bone transforms (name -> {position, quaternion, scale})"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filepath",)
    FUNCTION = "export_posed_fbx"
    CATEGORY = "unirig"
    OUTPUT_NODE = True

    def export_posed_fbx(self, rigged_mesh, output_filename, bone_transforms_json="{}"):
        """Export rigged mesh with custom pose to FBX using bpy directly."""
        print(f"[UniRigExportPosedFBX] Exporting posed FBX...")

        # Get original FBX path
        fbx_path = rigged_mesh.get("fbx_path")
        if not fbx_path or not os.path.exists(fbx_path):
            raise RuntimeError(f"Rigged mesh FBX not found: {fbx_path}")

        print(f"[UniRigExportPosedFBX] Source FBX: {fbx_path}")

        # Parse bone transforms
        try:
            bone_transforms = json.loads(bone_transforms_json)
            print(f"[UniRigExportPosedFBX] Loaded transforms for {len(bone_transforms)} bones")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in bone_transforms_json: {e}")

        # Prepare output path
        output_dir = folder_paths.get_output_directory()
        if not output_filename.endswith('.fbx'):
            output_filename = output_filename + '.fbx'
        output_fbx_path = os.path.join(output_dir, output_filename)

        # Use direct bpy export
        try:
            from .lib.direct_export_posed_fbx import export_posed_fbx as direct_export
            direct_export(fbx_path, output_fbx_path, bone_transforms)
        except ImportError as e:
            raise RuntimeError(
                f"Failed to import direct_export_posed_fbx: {e}\n"
                "Make sure bpy is available in your environment (unirig isolated environment)."
            )

        if not os.path.exists(output_fbx_path):
            raise RuntimeError(f"Export completed but output file not found: {output_fbx_path}")

        print(f"[UniRigExportPosedFBX] [OK] Successfully exported to: {output_fbx_path}")

        return (output_fbx_path,)


class UniRigViewRigging:
    """
    View rigging debug information.

    Displays skeleton bones with names, roll/rotation values, and other
    debugging information in an interactive 3D viewer.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fbx_output_path": ("STRING", {
                    "tooltip": "FBX filename from output directory"
                }),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "view_rigging"
    CATEGORY = "unirig"

    def view_rigging(self, fbx_output_path):
        """View rigging debug information for the FBX file."""
        print(f"[UniRigViewRigging] Preparing debug view...")

        # FBX should already be in output directory
        output_dir = folder_paths.get_output_directory()

        # Handle both relative paths and absolute paths
        if os.path.isabs(fbx_output_path):
            fbx_path = fbx_output_path
        else:
            fbx_path = os.path.join(output_dir, fbx_output_path)

        if not os.path.exists(fbx_path):
            raise RuntimeError(f"FBX file not found: {fbx_output_path}")

        print(f"[UniRigViewRigging] FBX path: {fbx_path}")

        # Extract bone debug data using direct bpy module
        bone_debug_data = None
        bone_debug_module = _get_direct_bone_debug()

        if bone_debug_module:
            try:
                bone_debug_data = bone_debug_module.extract_bone_debug(fbx_path)
                print(f"[UniRigViewRigging] Extracted debug data for {bone_debug_data.get('bone_count', 0)} bones")
            except Exception as e:
                print(f"[UniRigViewRigging] Warning: Could not extract bone debug data: {e}")
                bone_debug_data = {'error': str(e), 'bones': [], 'bone_count': 0}
        else:
            print("[UniRigViewRigging] Warning: bpy module not available, bone debug data will be limited")
            bone_debug_data = {'error': 'bpy module not available', 'bones': [], 'bone_count': 0}

        # Return data for the viewer widget
        # For relative path, just use the filename for the viewer
        if os.path.isabs(fbx_output_path):
            viewer_filename = os.path.basename(fbx_output_path)
        else:
            viewer_filename = fbx_output_path

        return {
            "ui": {
                "fbx_file": [viewer_filename],
                "bone_debug_data": [json.dumps(bone_debug_data)],
            }
        }


class UniRigDebugSkeleton:
    """
    Debug skeleton visualization with bone roll/orientation analysis.

    Opens the FBX in an enhanced debug viewer with:
    - RGB axes showing local bone coordinate systems (X=red, Y=green, Z=blue/roll)
    - Bone name labels
    - Detailed bone information panel with roll angles
    - Animation playback controls
    - Bone filtering and size controls
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fbx_path": ("STRING", {
                    "tooltip": "Path to FBX file (from output directory or absolute path)"
                }),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "debug_skeleton"
    CATEGORY = "unirig"

    def debug_skeleton(self, fbx_path):
        """Open the FBX in the debug skeleton viewer."""
        print(f"[UniRigDebugSkeleton] Preparing debug skeleton view...")

        # Handle both relative paths and absolute paths
        output_dir = folder_paths.get_output_directory()

        if os.path.isabs(fbx_path):
            full_path = fbx_path
        else:
            full_path = os.path.join(output_dir, fbx_path)

        if not os.path.exists(full_path):
            raise RuntimeError(f"FBX file not found: {fbx_path}")

        print(f"[UniRigDebugSkeleton] FBX path: {full_path}")

        # For the viewer, use relative path if in output, otherwise basename
        if os.path.isabs(fbx_path):
            viewer_filename = os.path.basename(fbx_path)
        else:
            viewer_filename = fbx_path

        return {
            "ui": {
                "fbx_file": [viewer_filename],
            }
        }


class UniRigCompareSkeletons:
    """
    Compare two skeletons side-by-side with synced rotation.

    Opens two FBX files in a split-view debug viewer where:
    - Both skeletons are displayed side-by-side
    - Camera rotation and zoom are synced between views
    - Clicking a bone in one view highlights the matching bone (by name) in the other
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fbx_path_left": ("STRING", {
                    "tooltip": "Path to left skeleton FBX file (from output directory or absolute path)"
                }),
                "fbx_path_right": ("STRING", {
                    "tooltip": "Path to right skeleton FBX file (from output directory or absolute path)"
                }),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "compare_skeletons"
    CATEGORY = "unirig"

    def compare_skeletons(self, fbx_path_left, fbx_path_right):
        """Open both FBX files in the comparison skeleton viewer."""
        print(f"[UniRig Compare] Preparing skeleton comparison view...")

        output_dir = folder_paths.get_output_directory()

        # Validate left FBX path
        if os.path.isabs(fbx_path_left):
            full_path_left = fbx_path_left
        else:
            full_path_left = os.path.join(output_dir, fbx_path_left)

        if not os.path.exists(full_path_left):
            raise RuntimeError(f"Left FBX file not found: {fbx_path_left}")

        # Validate right FBX path
        if os.path.isabs(fbx_path_right):
            full_path_right = fbx_path_right
        else:
            full_path_right = os.path.join(output_dir, fbx_path_right)

        if not os.path.exists(full_path_right):
            raise RuntimeError(f"Right FBX file not found: {fbx_path_right}")

        print(f"[UniRig Compare] Left FBX: {full_path_left}")
        print(f"[UniRig Compare] Right FBX: {full_path_right}")

        # For the viewer, use relative path if in output, otherwise basename
        if os.path.isabs(fbx_path_left):
            viewer_filename_left = os.path.basename(fbx_path_left)
        else:
            viewer_filename_left = fbx_path_left

        if os.path.isabs(fbx_path_right):
            viewer_filename_right = os.path.basename(fbx_path_right)
        else:
            viewer_filename_right = fbx_path_right

        return {
            "ui": {
                "fbx_file_left": [viewer_filename_left],
                "fbx_file_right": [viewer_filename_right],
            }
        }
