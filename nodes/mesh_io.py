"""
UniRig Mesh I/O Nodes - Load and save mesh files
"""

import os
import subprocess
import tempfile
import numpy as np
import trimesh
import igl
from pathlib import Path
from typing import Tuple, Optional

# ComfyUI folder paths
try:
    import folder_paths
    COMFYUI_INPUT_FOLDER = folder_paths.get_input_directory()
    COMFYUI_OUTPUT_FOLDER = folder_paths.get_output_directory()
except:
    # Fallback if folder_paths not available (e.g., during testing)
    COMFYUI_INPUT_FOLDER = None
    COMFYUI_OUTPUT_FOLDER = None

# Import Blender path from base module
try:
    from .base import BLENDER_EXE, LIB_DIR
except ImportError:
    from base import BLENDER_EXE, LIB_DIR

# Blender FBX loading script
BLENDER_LOAD_FBX = str(LIB_DIR / "blender_load_fbx.py")

# Timeout for Blender operations (seconds)
BLENDER_TIMEOUT = 120


def load_fbx_with_blender(file_path: str) -> Tuple[Optional[trimesh.Trimesh], str]:
    """
    Load an FBX file using Blender as an intermediary.

    FBX files cannot be loaded directly by trimesh/igl, so we use Blender
    to convert FBX to GLB format, which trimesh can handle.

    Args:
        file_path: Path to FBX file

    Returns:
        Tuple of (mesh, error_message)
    """
    if not BLENDER_EXE or not os.path.exists(BLENDER_EXE):
        return None, (
            "FBX loading requires Blender. "
            "Install Blender and add to PATH, or run: python blender_install.py"
        )

    if not os.path.exists(BLENDER_LOAD_FBX):
        return None, f"Blender FBX script not found: {BLENDER_LOAD_FBX}"

    print(f"[UniRigLoadMesh] Loading FBX via Blender: {file_path}")

    try:
        # Create a temporary GLB file
        with tempfile.TemporaryDirectory() as tmpdir:
            glb_path = os.path.join(tmpdir, "converted.glb")

            # Run Blender to convert FBX to GLB
            cmd = [
                BLENDER_EXE,
                "--background",
                "--python", BLENDER_LOAD_FBX,
                "--",
                file_path,
                glb_path
            ]

            print(f"[UniRigLoadMesh] Running Blender FBX conversion...")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=BLENDER_TIMEOUT
            )

            # Log output for debugging
            if result.stdout:
                for line in result.stdout.strip().split('\n'):
                    if line.startswith('[Blender'):
                        print(f"[UniRigLoadMesh] {line}")

            if result.returncode != 0:
                error_msg = f"Blender FBX conversion failed (code {result.returncode})"
                if result.stderr:
                    error_msg += f": {result.stderr[:500]}"
                return None, error_msg

            if not os.path.exists(glb_path):
                return None, "Blender did not produce output GLB file"

            print(f"[UniRigLoadMesh] Loading converted GLB: {glb_path}")

            # Load the converted GLB with trimesh
            loaded = trimesh.load(glb_path, force='mesh')

            # Handle Scene vs Mesh
            if isinstance(loaded, trimesh.Scene):
                mesh = loaded.dump(concatenate=True)
            else:
                mesh = loaded

            if mesh is None or len(mesh.vertices) == 0:
                return None, "Converted mesh is empty"

            # Store original FBX metadata
            mesh.metadata['file_path'] = file_path
            mesh.metadata['file_name'] = os.path.basename(file_path)
            mesh.metadata['file_format'] = '.fbx'
            mesh.metadata['loaded_via'] = 'blender'

            print(f"[UniRigLoadMesh] FBX loaded via Blender: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
            return mesh, ""

    except subprocess.TimeoutExpired:
        return None, f"Blender FBX conversion timed out after {BLENDER_TIMEOUT}s"
    except Exception as e:
        return None, f"FBX loading error: {str(e)}"


def load_mesh_file(file_path: str) -> Tuple[Optional[trimesh.Trimesh], str]:
    """
    Load a mesh from file.

    Ensures the returned mesh has only triangular faces and is properly processed.

    Args:
        file_path: Path to mesh file (OBJ, PLY, STL, OFF, FBX, etc.)

    Returns:
        Tuple of (mesh, error_message)
    """
    if not os.path.exists(file_path):
        return None, f"File not found: {file_path}"

    # Check file extension - FBX requires Blender
    ext = Path(file_path).suffix.lower()
    if ext in ['.fbx']:
        print(f"[UniRigLoadMesh] Detected FBX file, using Blender loader")
        return load_fbx_with_blender(file_path)

    try:
        print(f"[UniRigLoadMesh] Loading: {file_path}")

        # Try to load with trimesh first (supports many formats)
        loaded = trimesh.load(file_path, force='mesh')

        print(f"[UniRigLoadMesh] Loaded type: {type(loaded).__name__}")

        # Handle case where trimesh.load returns a Scene instead of a mesh
        if isinstance(loaded, trimesh.Scene):
            print(f"[UniRigLoadMesh] Converting Scene to single mesh (scene has {len(loaded.geometry)} geometries)")
            # If it's a scene, dump it to a single mesh
            mesh = loaded.dump(concatenate=True)
        else:
            mesh = loaded

        if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            return None, f"Failed to read mesh or mesh is empty: {file_path}"

        print(f"[UniRigLoadMesh] Initial mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

        # Ensure mesh is properly triangulated
        if hasattr(mesh, 'faces') and len(mesh.faces) > 0:
            # Check if faces are triangular
            if mesh.faces.shape[1] != 3:
                print(f"[UniRigLoadMesh] Warning: Mesh has non-triangular faces, triangulating...")
                mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=True)
                print(f"[UniRigLoadMesh] After triangulation: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

        # Count before cleanup
        verts_before = len(mesh.vertices)
        faces_before = len(mesh.faces)

        # Merge duplicate vertices and clean up
        mesh.merge_vertices()

        # Remove duplicate and degenerate faces (trimesh 4.x compatible)
        # unique_faces() returns boolean mask of non-duplicate faces
        # nondegenerate_faces() returns boolean mask of non-degenerate faces
        unique_mask = mesh.unique_faces()
        nondegenerate_mask = mesh.nondegenerate_faces()
        valid_faces_mask = unique_mask & nondegenerate_mask
        if not valid_faces_mask.all():
            mesh.update_faces(valid_faces_mask)

        verts_after = len(mesh.vertices)
        faces_after = len(mesh.faces)

        if verts_before != verts_after or faces_before != faces_after:
            print(f"[UniRigLoadMesh] Cleanup: {verts_before}->{verts_after} vertices, {faces_before}->{faces_after} faces")
            print(f"[UniRigLoadMesh]   Removed: {verts_before - verts_after} duplicate vertices, {faces_before - faces_after} bad faces")

        # Store file metadata
        mesh.metadata['file_path'] = file_path
        mesh.metadata['file_name'] = os.path.basename(file_path)
        mesh.metadata['file_format'] = os.path.splitext(file_path)[1].lower()

        print(f"[UniRigLoadMesh] Successfully loaded: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        return mesh, ""

    except Exception as e:
        print(f"[UniRigLoadMesh] Trimesh failed: {str(e)}, trying libigl fallback...")
        # Fallback to libigl
        try:
            v, f = igl.read_triangle_mesh(file_path)
            if v is None or f is None or len(v) == 0 or len(f) == 0:
                return None, f"Failed to read mesh: {file_path}"

            print(f"[UniRigLoadMesh] libigl loaded: {len(v)} vertices, {len(f)} faces")

            mesh = trimesh.Trimesh(vertices=v, faces=f, process=True)

            # Count before cleanup
            verts_before = len(mesh.vertices)
            faces_before = len(mesh.faces)

            # Clean up the mesh
            mesh.merge_vertices()

            # Remove duplicate and degenerate faces (trimesh 4.x compatible)
            unique_mask = mesh.unique_faces()
            nondegenerate_mask = mesh.nondegenerate_faces()
            valid_faces_mask = unique_mask & nondegenerate_mask
            if not valid_faces_mask.all():
                mesh.update_faces(valid_faces_mask)

            verts_after = len(mesh.vertices)
            faces_after = len(mesh.faces)

            if verts_before != verts_after or faces_before != faces_after:
                print(f"[UniRigLoadMesh] Cleanup: {verts_before}->{verts_after} vertices, {faces_before}->{faces_after} faces")

            # Store metadata
            mesh.metadata['file_path'] = file_path
            mesh.metadata['file_name'] = os.path.basename(file_path)
            mesh.metadata['file_format'] = os.path.splitext(file_path)[1].lower()

            print(f"[UniRigLoadMesh] Successfully loaded via libigl: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
            return mesh, ""
        except Exception as e2:
            print(f"[UniRigLoadMesh] Both loaders failed!")
            return None, f"Error loading mesh: {str(e)}; Fallback error: {str(e2)}"


def save_mesh_file(mesh: trimesh.Trimesh, file_path: str) -> Tuple[bool, str]:
    """
    Save a mesh to file.

    Args:
        mesh: Trimesh object
        file_path: Output file path

    Returns:
        Tuple of (success, error_message)
    """
    if not isinstance(mesh, trimesh.Trimesh):
        return False, "Input must be a trimesh.Trimesh object"

    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        return False, "Mesh is empty"

    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(file_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Export the mesh
        mesh.export(file_path)

        return True, ""

    except Exception as e:
        return False, f"Error saving mesh: {str(e)}"


class UniRigLoadMesh:
    """
    Load a mesh from ComfyUI input or output folder (OBJ, PLY, STL, OFF, etc.)
    Returns trimesh.Trimesh objects for mesh handling.
    """

    # Supported mesh extensions for file browser
    SUPPORTED_EXTENSIONS = ['.obj', '.ply', '.stl', '.off', '.gltf', '.glb', '.fbx', '.dae', '.3ds', '.vtp']

    @classmethod
    def INPUT_TYPES(cls):
        # Get list of available mesh files from input folder (default)
        mesh_files = cls.get_mesh_files_from_input()

        # If no files found, provide a default message
        if not mesh_files:
            mesh_files = ["No mesh files found"]

        return {
            "required": {
                "source_folder": (["input", "output"], {
                    "default": "input",
                    "tooltip": "Source folder to load mesh from (ComfyUI input or output directory)"
                }),
                "file_path": (mesh_files, {
                    "tooltip": "Mesh file to load. Refresh the node after changing source_folder."
                }),
            },
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "load_mesh"
    CATEGORY = "UniRig/IO"

    @classmethod
    def get_mesh_files_from_input(cls):
        """Get list of available mesh files in input/3d and input folders."""
        mesh_files = []

        if COMFYUI_INPUT_FOLDER is not None:
            # Scan input/3d first
            input_3d = os.path.join(COMFYUI_INPUT_FOLDER, "3d")
            if os.path.exists(input_3d):
                for file in os.listdir(input_3d):
                    if any(file.lower().endswith(ext) for ext in cls.SUPPORTED_EXTENSIONS):
                        mesh_files.append(f"3d/{file}")

            # Then scan input root
            for file in os.listdir(COMFYUI_INPUT_FOLDER):
                file_path = os.path.join(COMFYUI_INPUT_FOLDER, file)
                if os.path.isfile(file_path):
                    if any(file.lower().endswith(ext) for ext in cls.SUPPORTED_EXTENSIONS):
                        mesh_files.append(file)

        return sorted(mesh_files)

    @classmethod
    def get_mesh_files_from_output(cls):
        """Get list of available mesh files in output folder."""
        mesh_files = []

        if COMFYUI_OUTPUT_FOLDER is not None and os.path.exists(COMFYUI_OUTPUT_FOLDER):
            # Scan output folder recursively
            for root, dirs, files in os.walk(COMFYUI_OUTPUT_FOLDER):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in cls.SUPPORTED_EXTENSIONS):
                        # Get relative path from output folder
                        full_path = os.path.join(root, file)
                        rel_path = os.path.relpath(full_path, COMFYUI_OUTPUT_FOLDER)
                        mesh_files.append(rel_path)

        return sorted(mesh_files)

    @classmethod
    def IS_CHANGED(cls, source_folder, file_path):
        """Force re-execution when file changes."""
        base_folder = COMFYUI_INPUT_FOLDER if source_folder == "input" else COMFYUI_OUTPUT_FOLDER

        if base_folder is not None:
            if source_folder == "input":
                # Check in input/3d first, then input root
                input_3d_path = os.path.join(base_folder, "3d", file_path)
                input_path = os.path.join(base_folder, file_path)

                if os.path.exists(input_3d_path):
                    return os.path.getmtime(input_3d_path)
                elif os.path.exists(input_path):
                    return os.path.getmtime(input_path)
            else:
                # Check in output folder
                full_path = os.path.join(base_folder, file_path)
                if os.path.exists(full_path):
                    return os.path.getmtime(full_path)

        return f"{source_folder}:{file_path}"

    def load_mesh(self, source_folder, file_path):
        """
        Load mesh from file.

        Looks for files in the specified source folder (input or output).

        Args:
            source_folder: "input" or "output"
            file_path: Path to mesh file (relative to source folder or absolute)

        Returns:
            tuple: (trimesh.Trimesh,)
        """
        if not file_path or file_path.strip() == "":
            raise ValueError("File path cannot be empty")

        # Try to find the file
        full_path = None
        searched_paths = []

        if source_folder == "input" and COMFYUI_INPUT_FOLDER is not None:
            # First, try in ComfyUI input/3d folder
            input_3d_path = os.path.join(COMFYUI_INPUT_FOLDER, "3d", file_path)
            searched_paths.append(input_3d_path)
            if os.path.exists(input_3d_path):
                full_path = input_3d_path
                print(f"[UniRigLoadMesh] Found mesh in input/3d folder: {file_path}")

            # Second, try in ComfyUI input folder
            if full_path is None:
                input_path = os.path.join(COMFYUI_INPUT_FOLDER, file_path)
                searched_paths.append(input_path)
                if os.path.exists(input_path):
                    full_path = input_path
                    print(f"[UniRigLoadMesh] Found mesh in input folder: {file_path}")

        elif source_folder == "output" and COMFYUI_OUTPUT_FOLDER is not None:
            output_path = os.path.join(COMFYUI_OUTPUT_FOLDER, file_path)
            searched_paths.append(output_path)
            if os.path.exists(output_path):
                full_path = output_path
                print(f"[UniRigLoadMesh] Found mesh in output folder: {file_path}")

        # If not found in source folder, try as absolute path
        if full_path is None:
            searched_paths.append(file_path)
            if os.path.exists(file_path):
                full_path = file_path
                print(f"[UniRigLoadMesh] Loading from absolute path: {file_path}")
            else:
                # Generate error message with all searched paths
                error_msg = f"File not found: '{file_path}'\nSearched in:"
                for path in searched_paths:
                    error_msg += f"\n  - {path}"
                raise ValueError(error_msg)

        # Load the mesh
        loaded_mesh, error = load_mesh_file(full_path)

        if loaded_mesh is None:
            raise ValueError(f"Failed to load mesh: {error}")

        print(f"[UniRigLoadMesh] Loaded: {len(loaded_mesh.vertices)} vertices, {len(loaded_mesh.faces)} faces")

        return (loaded_mesh,)


class UniRigSaveMesh:
    """
    Save a mesh to file (OBJ, PLY, STL, OFF, etc.)
    Supports all formats provided by trimesh.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "file_path": ("STRING", {
                    "default": "output.obj",
                    "multiline": False
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "save_mesh"
    CATEGORY = "UniRig/IO"
    OUTPUT_NODE = True

    def save_mesh(self, trimesh, file_path):
        """
        Save mesh to file.

        Saves to ComfyUI's output folder if path is relative, otherwise uses absolute path.

        Args:
            trimesh: trimesh.Trimesh object
            file_path: Output file path (relative to output folder or absolute)

        Returns:
            tuple: (status_message,)
        """
        if not file_path or file_path.strip() == "":
            raise ValueError("File path cannot be empty")

        # Debug: Check what we received
        print(f"[UniRigSaveMesh] Received mesh type: {type(trimesh)}")
        if trimesh is None:
            raise ValueError("Cannot save mesh: received None instead of a mesh object. Check that the previous node is outputting a mesh.")

        # Check if mesh has data
        try:
            vertex_count = len(trimesh.vertices) if hasattr(trimesh, 'vertices') else 0
            face_count = len(trimesh.faces) if hasattr(trimesh, 'faces') else 0
            print(f"[UniRigSaveMesh] Mesh has {vertex_count} vertices, {face_count} faces")

            if vertex_count == 0 or face_count == 0:
                raise ValueError(
                    f"Cannot save empty mesh (vertices: {vertex_count}, faces: {face_count}). "
                    "Check that the previous node is producing valid geometry."
                )
        except Exception as e:
            raise ValueError(f"Error checking mesh properties: {e}. Received object may not be a valid mesh.")

        # Determine full output path
        full_path = file_path

        # If path is relative and we have output folder, use it
        if not os.path.isabs(file_path) and COMFYUI_OUTPUT_FOLDER is not None:
            full_path = os.path.join(COMFYUI_OUTPUT_FOLDER, file_path)
            print(f"[UniRigSaveMesh] Saving to output folder: {file_path}")
        else:
            print(f"[UniRigSaveMesh] Saving to: {file_path}")

        # Save the mesh
        success, error = save_mesh_file(trimesh, full_path)

        if not success:
            raise ValueError(f"Failed to save trimesh: {error}")

        status = f"Successfully saved mesh to: {full_path}\n"
        status += f"  Vertices: {len(trimesh.vertices)}\n"
        status += f"  Faces: {len(trimesh.faces)}"

        print(f"[UniRigSaveMesh] {status}")

        return (status,)
