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

# Support both relative imports (ComfyUI) and absolute imports (testing)
try:
    from ..constants import BLENDER_TIMEOUT, PARSE_TIMEOUT, MESH_INFO_TIMEOUT, DEFAULT_EXTRUDE_SIZE
except ImportError:
    from constants import BLENDER_TIMEOUT, PARSE_TIMEOUT, MESH_INFO_TIMEOUT, DEFAULT_EXTRUDE_SIZE

try:
    from .base import (
        BLENDER_EXE,
        BLENDER_PARSE_SKELETON,
        BLENDER_EXTRACT_MESH_INFO,
        NODE_DIR,
    )
except ImportError:
    from base import (
        BLENDER_EXE,
        BLENDER_PARSE_SKELETON,
        BLENDER_EXTRACT_MESH_INFO,
        NODE_DIR,
    )


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

        output_dir = folder_paths.get_output_directory()
        filepath = os.path.join(output_dir, filename)

        if not filepath.endswith(f'.{format}'):
            filepath = os.path.splitext(filepath)[0] + f'.{format}'

        vertices = skeleton['vertices']
        edges = skeleton['edges']

        has_hierarchy = 'bone_names' in skeleton and 'bone_parents' in skeleton

        if format in ['fbx', 'glb']:
            if not has_hierarchy:
                print(f"[UniRigSaveSkeleton] Warning: Skeleton has no hierarchy data. FBX/GLB will only contain line mesh.")
                self._save_line_mesh(vertices, edges, filepath, format)
            else:
                self._save_fbx_with_hierarchy(skeleton, filepath, format)
        else:
            self._save_line_mesh(vertices, edges, filepath, format)

        print(f"[UniRigSaveSkeleton] Saved to: {filepath}")
        return {}

    def _save_line_mesh(self, vertices, edges, filepath, format):
        """Save skeleton as a simple line mesh (OBJ or PLY)."""
        import trimesh

        entities = []
        for edge in edges:
            entities.append(trimesh.path.entities.Line([edge[0], edge[1]]))

        path = trimesh.path.Path3D(
            vertices=vertices,
            entities=entities
        )

        path.export(filepath, file_type=format)

    def _save_fbx_with_hierarchy(self, skeleton, filepath, format):
        """Save skeleton with full hierarchy using Blender."""
        vertices = skeleton['vertices']
        bone_names = skeleton['bone_names']
        bone_parents = skeleton['bone_parents']
        bone_to_head_vertex = skeleton['bone_to_head_vertex']

        # Denormalize vertices
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
        joints = []
        for bone_idx in range(len(bone_names)):
            head_vertex_idx = bone_to_head_vertex[bone_idx]
            joints.append(vertices_denorm[head_vertex_idx])

        joints = np.array(joints, dtype=np.float32)

        # Prepare data for Blender export
        data = {
            'joints': joints.tolist(),
            'parents': bone_parents,
            'names': bone_names,
            'vertices': None,
            'faces': None,
            'skin': None,
            'tails': None,
            'group_per_vertex': -1,
            'do_not_normalize': True,
        }

        # Save to temporary pickle file
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
            pickle_path = f.name
            pickle.dump(data, f)

        try:
            wrapper_script = os.path.join(NODE_DIR, 'lib', 'blender_export_fbx.py')

            if not os.path.exists(wrapper_script):
                raise RuntimeError(f"Blender export script not found: {wrapper_script}")

            if not os.path.exists(BLENDER_EXE):
                raise RuntimeError(f"Blender executable not found: {BLENDER_EXE}")

            # Determine output format
            if format == 'glb':
                temp_fbx = filepath.replace('.glb', '_temp.fbx')
                output_path = temp_fbx
            else:
                output_path = filepath

            cmd = [
                BLENDER_EXE,
                '--background',
                '--python', wrapper_script,
                '--',
                pickle_path,
                output_path,
                f'--extrude_size={DEFAULT_EXTRUDE_SIZE}',
            ]

            print(f"[UniRigSaveSkeleton] Running Blender to export {format.upper()}...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=BLENDER_TIMEOUT)

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

        output_dir = folder_paths.get_output_directory()
        filepath = os.path.join(output_dir, filename)

        if not filepath.endswith(f'.{format}'):
            filepath = os.path.splitext(filepath)[0] + f'.{format}'

        source_fbx = rigged_mesh.get("fbx_path")
        if not source_fbx or not os.path.exists(source_fbx):
            raise RuntimeError(f"Rigged mesh FBX not found: {source_fbx}")

        if format == "fbx":
            shutil.copy(source_fbx, filepath)
            print(f"[UniRigSaveRiggedMesh] Saved FBX to: {filepath}")
        elif format == "glb":
            print(f"[UniRigSaveRiggedMesh] Converting FBX to GLB...")
            try:
                import trimesh
                scene = trimesh.load(source_fbx)
                scene.export(filepath)
                print(f"[UniRigSaveRiggedMesh] Saved GLB to: {filepath}")
            except Exception as e:
                print(f"[UniRigSaveRiggedMesh] Warning: GLB conversion failed, saving as FBX: {e}")
                shutil.copy(source_fbx, filepath.replace('.glb', '.fbx'))
                filepath = filepath.replace('.glb', '.fbx')
                print(f"[UniRigSaveRiggedMesh] Saved FBX to: {filepath}")

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
        return {
            "required": {
                "source_folder": (["input", "output"], {
                    "default": "output",
                    "tooltip": "Source folder to load FBX from (ComfyUI input or output directory)"
                }),
                "fbx_file": ("COMBO", {
                    "remote": {
                        "route": "/unirig/fbx_files",
                        "refresh_button": True,
                    },
                }),
            },
        }

    RETURN_TYPES = ("RIGGED_MESH", "STRING")
    RETURN_NAMES = ("rigged_mesh", "info")
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
                        fbx_files.append(rel_path)

        return sorted(fbx_files)

    def load(self, source_folder, fbx_file):
        """Load an FBX file and return it as a RIGGED_MESH."""
        print(f"[UniRigLoadRiggedMesh] Loading FBX file: {fbx_file} from {source_folder}")

        if fbx_file == "No FBX files found":
            raise RuntimeError(f"No FBX files found in ComfyUI/{source_folder} directory. Please add an FBX file first.")

        # Determine base folder based on source_folder
        if source_folder == "input":
            base_dir = folder_paths.get_input_directory()
        else:  # output
            base_dir = folder_paths.get_output_directory()

        fbx_path = os.path.join(base_dir, fbx_file)

        if not os.path.exists(fbx_path):
            raise RuntimeError(f"FBX file not found: {fbx_path}")

        # Copy to temp directory
        temp_dir = folder_paths.get_temp_directory()
        temp_fbx = os.path.join(temp_dir, f"loaded_rigged_{int(time.time())}_{os.path.basename(fbx_file)}")
        shutil.copy(fbx_path, temp_fbx)

        print(f"[UniRigLoadRiggedMesh] Copied to temp: {temp_fbx}")

        # Extract mesh info with Blender
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
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=MESH_INFO_TIMEOUT)

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
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=MESH_INFO_TIMEOUT)

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
            "mesh": None,
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
        """Preview the rigged mesh in an interactive FBX viewer."""
        print(f"[UniRigPreviewRiggedMesh] Preparing preview...")

        fbx_path = rigged_mesh.get("fbx_path")
        if not fbx_path or not os.path.exists(fbx_path):
            raise RuntimeError(f"Rigged mesh FBX not found: {fbx_path}")

        print(f"[UniRigPreviewRiggedMesh] FBX path: {fbx_path}")

        # Copy FBX to output directory
        output_dir = folder_paths.get_output_directory()
        filename = f"rigged_preview_{int(time.time())}.fbx"
        output_fbx_path = os.path.join(output_dir, filename)

        shutil.copy2(fbx_path, output_fbx_path)
        print(f"[UniRigPreviewRiggedMesh] Copied FBX to output: {output_fbx_path}")

        has_skinning = rigged_mesh.get("has_skinning", False)
        has_skeleton = rigged_mesh.get("has_skeleton", False)

        print(f"[UniRigPreviewRiggedMesh] Has skinning: {has_skinning}")
        print(f"[UniRigPreviewRiggedMesh] Has skeleton: {has_skeleton}")

        return {
            "ui": {
                "fbx_file": [filename],
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
        """Export rigged mesh with custom pose to FBX."""
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

        # Save bone transforms to temporary JSON file
        temp_dir = folder_paths.get_temp_directory()
        transforms_json_path = os.path.join(temp_dir, f"bone_transforms_{int(time.time())}.json")
        with open(transforms_json_path, 'w') as f:
            json.dump(bone_transforms, f)

        print(f"[UniRigExportPosedFBX] Saved transforms to: {transforms_json_path}")

        # Prepare output path
        output_dir = folder_paths.get_output_directory()
        if not output_filename.endswith('.fbx'):
            output_filename = output_filename + '.fbx'
        output_fbx_path = os.path.join(output_dir, output_filename)

        try:
            # Path to Blender script
            blender_script = os.path.join(NODE_DIR, 'lib', 'blender_export_posed_fbx.py')

            if not os.path.exists(blender_script):
                raise RuntimeError(f"Blender export script not found: {blender_script}")

            if not os.path.exists(BLENDER_EXE):
                raise RuntimeError(f"Blender executable not found: {BLENDER_EXE}")

            # Build command
            cmd = [
                BLENDER_EXE,
                '--background',
                '--python', blender_script,
                '--',
                fbx_path,
                output_fbx_path,
                transforms_json_path,
            ]

            print(f"[UniRigExportPosedFBX] Running Blender to export posed FBX...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=BLENDER_TIMEOUT)

            if result.returncode != 0:
                print(f"[UniRigExportPosedFBX] Blender stderr: {result.stderr}")
                print(f"[UniRigExportPosedFBX] Blender stdout: {result.stdout}")
                raise RuntimeError(f"FBX export failed with return code {result.returncode}")

            if not os.path.exists(output_fbx_path):
                raise RuntimeError(f"Export completed but output file not found: {output_fbx_path}")

            print(f"[UniRigExportPosedFBX] ✓ Successfully exported to: {output_fbx_path}")

            return (output_fbx_path,)

        finally:
            # Clean up temporary JSON file
            if os.path.exists(transforms_json_path):
                os.unlink(transforms_json_path)
