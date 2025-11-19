"""
Data conversion nodes for UniRig - Convert between different 3D data formats.
"""

import os
import subprocess
import tempfile
import numpy as np
import time
import shutil

# Support both relative imports (ComfyUI) and absolute imports (testing)
try:
    from ..constants import PARSE_TIMEOUT
except ImportError:
    from constants import PARSE_TIMEOUT

try:
    from .base import (
        BLENDER_EXE,
        BLENDER_PARSE_SKELETON,
    )
except ImportError:
    from base import (
        BLENDER_EXE,
        BLENDER_PARSE_SKELETON,
    )


class UniRigConvertFBXToData:
    """
    Convert FBX file (skeleton.fbx) to GLB + NPZ format for ML model.

    This node extracts:
    - Mesh geometry → mesh.glb
    - Skeleton data → skeleton.npz

    Based on the original UniRig extraction pipeline.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "skeleton_fbx_path": ("STRING", {
                    "tooltip": "Path to skeleton.fbx file from Extract Skeleton node"
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("mesh_glb_path", "skeleton_npz_path")
    FUNCTION = "convert"
    CATEGORY = "UniRig/Data"

    def convert(self, skeleton_fbx_path):
        """
        Convert FBX to GLB+NPZ format.

        Args:
            skeleton_fbx_path: Path to skeleton.fbx

        Returns:
            (mesh_glb_path, skeleton_npz_path)
        """
        print(f"[UniRigConvertFBXToData] Converting FBX to GLB+NPZ...")
        print(f"[UniRigConvertFBXToData] Input FBX: {skeleton_fbx_path}")

        if not skeleton_fbx_path or not os.path.exists(skeleton_fbx_path):
            raise RuntimeError(f"Skeleton FBX not found: {skeleton_fbx_path}")

        # Create temporary directory for output
        temp_dir = tempfile.mkdtemp(prefix="unirig_fbx_convert_")

        try:
            # Step 1: Parse skeleton FBX with Blender to extract all data
            print(f"[UniRigConvertFBXToData] Parsing FBX with Blender...")
            skeleton_data_npz = os.path.join(temp_dir, "skeleton_data.npz")

            parse_cmd = [
                BLENDER_EXE,
                "--background",
                "--python", BLENDER_PARSE_SKELETON,
                "--",
                skeleton_fbx_path,
                skeleton_data_npz,
            ]

            try:
                result = subprocess.run(
                    parse_cmd,
                    capture_output=True,
                    text=True,
                    timeout=PARSE_TIMEOUT
                )
                if result.returncode != 0:
                    print(f"[UniRigConvertFBXToData] Blender stderr:\n{result.stderr}")
                    raise RuntimeError(f"Blender parse failed with exit code {result.returncode}")
            except subprocess.TimeoutExpired:
                raise RuntimeError(f"Blender parse timed out (>{PARSE_TIMEOUT}s)")

            # Step 2: Load parsed data
            skeleton_data = np.load(skeleton_data_npz, allow_pickle=True)
            print(f"[UniRigConvertFBXToData] Loaded skeleton data from FBX")

            # Extract mesh data
            mesh_vertices_fbx = skeleton_data['mesh_vertices_fbx']
            mesh_faces_fbx = skeleton_data['mesh_faces_fbx']
            mesh_vertex_normals_fbx = skeleton_data.get('mesh_vertex_normals_fbx')
            mesh_face_normals_fbx = skeleton_data.get('mesh_face_normals_fbx')

            # Log mesh data
            if len(mesh_vertices_fbx) > 0:
                mesh_min = mesh_vertices_fbx.min(axis=0)
                mesh_max = mesh_vertices_fbx.max(axis=0)
                mesh_center = (mesh_min + mesh_max) / 2
                print(f"[UniRigConvertFBXToData] Loaded mesh vertices:")
                print(f"  First vertex: ({mesh_vertices_fbx[0][0]:.2f}, {mesh_vertices_fbx[0][1]:.2f}, {mesh_vertices_fbx[0][2]:.2f})")
                print(f"  Bounds: min({mesh_min[0]:.2f}, {mesh_min[1]:.2f}, {mesh_min[2]:.2f}) max({mesh_max[0]:.2f}, {mesh_max[1]:.2f}, {mesh_max[2]:.2f})")
                print(f"  Center: ({mesh_center[0]:.2f}, {mesh_center[1]:.2f}, {mesh_center[2]:.2f})")

            # Extract skeleton data
            bone_names = skeleton_data['bone_names']
            bone_parents = skeleton_data['bone_parents']
            bone_to_head_vertex = skeleton_data.get('bone_to_head_vertex')
            all_joints = skeleton_data['vertices']  # All joint positions

            # Map bones to joints
            num_bones = len(bone_names)
            if bone_to_head_vertex is not None:
                bone_joints = np.array([all_joints[bone_to_head_vertex[i]] for i in range(num_bones)])
            else:
                bone_joints = all_joints[:num_bones]

            print(f"[UniRigConvertFBXToData] Extracted: {len(bone_joints)} bones, {len(mesh_vertices_fbx)} vertices, {len(mesh_faces_fbx)} faces")

            # Log bone joint data
            if len(bone_joints) > 0:
                bone_min = bone_joints.min(axis=0)
                bone_max = bone_joints.max(axis=0)
                bone_center = (bone_min + bone_max) / 2
                print(f"[UniRigConvertFBXToData] Bone joints from NPZ:")
                print(f"  First joint: ({bone_joints[0][0]:.2f}, {bone_joints[0][1]:.2f}, {bone_joints[0][2]:.2f})")
                print(f"  Bounds: min({bone_min[0]:.2f}, {bone_min[1]:.2f}, {bone_min[2]:.2f}) max({bone_max[0]:.2f}, {bone_max[1]:.2f}, {bone_max[2]:.2f})")
                print(f"  Center: ({bone_center[0]:.2f}, {bone_center[1]:.2f}, {bone_center[2]:.2f})")

            # Step 3: Compute bone tails (simplified - use children positions or extend from parent)
            parents_list = [None if p == -1 else int(p) for p in bone_parents]
            tails = np.zeros((num_bones, 3))
            for i in range(num_bones):
                children = [j for j, p in enumerate(parents_list) if p == i]
                if children:
                    # Use average of children positions
                    tails[i] = np.mean([bone_joints[c] for c in children], axis=0)
                else:
                    # Leaf bone: extend from parent
                    if parents_list[i] is not None:
                        direction = bone_joints[i] - bone_joints[parents_list[i]]
                        tails[i] = bone_joints[i] + direction * 0.3
                    else:
                        # Root bone with no children: extend upward
                        tails[i] = bone_joints[i] + np.array([0, 0.1, 0])

            # Log computed tails
            if len(tails) > 0:
                tail_min = tails.min(axis=0)
                tail_max = tails.max(axis=0)
                tail_center = (tail_min + tail_max) / 2
                print(f"[UniRigConvertFBXToData] Computed bone tails:")
                print(f"  First tail: ({tails[0][0]:.2f}, {tails[0][1]:.2f}, {tails[0][2]:.2f})")
                print(f"  Bounds: min({tail_min[0]:.2f}, {tail_min[1]:.2f}, {tail_min[2]:.2f}) max({tail_max[0]:.2f}, {tail_max[1]:.2f}, {tail_max[2]:.2f})")
                print(f"  Center: ({tail_center[0]:.2f}, {tail_center[1]:.2f}, {tail_center[2]:.2f})")

            # Step 4: Export mesh to GLB using trimesh
            from trimesh import Trimesh

            mesh_glb_path = os.path.join(temp_dir, "mesh.glb")

            fbx_mesh = Trimesh(
                vertices=mesh_vertices_fbx,
                faces=mesh_faces_fbx,
                vertex_normals=mesh_vertex_normals_fbx,
                process=False
            )
            fbx_mesh.export(mesh_glb_path)
            print(f"[UniRigConvertFBXToData] Exported mesh to GLB: {mesh_glb_path}")

            # Step 5: Create skeleton.npz with proper structure
            skeleton_npz_path = os.path.join(temp_dir, "skeleton.npz")

            npz_data = {
                # Skeleton data
                'joints': bone_joints,
                'names': np.array([str(n) for n in bone_names], dtype=object),
                'parents': np.array(parents_list, dtype=object),
                'tails': tails,
                'no_skin': None,  # Not used in skinning inference
                'matrix_local': None,  # Not needed for inference

                # Mesh data (needed for skinning inference)
                'vertices': mesh_vertices_fbx,
                'faces': mesh_faces_fbx,
                'vertex_normals': mesh_vertex_normals_fbx,
                'face_normals': mesh_face_normals_fbx,

                # Skinning data (empty for skeleton-only FBX)
                'skin': None,

                # Metadata
                'path': None,
                'cls': None,

                # Texture data (extract if available)
                'uv_coords': skeleton_data.get('uv_coords', np.array([], dtype=np.float32)),
                'uv_faces': skeleton_data.get('uv_faces', np.array([], dtype=np.int32)),
                'texture_data_base64': skeleton_data.get('texture_data_base64', ""),
                'texture_format': skeleton_data.get('texture_format', ""),
                'texture_width': skeleton_data.get('texture_width', 0),
                'texture_height': skeleton_data.get('texture_height', 0),
                'material_name': skeleton_data.get('material_name', ""),
            }

            # Log coordinate summary before saving
            print(f"[UniRigConvertFBXToData] === COORDINATE SUMMARY ===")
            if len(mesh_vertices_fbx) > 0 and len(bone_joints) > 0:
                mesh_min_final = mesh_vertices_fbx.min(axis=0)
                mesh_max_final = mesh_vertices_fbx.max(axis=0)
                mesh_center_final = (mesh_min_final + mesh_max_final) / 2

                bone_min_final = bone_joints.min(axis=0)
                bone_max_final = bone_joints.max(axis=0)
                bone_center_final = (bone_min_final + bone_max_final) / 2

                center_dist = np.linalg.norm(mesh_center_final - bone_center_final)
                mesh_size_scalar = np.linalg.norm(mesh_max_final - mesh_min_final)
                bone_size_scalar = np.linalg.norm(bone_max_final - bone_min_final)
                size_ratio = mesh_size_scalar / bone_size_scalar

                print(f"  Mesh center: ({mesh_center_final[0]:.2f}, {mesh_center_final[1]:.2f}, {mesh_center_final[2]:.2f})")
                print(f"  Skeleton center: ({bone_center_final[0]:.2f}, {bone_center_final[1]:.2f}, {bone_center_final[2]:.2f})")
                print(f"  Center distance: {center_dist:.4f} units")
                print(f"  Size ratio (mesh/skeleton): {size_ratio:.3f}")
                if center_dist > 1.0:
                    print(f"  WARNING: Large center distance detected!")
                else:
                    print(f"  ✓ Mesh and skeleton centers are well aligned")

            np.savez(skeleton_npz_path, **npz_data)
            print(f"[UniRigConvertFBXToData] Created skeleton NPZ: {skeleton_npz_path}")

            # Copy files to output directory for persistence
            import folder_paths
            output_dir = folder_paths.get_output_directory()
            timestamp = int(time.time())

            # Create output filenames with timestamp
            output_mesh_filename = f"converted_mesh_{timestamp}.glb"
            output_skeleton_filename = f"converted_skeleton_{timestamp}.npz"

            output_mesh_path = os.path.join(output_dir, output_mesh_filename)
            output_skeleton_path = os.path.join(output_dir, output_skeleton_filename)

            # Copy files to output directory
            shutil.copy2(mesh_glb_path, output_mesh_path)
            shutil.copy2(skeleton_npz_path, output_skeleton_path)

            print(f"[UniRigConvertFBXToData] Conversion complete!")
            print(f"[UniRigConvertFBXToData] → mesh.glb: {output_mesh_path}")
            print(f"[UniRigConvertFBXToData] → skeleton.npz: {output_skeleton_path}")
            print(f"[UniRigConvertFBXToData] Files saved to output directory")

            # Return output paths (not temp paths)
            return (output_mesh_path, output_skeleton_path)

        except Exception as e:
            # Clean up on error
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
            raise e


class UniRigPreviewGLBAndNPZ:
    """
    Preview GLB mesh and NPZ skeleton together in 3D viewer.

    Useful for inspecting the converted data before running ML skinning.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_glb_path": ("STRING", {
                    "tooltip": "Path to mesh.glb file"
                }),
                "skeleton_npz_path": ("STRING", {
                    "tooltip": "Path to skeleton.npz file"
                }),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "preview"
    CATEGORY = "UniRig/Data"

    def preview(self, mesh_glb_path, skeleton_npz_path):
        """
        Preview GLB mesh and NPZ skeleton in interactive 3D viewer.

        Args:
            mesh_glb_path: Path to mesh.glb
            skeleton_npz_path: Path to skeleton.npz

        Returns:
            UI output with GLB filename and skeleton data for frontend widget
        """
        print(f"[UniRigPreviewGLBAndNPZ] Preparing preview...")
        print(f"[UniRigPreviewGLBAndNPZ] GLB: {mesh_glb_path}")
        print(f"[UniRigPreviewGLBAndNPZ] NPZ: {skeleton_npz_path}")

        if not mesh_glb_path or not os.path.exists(mesh_glb_path):
            raise RuntimeError(f"Mesh GLB not found: {mesh_glb_path}")

        if not skeleton_npz_path or not os.path.exists(skeleton_npz_path):
            raise RuntimeError(f"Skeleton NPZ not found: {skeleton_npz_path}")

        # Load skeleton data
        skeleton_data = np.load(skeleton_npz_path, allow_pickle=True)

        # Extract relevant fields
        joints = skeleton_data['joints'].tolist()
        parents = skeleton_data['parents'].tolist()
        names = skeleton_data['names'].tolist() if 'names' in skeleton_data else None

        print(f"[UniRigPreviewGLBAndNPZ] Skeleton: {len(joints)} joints")

        # Copy GLB to output directory for web access
        import folder_paths
        output_dir = folder_paths.get_output_directory()
        filename = f"preview_glb_{int(time.time())}.glb"
        output_glb_path = os.path.join(output_dir, filename)

        shutil.copy2(mesh_glb_path, output_glb_path)
        print(f"[UniRigPreviewGLBAndNPZ] Copied GLB to: {output_glb_path}")

        # Extract texture data if available
        texture_data_base64 = None
        texture_format = None
        texture_width = None
        texture_height = None
        uv_coords = None
        uv_faces = None

        if 'texture_data_base64' in skeleton_data:
            tex_data = skeleton_data['texture_data_base64']
            if tex_data is not None and len(str(tex_data)) > 0:
                texture_data_base64 = str(tex_data)
                texture_format = str(skeleton_data.get('texture_format', '')) if 'texture_format' in skeleton_data else None
                texture_width = int(skeleton_data.get('texture_width', 0)) if 'texture_width' in skeleton_data else None
                texture_height = int(skeleton_data.get('texture_height', 0)) if 'texture_height' in skeleton_data else None
                print(f"[UniRigPreviewGLBAndNPZ] Texture data found: {texture_format} {texture_width}x{texture_height}")

        if 'uv_coords' in skeleton_data and len(skeleton_data['uv_coords']) > 0:
            uv_coords = skeleton_data['uv_coords'].tolist()
            uv_faces = skeleton_data['uv_faces'].tolist() if 'uv_faces' in skeleton_data else None
            print(f"[UniRigPreviewGLBAndNPZ] UV coordinates found: {len(uv_coords)} UVs")

        # Prepare skeleton data dict for frontend
        skeleton_data_dict = {
            'joints': joints,
            'parents': parents,
            'names': names,
            'texture_data_base64': texture_data_base64,
            'texture_format': texture_format,
            'texture_width': texture_width,
            'texture_height': texture_height,
            'uv_coords': uv_coords,
            'uv_faces': uv_faces,
        }

        print(f"[UniRigPreviewGLBAndNPZ] Returning data for frontend widget")

        # Return data for frontend JavaScript extension
        # Frontend will create iframe and send this data via postMessage
        return {
            "ui": {
                "glb_file": [filename],
                "skeleton_data": [skeleton_data_dict]
            }
        }


# Node export list
NODE_CLASS_MAPPINGS = {
    "UniRigConvertFBXToData": UniRigConvertFBXToData,
    "UniRigPreviewGLBAndNPZ": UniRigPreviewGLBAndNPZ,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UniRigConvertFBXToData": "Convert FBX to Data",
    "UniRigPreviewGLBAndNPZ": "Preview GLB + NPZ",
}
