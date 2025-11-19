"""
Blender script to parse skeleton from FBX file.
This script extracts bone positions and connections from an armature.
Usage: blender --background --python blender_parse_skeleton.py -- <input_fbx> <output_npz>
"""

import bpy
import sys
import os
import numpy as np
from pathlib import Path

# Get arguments after '--'
argv = sys.argv
argv = argv[argv.index("--") + 1:] if "--" in argv else []

if len(argv) < 2:
    print("Usage: blender --background --python blender_parse_skeleton.py -- <input_fbx> <output_npz>")
    sys.exit(1)

input_fbx = argv[0]
output_npz = argv[1]

print(f"[Blender Skeleton Parse] Input: {input_fbx}")
print(f"[Blender Skeleton Parse] Output: {output_npz}")

# Clear default scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import FBX
print(f"[Blender Skeleton Parse] Loading FBX...")
try:
    bpy.ops.import_scene.fbx(filepath=input_fbx)
    print(f"[Blender Skeleton Parse] Import successful")
except Exception as e:
    print(f"[Blender Skeleton Parse] Import failed: {e}")
    sys.exit(1)

# Find armature
armatures = [obj for obj in bpy.context.scene.objects if obj.type == 'ARMATURE']

if not armatures:
    print("[Blender Skeleton Parse] No armature found in FBX")
    # Fall back to mesh edges if no armature
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    if meshes:
        print(f"[Blender Skeleton Parse] Found {len(meshes)} meshes, extracting geometry...")

        # Combine all meshes
        if len(meshes) > 1:
            bpy.ops.object.select_all(action='DESELECT')
            for obj in meshes:
                obj.select_set(True)
            bpy.context.view_layer.objects.active = meshes[0]
            bpy.ops.object.join()
            mesh_obj = bpy.context.active_object
        else:
            mesh_obj = meshes[0]

        mesh = mesh_obj.data

        # Extract vertices and edges
        vertices = np.zeros((len(mesh.vertices), 3), dtype=np.float32)
        for i, v in enumerate(mesh.vertices):
            vertices[i] = v.co

        edges = np.zeros((len(mesh.edges), 2), dtype=np.int32)
        for i, e in enumerate(mesh.edges):
            edges[i] = [e.vertices[0], e.vertices[1]]

        print(f"[Blender Skeleton Parse] Extracted {len(vertices)} vertices, {len(edges)} edges from mesh")
    else:
        print("[Blender Skeleton Parse] Error: No armature or mesh found")
        sys.exit(1)
else:
    armature = armatures[0]
    print(f"[Blender Skeleton Parse] Found armature: {armature.name}")

    # Get bones in pose position
    bones = armature.pose.bones
    print(f"[Blender Skeleton Parse] Found {len(bones)} bones")

    # Extract bone head/tail positions (world space)
    bone_heads = []
    bone_tails = []
    bone_names = []
    bone_parents = {}

    for i, bone in enumerate(bones):
        # Get world space positions
        head = armature.matrix_world @ bone.head
        tail = armature.matrix_world @ bone.tail

        bone_heads.append([head.x, head.y, head.z])
        bone_tails.append([tail.x, tail.y, tail.z])
        bone_names.append(bone.name)

        # Track parent relationship
        if bone.parent:
            bone_parents[i] = bone_names.index(bone.parent.name)

    # Create vertex list (unique positions)
    all_positions = bone_heads + bone_tails
    unique_positions = []
    position_map = {}  # maps (head/tail, bone_idx) to unique vertex index

    for i, pos in enumerate(all_positions):
        # Check if position already exists (within tolerance)
        found = False
        for j, unique_pos in enumerate(unique_positions):
            if np.allclose(pos, unique_pos, atol=1e-6):
                if i < len(bone_heads):
                    position_map[('head', i)] = j
                else:
                    position_map[('tail', i - len(bone_heads))] = j
                found = True
                break

        if not found:
            position_map[('head' if i < len(bone_heads) else 'tail',
                         i if i < len(bone_heads) else i - len(bone_heads))] = len(unique_positions)
            unique_positions.append(pos)

    vertices = np.array(unique_positions, dtype=np.float32)

    # Create edges (bones connect head to tail)
    edges = []
    bone_to_head_vertex = []  # Map bone index to head vertex index

    for i in range(len(bone_heads)):
        head_idx = position_map[('head', i)]
        tail_idx = position_map[('tail', i)]
        edges.append([head_idx, tail_idx])
        bone_to_head_vertex.append(head_idx)

    edges = np.array(edges, dtype=np.int32)

    # Create parent array: for each bone, store parent bone index (or -1 if root)
    parents = np.full(len(bones), -1, dtype=np.int32)
    for bone_idx, parent_idx in bone_parents.items():
        parents[bone_idx] = parent_idx

    print(f"[Blender Skeleton Parse] Extracted {len(vertices)} unique joint positions, {len(edges)} bones")
    print(f"[Blender Skeleton Parse] Hierarchy: {len(bones)} bones with {len([p for p in parents if p >= 0])} parent relationships")

    # Log bone coordinate information
    if len(vertices) > 0:
        bone_min = vertices.min(axis=0)
        bone_max = vertices.max(axis=0)
        bone_center = (bone_min + bone_max) / 2
        bone_size = bone_max - bone_min
        print(f"[Blender Skeleton Parse] Bone joints (WORLD SPACE):")
        print(f"  First joint: ({vertices[0][0]:.2f}, {vertices[0][1]:.2f}, {vertices[0][2]:.2f})")
        print(f"  Bounds: min({bone_min[0]:.2f}, {bone_min[1]:.2f}, {bone_min[2]:.2f}) max({bone_max[0]:.2f}, {bone_max[1]:.2f}, {bone_max[2]:.2f})")
        print(f"  Center: ({bone_center[0]:.2f}, {bone_center[1]:.2f}, {bone_center[2]:.2f})")
        print(f"  Size: ({bone_size[0]:.2f}, {bone_size[1]:.2f}, {bone_size[2]:.2f})")

# Extract mesh data from FBX (if present)
mesh_vertices = None
mesh_faces = None
meshes = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
if meshes:
    print(f"[Blender Skeleton Parse] Found {len(meshes)} mesh(es) in FBX, extracting geometry...")

    # Use the first mesh (or join multiple meshes)
    if len(meshes) > 1:
        bpy.ops.object.select_all(action='DESELECT')
        for obj in meshes:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = meshes[0]
        bpy.ops.object.join()
        mesh_obj = bpy.context.active_object
    else:
        mesh_obj = meshes[0]

    mesh = mesh_obj.data

    # Extract mesh vertices in WORLD SPACE (to match bone positions)
    mesh_vertices = np.zeros((len(mesh.vertices), 3), dtype=np.float32)
    for i, v in enumerate(mesh.vertices):
        # Transform vertex from local to world space
        world_co = mesh_obj.matrix_world @ v.co
        mesh_vertices[i] = [world_co.x, world_co.y, world_co.z]

    # Extract mesh vertex normals in WORLD SPACE
    mesh_vertex_normals = np.zeros((len(mesh.vertices), 3), dtype=np.float32)
    # Extract rotation component from world matrix
    world_rotation = mesh_obj.matrix_world.to_3x3()
    for i, v in enumerate(mesh.vertices):
        # Rotate normal to world space
        world_normal = (world_rotation @ v.normal).normalized()
        mesh_vertex_normals[i] = [world_normal.x, world_normal.y, world_normal.z]

    # Extract mesh face normals in WORLD SPACE
    mesh_face_normals = np.zeros((len(mesh.polygons), 3), dtype=np.float32)
    for i, poly in enumerate(mesh.polygons):
        # Rotate face normal to world space
        world_normal = (world_rotation @ poly.normal).normalized()
        mesh_face_normals[i] = [world_normal.x, world_normal.y, world_normal.z]

    # Extract mesh faces (triangulated)
    mesh_faces = []
    for poly in mesh.polygons:
        if len(poly.vertices) == 3:
            mesh_faces.append([poly.vertices[0], poly.vertices[1], poly.vertices[2]])
        elif len(poly.vertices) == 4:
            # Triangulate quads
            mesh_faces.append([poly.vertices[0], poly.vertices[1], poly.vertices[2]])
            mesh_faces.append([poly.vertices[0], poly.vertices[2], poly.vertices[3]])

    if mesh_faces:
        mesh_faces = np.array(mesh_faces, dtype=np.int32)
    else:
        mesh_faces = None

    print(f"[Blender Skeleton Parse] Extracted mesh: {len(mesh_vertices)} vertices, {len(mesh_faces) if mesh_faces is not None else 0} faces")

    # Log mesh coordinate information
    if len(mesh_vertices) > 0:
        mesh_min = mesh_vertices.min(axis=0)
        mesh_max = mesh_vertices.max(axis=0)
        mesh_center = (mesh_min + mesh_max) / 2
        mesh_size = mesh_max - mesh_min
        print(f"[Blender Skeleton Parse] Mesh vertices (WORLD SPACE):")
        print(f"  First vertex: ({mesh_vertices[0][0]:.2f}, {mesh_vertices[0][1]:.2f}, {mesh_vertices[0][2]:.2f})")
        print(f"  Bounds: min({mesh_min[0]:.2f}, {mesh_min[1]:.2f}, {mesh_min[2]:.2f}) max({mesh_max[0]:.2f}, {mesh_max[1]:.2f}, {mesh_max[2]:.2f})")
        print(f"  Center: ({mesh_center[0]:.2f}, {mesh_center[1]:.2f}, {mesh_center[2]:.2f})")
        print(f"  Size: ({mesh_size[0]:.2f}, {mesh_size[1]:.2f}, {mesh_size[2]:.2f})")

    # Extract UV coordinates if available
    uv_coords = None
    uv_faces = None
    if mesh.uv_layers.active:
        uv_layer = mesh.uv_layers.active.data
        # UV coordinates are stored per loop (face corner), not per vertex
        # We need to map them to face corners
        uv_coords_list = []
        uv_faces_list = []

        for i, poly in enumerate(mesh.polygons):
            face_uvs = []
            for loop_idx in poly.loop_indices:
                uv = uv_layer[loop_idx].uv
                uv_coords_list.append([uv[0], uv[1]])
                face_uvs.append(len(uv_coords_list) - 1)
            uv_faces_list.append(face_uvs)

        uv_coords = np.array(uv_coords_list, dtype=np.float32)
        uv_faces = np.array(uv_faces_list, dtype=np.int32)
        print(f"[Blender Skeleton Parse] Extracted UV coordinates: {len(uv_coords)} UVs for {len(uv_faces)} faces")
    else:
        print("[Blender Skeleton Parse] No UV layer found")

    # Extract material/texture info if available
    import base64
    import struct
    import zlib

    material_name = None
    texture_path = None
    texture_data_base64 = ""
    texture_format = ""
    texture_width = 0
    texture_height = 0

    def extract_texture_from_image(image, max_size=2048):
        """Extract texture data from Blender image as base64 encoded PNG string.
        Uses pure Python PNG encoding without PIL dependency."""

        try:
            # Get image dimensions
            width, height = image.size
            channels = image.channels

            print(f"[Blender Skeleton Parse] Extracting texture: {width}x{height}, {channels} channels")

            # Get pixel data - Blender stores as flat RGBA float array
            pixels = np.array(image.pixels[:])

            # Reshape to (height, width, channels)
            pixels = pixels.reshape((height, width, channels))

            # Convert from float [0,1] to uint8 [0,255]
            pixels = (np.clip(pixels, 0, 1) * 255).astype(np.uint8)

            # Flip vertically (Blender stores bottom-to-top)
            pixels = np.flipud(pixels)

            # Resize if too large (simple nearest-neighbor downsampling)
            if width > max_size or height > max_size:
                scale = max_size / max(width, height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                print(f"[Blender Skeleton Parse] Resizing texture from {width}x{height} to {new_width}x{new_height}")

                # Simple nearest-neighbor resize using numpy
                row_indices = (np.arange(new_height) * height / new_height).astype(int)
                col_indices = (np.arange(new_width) * width / new_width).astype(int)
                pixels = pixels[row_indices][:, col_indices]
                width, height = new_width, new_height

            # Encode as PNG without PIL - use pure Python PNG encoder
            def encode_png(pixels, width, height, channels):
                """
                Encode numpy pixel array to PNG format without PIL.
                Uses minimal PNG implementation with zlib compression.
                """
                def png_chunk(chunk_type, data):
                    """Create a PNG chunk with CRC."""
                    chunk_len = struct.pack('>I', len(data))
                    chunk_data = chunk_type + data
                    checksum = struct.pack('>I', zlib.crc32(chunk_data) & 0xffffffff)
                    return chunk_len + chunk_data + checksum

                # PNG signature
                signature = b'\x89PNG\r\n\x1a\n'

                # IHDR chunk (image header)
                if channels == 4:
                    color_type = 6  # RGBA
                elif channels == 3:
                    color_type = 2  # RGB
                else:
                    # Convert to RGB
                    if channels == 1:
                        pixels = np.repeat(pixels, 3, axis=2)
                        channels = 3
                        color_type = 2
                    else:
                        raise ValueError(f"Unsupported channel count: {channels}")

                bit_depth = 8
                compression = 0
                filter_method = 0
                interlace = 0

                ihdr_data = struct.pack('>IIBBBBB', width, height, bit_depth, color_type,
                                        compression, filter_method, interlace)
                ihdr_chunk = png_chunk(b'IHDR', ihdr_data)

                # IDAT chunk (compressed image data)
                # Prepare raw data with filter bytes
                raw_data = b''
                for row in pixels:
                    # Filter type 0 (None) for each row
                    raw_data += b'\x00'
                    raw_data += row.tobytes()

                # Compress with zlib
                compressed = zlib.compress(raw_data, level=6)
                idat_chunk = png_chunk(b'IDAT', compressed)

                # IEND chunk (end of image)
                iend_chunk = png_chunk(b'IEND', b'')

                # Combine all chunks
                png_data = signature + ihdr_chunk + idat_chunk + iend_chunk

                return png_data

            png_data = encode_png(pixels, width, height, channels)

            # Encode to base64
            encoded = base64.b64encode(png_data).decode('utf-8')

            print(f"[Blender Skeleton Parse] Texture encoded: {len(encoded) / 1024:.1f} KB base64")
            return encoded, 'PNG', width, height

        except Exception as e:
            print(f"[Blender Skeleton Parse] Error extracting texture: {e}")
            import traceback
            traceback.print_exc()
            return None, None, 0, 0

    if mesh_obj.material_slots:
        mat = mesh_obj.material_slots[0].material
        if mat:
            material_name = mat.name
            # Try to find base color texture
            if mat.use_nodes:
                for node in mat.node_tree.nodes:
                    if node.type == 'TEX_IMAGE' and node.image:
                        texture_path = node.image.filepath
                        print(f"[Blender Skeleton Parse] Found texture node: {node.name}")
                        print(f"[Blender Skeleton Parse] Texture path: {texture_path}")

                        # Extract actual image data
                        tex_base64, tex_fmt, tex_w, tex_h = extract_texture_from_image(node.image)
                        if tex_base64:
                            texture_data_base64 = tex_base64
                            texture_format = tex_fmt
                            texture_width = tex_w
                            texture_height = tex_h
                            print(f"[Blender Skeleton Parse] Texture extracted successfully: {tex_w}x{tex_h} {tex_fmt}")
                        break
            print(f"[Blender Skeleton Parse] Material: {material_name}")

# Save skeleton data
os.makedirs(os.path.dirname(output_npz) if os.path.dirname(output_npz) else '.', exist_ok=True)

# Prepare data to save
save_data = {
    'vertices': vertices.astype(np.float32),
    'edges': edges.astype(np.int32),
}

# Add hierarchy data if armature was found
if armatures:
    save_data['bone_names'] = np.array(bone_names, dtype=object)
    save_data['bone_parents'] = parents
    save_data['bone_to_head_vertex'] = np.array(bone_to_head_vertex, dtype=np.int32)

# Add mesh data if extracted
if mesh_vertices is not None:
    save_data['mesh_vertices_fbx'] = mesh_vertices
if mesh_faces is not None:
    save_data['mesh_faces_fbx'] = mesh_faces
if 'mesh_vertex_normals' in locals() and mesh_vertex_normals is not None:
    save_data['mesh_vertex_normals_fbx'] = mesh_vertex_normals
if 'mesh_face_normals' in locals() and mesh_face_normals is not None:
    save_data['mesh_face_normals_fbx'] = mesh_face_normals

# Add UV and texture data if extracted
if 'uv_coords' in locals() and uv_coords is not None:
    save_data['uv_coords'] = uv_coords
if 'uv_faces' in locals() and uv_faces is not None:
    save_data['uv_faces'] = uv_faces
if 'material_name' in locals() and material_name:
    save_data['material_name'] = material_name
if 'texture_data_base64' in locals() and texture_data_base64:
    save_data['texture_data_base64'] = texture_data_base64
if 'texture_format' in locals() and texture_format:
    save_data['texture_format'] = texture_format
if 'texture_width' in locals() and texture_width:
    save_data['texture_width'] = texture_width
if 'texture_height' in locals() and texture_height:
    save_data['texture_height'] = texture_height

# Log summary before saving
print(f"[Blender Skeleton Parse] === COORDINATE SUMMARY ===")
if 'mesh_vertices' in locals() and mesh_vertices is not None and 'vertices' in locals() and vertices is not None:
    mesh_min = mesh_vertices.min(axis=0)
    mesh_max = mesh_vertices.max(axis=0)
    mesh_center = (mesh_min + mesh_max) / 2

    bone_min = vertices.min(axis=0)
    bone_max = vertices.max(axis=0)
    bone_center = (bone_min + bone_max) / 2

    center_dist = np.linalg.norm(mesh_center - bone_center)
    mesh_size_scalar = np.linalg.norm(mesh_max - mesh_min)
    bone_size_scalar = np.linalg.norm(bone_max - bone_min)
    size_ratio = mesh_size_scalar / bone_size_scalar if bone_size_scalar > 0 else 0

    print(f"  Mesh center: ({mesh_center[0]:.2f}, {mesh_center[1]:.2f}, {mesh_center[2]:.2f})")
    print(f"  Skeleton center: ({bone_center[0]:.2f}, {bone_center[1]:.2f}, {bone_center[2]:.2f})")
    print(f"  Center distance: {center_dist:.4f} units")
    print(f"  Size ratio (mesh/skeleton): {size_ratio:.3f}")
    if center_dist > 1.0:
        print(f"  WARNING: Large center distance detected! Mesh and skeleton may be misaligned.")
    else:
        print(f"  ✓ Mesh and skeleton centers are well aligned")
print(f"[Blender Skeleton Parse] ============================")

np.savez_compressed(output_npz, **save_data)

print(f"[Blender Skeleton Parse] Saved to: {output_npz}")
print("[Blender Skeleton Parse] Done!")
