"""
Blender script to extract mesh statistics from FBX file.
This script extracts mesh info including vertex count, face count, bounding box, etc.
Usage: blender --background --python blender_extract_mesh_info.py -- <input_fbx> <output_npz>
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
    print("Usage: blender --background --python blender_extract_mesh_info.py -- <input_fbx> <output_npz>")
    sys.exit(1)

input_fbx = argv[0]
output_npz = argv[1]

print(f"[Blender Mesh Info] Input: {input_fbx}")
print(f"[Blender Mesh Info] Output: {output_npz}")

# Clear default scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import FBX
print(f"[Blender Mesh Info] Loading FBX...")
try:
    bpy.ops.import_scene.fbx(filepath=input_fbx)
    print(f"[Blender Mesh Info] Import successful")
except Exception as e:
    print(f"[Blender Mesh Info] Import failed: {e}")
    sys.exit(1)

# Find all meshes
meshes = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']

if not meshes:
    print("[Blender Mesh Info] No meshes found in FBX")
    sys.exit(1)

print(f"[Blender Mesh Info] Found {len(meshes)} mesh object(s)")

# Collect statistics
total_vertices = 0
total_faces = 0
mesh_names = []
all_vertices = []

for mesh_obj in meshes:
    mesh = mesh_obj.data
    mesh_names.append(mesh_obj.name)

    vertex_count = len(mesh.vertices)
    face_count = len(mesh.polygons)

    total_vertices += vertex_count
    total_faces += face_count

    print(f"[Blender Mesh Info]   Mesh '{mesh_obj.name}': {vertex_count} verts, {face_count} faces")

    # Collect world-space vertex positions for bounding box
    for v in mesh.vertices:
        world_pos = mesh_obj.matrix_world @ v.co
        all_vertices.append([world_pos.x, world_pos.y, world_pos.z])

# Calculate global bounding box
all_vertices = np.array(all_vertices, dtype=np.float32)
bbox_min = all_vertices.min(axis=0)
bbox_max = all_vertices.max(axis=0)
extents = bbox_max - bbox_min
center = (bbox_min + bbox_max) / 2

print(f"[Blender Mesh Info] Total: {total_vertices} vertices, {total_faces} faces")
print(f"[Blender Mesh Info] Bounding Box:")
print(f"[Blender Mesh Info]   Min: [{bbox_min[0]:.4f}, {bbox_min[1]:.4f}, {bbox_min[2]:.4f}]")
print(f"[Blender Mesh Info]   Max: [{bbox_max[0]:.4f}, {bbox_max[1]:.4f}, {bbox_max[2]:.4f}]")
print(f"[Blender Mesh Info]   Extents: [{extents[0]:.4f}, {extents[1]:.4f}, {extents[2]:.4f}]")
print(f"[Blender Mesh Info]   Center: [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}]")

# Save mesh statistics
os.makedirs(os.path.dirname(output_npz) if os.path.dirname(output_npz) else '.', exist_ok=True)

save_data = {
    'total_vertices': np.array(total_vertices, dtype=np.int32),
    'total_faces': np.array(total_faces, dtype=np.int32),
    'mesh_count': np.array(len(meshes), dtype=np.int32),
    'mesh_names': np.array(mesh_names, dtype=object),
    'bbox_min': bbox_min.astype(np.float32),
    'bbox_max': bbox_max.astype(np.float32),
    'extents': extents.astype(np.float32),
    'center': center.astype(np.float32),
}

np.savez_compressed(output_npz, **save_data)

print(f"[Blender Mesh Info] Saved to: {output_npz}")
print("[Blender Mesh Info] Done!")
