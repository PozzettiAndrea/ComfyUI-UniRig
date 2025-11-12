"""
Blender script for mesh extraction and preprocessing for UniRig.
This script is designed to be run via: blender --background --python blender_extract.py -- <args>
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
    print("Usage: blender --background --python blender_extract.py -- <input_file> <output_npz>")
    sys.exit(1)

input_file = argv[0]
output_npz = argv[1]
target_face_count = int(argv[2]) if len(argv) > 2 else 50000

print(f"[Blender Extract] Input: {input_file}")
print(f"[Blender Extract] Output: {output_npz}")
print(f"[Blender Extract] Target faces: {target_face_count}")

# Clear default scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import mesh based on file extension
ext = Path(input_file).suffix.lower()
print(f"[Blender Extract] Loading {ext} file...")

try:
    if ext == '.obj':
        bpy.ops.wm.obj_import(filepath=input_file)
    elif ext in ['.fbx', '.FBX']:
        bpy.ops.import_scene.fbx(filepath=input_file, ignore_leaf_bones=False, use_image_search=False)
    elif ext in ['.glb', '.gltf']:
        bpy.ops.import_scene.gltf(filepath=input_file, import_pack_images=False)
    elif ext == '.dae':
        bpy.ops.wm.collada_import(filepath=input_file)
    elif ext == '.stl':
        bpy.ops.wm.stl_import(filepath=input_file)
    else:
        print(f"[Blender Extract] Unsupported format: {ext}")
        sys.exit(1)

    print(f"[Blender Extract] Import successful")

except Exception as e:
    print(f"[Blender Extract] Import failed: {e}")
    sys.exit(1)

# Get all meshes
meshes = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']

if not meshes:
    print("[Blender Extract] Error: No meshes found in file")
    sys.exit(1)

print(f"[Blender Extract] Found {len(meshes)} mesh(es)")

# Combine all meshes
if len(meshes) > 1:
    # Select all meshes
    bpy.ops.object.select_all(action='DESELECT')
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]

    # Join meshes
    bpy.ops.object.join()
    mesh_obj = bpy.context.active_object
else:
    mesh_obj = meshes[0]

print(f"[Blender Extract] Processing mesh: {mesh_obj.name}")

# Apply all transforms
bpy.ops.object.select_all(action='DESELECT')
mesh_obj.select_set(True)
bpy.context.view_layer.objects.active = mesh_obj
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

# Get mesh data
mesh = mesh_obj.data

# Triangulate
print("[Blender Extract] Triangulating...")
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.quads_convert_to_tris()
bpy.ops.object.mode_set(mode='OBJECT')

# Simplify if needed
current_faces = len(mesh.polygons)
print(f"[Blender Extract] Current face count: {current_faces}")

if current_faces > target_face_count:
    print(f"[Blender Extract] Decimating to {target_face_count} faces...")

    # Add decimate modifier
    decimate_mod = mesh_obj.modifiers.new(name='Decimate', type='DECIMATE')
    decimate_mod.ratio = target_face_count / current_faces
    decimate_mod.use_collapse_triangulate = True

    # Apply modifier
    bpy.ops.object.modifier_apply(modifier=decimate_mod.name)

    print(f"[Blender Extract] Decimated to {len(mesh.polygons)} faces")

# Extract vertex and face data
vertices = np.zeros((len(mesh.vertices), 3), dtype=np.float32)
for i, v in enumerate(mesh.vertices):
    vertices[i] = v.co

faces = np.zeros((len(mesh.polygons), 3), dtype=np.int32)
for i, p in enumerate(mesh.polygons):
    if len(p.vertices) != 3:
        print(f"[Blender Extract] Warning: Non-triangular face found")
        continue
    faces[i] = [p.vertices[0], p.vertices[1], p.vertices[2]]

print(f"[Blender Extract] Extracted {len(vertices)} vertices, {len(faces)} faces")

# Calculate vertex normals (Blender 4.2+ compatible)
# Force recalculation by updating the mesh
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.object.mode_set(mode='OBJECT')

vertex_normals = np.zeros((len(vertices), 3), dtype=np.float32)
for i, v in enumerate(mesh.vertices):
    vertex_normals[i] = v.normal

print("[Blender Extract] Calculated vertex normals")

# Calculate face normals
face_normals = np.zeros((len(faces), 3), dtype=np.float32)
for i, p in enumerate(mesh.polygons):
    face_normals[i] = p.normal

print("[Blender Extract] Calculated face normals")

# Save as NPZ (raw_data format expected by UniRig)
# For skeleton extraction, skeleton fields are set to None
os.makedirs(os.path.dirname(output_npz), exist_ok=True)

np.savez_compressed(
    output_npz,
    vertices=vertices.astype(np.float32),
    vertex_normals=vertex_normals.astype(np.float32),
    faces=faces.astype(np.int32),
    face_normals=face_normals.astype(np.float32),
    joints=None,
    skin=None,
    parents=None,
    names=None,
    matrix_local=None,
)

print(f"[Blender Extract] Saved to: {output_npz}")
print("[Blender Extract] Done!")
