"""
Blender script to convert FBX to GLB format for mesh loading.
This script is designed to be run via: blender --background --python blender_load_fbx.py -- <input_fbx> <output_glb>

FBX files cannot be loaded directly by trimesh/igl, so we use Blender as an intermediary
to convert FBX to GLB which trimesh can handle.
"""

import bpy
import sys
import os
from pathlib import Path

# Get arguments after '--'
argv = sys.argv
argv = argv[argv.index("--") + 1:] if "--" in argv else []

if len(argv) < 2:
    print("Usage: blender --background --python blender_load_fbx.py -- <input_fbx> <output_glb>")
    sys.exit(1)

input_file = argv[0]
output_file = argv[1]

print(f"[Blender FBX Load] Input: {input_file}")
print(f"[Blender FBX Load] Output: {output_file}")

# Clear default scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import FBX
ext = Path(input_file).suffix.lower()
print(f"[Blender FBX Load] Loading {ext} file...")

try:
    if ext in ['.fbx', '.FBX']:
        bpy.ops.import_scene.fbx(
            filepath=input_file,
            ignore_leaf_bones=False,
            use_image_search=False,
            automatic_bone_orientation=False
        )
    else:
        print(f"[Blender FBX Load] Expected FBX file, got: {ext}")
        sys.exit(1)

    print(f"[Blender FBX Load] Import successful")

except Exception as e:
    print(f"[Blender FBX Load] Import failed: {e}")
    sys.exit(1)

# Get all meshes
meshes = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']

if not meshes:
    print("[Blender FBX Load] Error: No meshes found in FBX file")
    sys.exit(1)

print(f"[Blender FBX Load] Found {len(meshes)} mesh(es)")

# Select all meshes for export
bpy.ops.object.select_all(action='DESELECT')
for obj in meshes:
    obj.select_set(True)

# If there are multiple meshes, combine them
if len(meshes) > 1:
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.join()
    print(f"[Blender FBX Load] Combined meshes into one")

mesh_obj = bpy.context.active_object if bpy.context.active_object else meshes[0]

# Apply any transforms
bpy.context.view_layer.objects.active = mesh_obj
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

# Count vertices and faces
mesh_data = mesh_obj.data
vertex_count = len(mesh_data.vertices)
face_count = len(mesh_data.polygons)
print(f"[Blender FBX Load] Mesh: {vertex_count} vertices, {face_count} faces")

# Export as GLB
try:
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Select the mesh for export
    bpy.ops.object.select_all(action='DESELECT')
    mesh_obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_obj

    # Export to GLB format
    bpy.ops.export_scene.gltf(
        filepath=output_file,
        export_format='GLB',
        use_selection=True,
        export_apply=True,
        export_texcoords=True,
        export_normals=True,
        export_colors=True,
        export_materials='NONE',  # Skip materials for mesh-only export
    )

    print(f"[Blender FBX Load] Exported to: {output_file}")
    print(f"[Blender FBX Load] Done!")

except Exception as e:
    print(f"[Blender FBX Load] Export failed: {e}")
    sys.exit(1)
