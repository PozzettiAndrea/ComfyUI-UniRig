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
    for i in range(len(bone_heads)):
        head_idx = position_map[('head', i)]
        tail_idx = position_map[('tail', i)]
        edges.append([head_idx, tail_idx])

    edges = np.array(edges, dtype=np.int32)

    print(f"[Blender Skeleton Parse] Extracted {len(vertices)} unique joint positions, {len(edges)} bones")

# Save skeleton data
os.makedirs(os.path.dirname(output_npz) if os.path.dirname(output_npz) else '.', exist_ok=True)

np.savez_compressed(
    output_npz,
    vertices=vertices.astype(np.float32),
    edges=edges.astype(np.int32),
)

print(f"[Blender Skeleton Parse] Saved to: {output_npz}")
print("[Blender Skeleton Parse] Done!")
