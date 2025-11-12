"""
Blender script to export skeleton data to FBX file.
This script takes skeleton data from pickle and creates a Blender armature, then exports to FBX.
Usage: blender --background --python blender_export_fbx.py -- <input_pkl> <output_fbx> [options]
"""

import bpy
import sys
import os
import pickle
import numpy as np
from mathutils import Vector
from collections import defaultdict

# Get arguments after '--'
argv = sys.argv
argv = argv[argv.index("--") + 1:] if "--" in argv else []

if len(argv) < 2:
    print("Usage: blender --background --python blender_export_fbx.py -- <input_pkl> <output_fbx> [options]")
    sys.exit(1)

input_pkl = argv[0]
output_fbx = argv[1]

# Parse optional parameters
extrude_size = 0.03
add_root = False
use_extrude_bone = True
use_connect_unique_child = True
extrude_from_parent = True

for arg in argv[2:]:
    if arg.startswith("--extrude_size="):
        extrude_size = float(arg.split("=")[1])
    elif arg == "--add_root":
        add_root = True
    elif arg == "--no_extrude_bone":
        use_extrude_bone = False
    elif arg == "--no_connect_unique_child":
        use_connect_unique_child = False
    elif arg == "--no_extrude_from_parent":
        extrude_from_parent = False

print(f"[Blender FBX Export] Input: {input_pkl}")
print(f"[Blender FBX Export] Output: {output_fbx}")

# Load skeleton data from pickle
try:
    with open(input_pkl, 'rb') as f:
        data = pickle.load(f)  # No numpy imports needed - only plain Python types!

    # Now convert plain Python lists to numpy arrays using Blender's own numpy
    joints = np.array(data['joints'], dtype=np.float32)
    parents = data['parents']  # Plain list, keep as-is
    names = data['names']  # Plain list, keep as-is

    # Optional data - convert to numpy if present
    vertices = np.array(data['vertices'], dtype=np.float32) if 'vertices' in data else None
    faces = np.array(data['faces'], dtype=np.int32) if 'faces' in data else None
    skin = np.array(data['skin'], dtype=np.float32) if 'skin' in data else None
    tails = np.array(data['tails'], dtype=np.float32) if 'tails' in data else None

    print(f"[Blender FBX Export] Loaded skeleton with {len(joints)} joints")
    if vertices is not None:
        print(f"[Blender FBX Export] Found mesh with {len(vertices)} vertices")
except Exception as e:
    print(f"[Blender FBX Export] Failed to load pickle: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Clean default scene
def clean_bpy():
    for c in bpy.data.actions:
        bpy.data.actions.remove(c)
    for c in bpy.data.armatures:
        bpy.data.armatures.remove(c)
    for c in bpy.data.cameras:
        bpy.data.cameras.remove(c)
    for c in bpy.data.collections:
        bpy.data.collections.remove(c)
    for c in bpy.data.images:
        bpy.data.images.remove(c)
    for c in bpy.data.materials:
        bpy.data.materials.remove(c)
    for c in bpy.data.meshes:
        bpy.data.meshes.remove(c)
    for c in bpy.data.objects:
        bpy.data.objects.remove(c)
    for c in bpy.data.textures:
        bpy.data.textures.remove(c)

clean_bpy()

# Make collection
collection = bpy.data.collections.new('new_collection')
bpy.context.scene.collection.children.link(collection)

# Make mesh if vertices provided
if vertices is not None:
    mesh = bpy.data.meshes.new('mesh')
    if faces is None:
        faces = []
    mesh.from_pydata(vertices.tolist(), [], faces.tolist() if faces is not None else [])
    mesh.update()

    # Make object from mesh
    obj = bpy.data.objects.new('character', mesh)

    # Add object to scene collection
    collection.objects.link(obj)

# Create armature
print("[Blender FBX Export] Creating armature...")
bpy.ops.object.armature_add(enter_editmode=True)
armature = bpy.data.armatures.get('Armature')
edit_bones = armature.edit_bones

J = joints.shape[0]
if tails is None:
    tails = joints.copy()
    tails[:, 2] += extrude_size

connects = [False for _ in range(J)]
children = defaultdict(list)
for i in range(1, J):
    if parents[i] is not None and parents[i] != -1:
        children[parents[i]].append(i)

if tails is not None:
    if use_extrude_bone:
        for i in range(J):
            if len(children[i]) != 1 and extrude_from_parent and i != 0:
                if parents[i] is not None and parents[i] != -1:
                    pjoint = joints[parents[i]]
                    joint = joints[i]
                    d = joint - pjoint
                    if np.linalg.norm(d) < 0.000001:
                        d = np.array([0., 0., 1.])  # in case son.head == parent.head
                    else:
                        d = d / np.linalg.norm(d)
                    tails[i] = joint + d * extrude_size
    if use_connect_unique_child:
        for i in range(J):
            if len(children[i]) == 1:
                child = children[i][0]
                tails[i] = joints[child]
            if parents[i] is not None and parents[i] != -1 and len(children[parents[i]]) == 1:
                connects[i] = True

# Create root bone
if add_root:
    bone_root = edit_bones.get('Bone')
    bone_root.name = 'Root'
    bone_root.tail = Vector((joints[0, 0], joints[0, 1], joints[0, 2]))
else:
    bone_root = edit_bones.get('Bone')
    bone_root.name = names[0]
    bone_root.head = Vector((joints[0, 0], joints[0, 1], joints[0, 2]))
    bone_root.tail = Vector((joints[0, 0], joints[0, 1], joints[0, 2] + extrude_size))

# Create bones
def extrude_bone(edit_bones, name, parent_name, head, tail, connect):
    bone = edit_bones.new(name)
    bone.head = Vector((head[0], head[1], head[2]))
    bone.tail = Vector((tail[0], tail[1], tail[2]))
    bone.name = name
    parent_bone = edit_bones.get(parent_name)
    bone.parent = parent_bone
    bone.use_connect = connect
    assert not np.isnan(head).any(), f"nan found in head of bone {name}"
    assert not np.isnan(tail).any(), f"nan found in tail of bone {name}"

for i in range(J):
    if add_root is False and i == 0:
        continue
    edit_bones = armature.edit_bones
    pname = 'Root' if parents[i] is None or parents[i] == -1 else names[parents[i]]
    extrude_bone(edit_bones, names[i], pname, joints[i], tails[i], connects[i])

# Update bone positions
for i in range(J):
    bone = edit_bones.get(names[i])
    bone.head = Vector((joints[i, 0], joints[i, 1], joints[i, 2]))
    bone.tail = Vector((tails[i, 0], tails[i, 1], tails[i, 2]))

# Add skinning weights if vertices and skin provided
if vertices is not None and skin is not None:
    print("[Blender FBX Export] Adding skinning weights...")
    # Must set to object mode to enable parent_set
    bpy.ops.object.mode_set(mode='OBJECT')
    objects = bpy.data.objects
    for o in bpy.context.selected_objects:
        o.select_set(False)
    ob = objects['character']
    arm = bpy.data.objects['Armature']
    ob.select_set(True)
    arm.select_set(True)
    bpy.ops.object.parent_set(type='ARMATURE_NAME')

    vis = []
    for x in ob.vertex_groups:
        vis.append(x.name)

    # Sparsify
    argsorted = np.argsort(-skin, axis=1)
    vertex_group_reweight = skin[np.arange(skin.shape[0])[..., None], argsorted]

    group_per_vertex = data.get('group_per_vertex', -1)
    if group_per_vertex == -1:
        group_per_vertex = vertex_group_reweight.shape[-1]

    do_not_normalize = data.get('do_not_normalize', False)
    if not do_not_normalize:
        vertex_group_reweight = vertex_group_reweight / vertex_group_reweight[..., :group_per_vertex].sum(axis=1)[..., None]

    for v, w in enumerate(skin):
        for ii in range(group_per_vertex):
            i = argsorted[v, ii]
            if i >= J:
                continue
            n = names[i]
            if n not in vis:
                continue
            ob.vertex_groups[n].add([v], vertex_group_reweight[v, ii], 'REPLACE')

# Export to FBX
print("[Blender FBX Export] Exporting to FBX...")
os.makedirs(os.path.dirname(output_fbx) if os.path.dirname(output_fbx) else '.', exist_ok=True)

try:
    bpy.ops.export_scene.fbx(filepath=output_fbx, check_existing=False, add_leaf_bones=False)
    print(f"[Blender FBX Export] Saved to: {output_fbx}")
    print("[Blender FBX Export] Done!")
except Exception as e:
    print(f"[Blender FBX Export] Export failed: {e}")
    sys.exit(1)
