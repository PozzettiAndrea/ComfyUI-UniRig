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
import tempfile
import base64
import struct
import zlib

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

    # Optional data - convert to numpy if present and not None
    vertices = np.array(data['vertices'], dtype=np.float32) if data.get('vertices') is not None else None
    faces = np.array(data['faces'], dtype=np.int32) if data.get('faces') is not None else None
    skin = np.array(data['skin'], dtype=np.float32) if data.get('skin') is not None else None
    tails = np.array(data['tails'], dtype=np.float32) if data.get('tails') is not None else None

    # UV data - convert to numpy if present
    uv_coords = np.array(data['uv_coords'], dtype=np.float32) if data.get('uv_coords') is not None and len(data.get('uv_coords', [])) > 0 else None
    uv_faces = np.array(data['uv_faces'], dtype=np.int32) if data.get('uv_faces') is not None and len(data.get('uv_faces', [])) > 0 else None

    # Texture data - keep as string (base64 encoded)
    texture_data_base64 = data.get('texture_data_base64', "")
    texture_format = data.get('texture_format', "PNG")
    material_name_from_data = data.get('material_name', "Material")

    if texture_data_base64 and len(texture_data_base64) > 0:
        print(f"[Blender FBX Export] Found texture data: {texture_format} ({len(texture_data_base64) // 1024}KB base64)")
    else:
        print(f"[Blender FBX Export] No texture data found")

    print(f"[Blender FBX Export] Loaded skeleton with {len(joints)} joints")
    print(f"[Blender FBX Export] Joints shape: {joints.shape}")
    print(f"[Blender FBX Export] Tails: {'None' if tails is None else f'shape {tails.shape}'}")
    if vertices is not None:
        print(f"[Blender FBX Export] Found mesh with {len(vertices)} vertices")
    if skin is not None:
        print(f"[Blender FBX Export] Skin weights shape: {skin.shape}")

    # DEBUG: Show bounds of what Blender is receiving
    print(f"[Blender FBX Export] DEBUG - Joints bounds: {joints.min(axis=0)} to {joints.max(axis=0)}")
    if tails is not None:
        print(f"[Blender FBX Export] DEBUG - Tails bounds: {tails.min(axis=0)} to {tails.max(axis=0)}")
    if vertices is not None:
        print(f"[Blender FBX Export] DEBUG - Mesh vertices bounds: {vertices.min(axis=0)} to {vertices.max(axis=0)}")

        # Debug: Y position of vertex with highest Z value
        max_z_idx = vertices[:, 2].argmax()
        max_z_vertex = vertices[max_z_idx]
        print(f"[Blender FBX Export] DEBUG - Mesh vertex with max Z: position={max_z_vertex}, Z={max_z_vertex[2]:.6f}, Y={max_z_vertex[1]:.6f}")

    max_z_joint_idx = joints[:, 2].argmax()
    max_z_joint = joints[max_z_joint_idx]
    print(f"[Blender FBX Export] DEBUG - Skeleton joint with max Z: position={max_z_joint}, Z={max_z_joint[2]:.6f}, Y={max_z_joint[1]:.6f}")
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

    # Add UV coordinates if available
    if uv_coords is not None and uv_faces is not None and len(uv_coords) > 0:
        print(f"[Blender FBX Export] Adding UV coordinates: {len(uv_coords)} UVs for {len(uv_faces)} faces")
        # Create UV layer
        uv_layer = mesh.uv_layers.new(name='UVMap')

        # Apply UV coordinates per face loop
        for face_idx, poly in enumerate(mesh.polygons):
            if face_idx < len(uv_faces):
                for loop_offset, loop_idx in enumerate(poly.loop_indices):
                    uv_idx = uv_faces[face_idx][loop_offset]
                    if uv_idx < len(uv_coords):
                        uv_layer.data[loop_idx].uv = uv_coords[uv_idx]

        print(f"[Blender FBX Export] ✓ UV coordinates applied")
    else:
        print(f"[Blender FBX Export] No UV coordinates available")

    # Make object from mesh
    obj = bpy.data.objects.new('character', mesh)

    # Add object to scene collection
    collection.objects.link(obj)

    # Create and apply textured material if texture data is available
    if texture_data_base64 and len(texture_data_base64) > 0:
        print(f"[Blender FBX Export] Creating textured material...")
        try:
            # Decode base64 PNG to raw bytes
            png_data = base64.b64decode(texture_data_base64)

            # Save PNG to temporary file (no PIL needed - already PNG format)
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                tmp.write(png_data)
                tmp_texture_path = tmp.name

            print(f"[Blender FBX Export] Saved texture to temp file: {tmp_texture_path}")
            print(f"[Blender FBX Export] PNG data size: {len(png_data)} bytes")

            # Create material with nodes
            mat = bpy.data.materials.new(name=material_name_from_data)
            mat.use_nodes = True

            # Clear default nodes
            mat.node_tree.nodes.clear()

            # Create nodes
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links

            # Create image texture node
            img_node = nodes.new(type='ShaderNodeTexImage')
            img_node.location = (-300, 300)

            # Create principled BSDF node
            bsdf_node = nodes.new(type='ShaderNodeBsdfPrincipled')
            bsdf_node.location = (0, 300)

            # Set material to be non-metallic and fairly rough (more diffuse/matte)
            # This prevents the shiny/translucent look when only diffuse texture is available
            bsdf_node.inputs['Metallic'].default_value = 0.0  # Non-metallic
            bsdf_node.inputs['Roughness'].default_value = 0.8  # Fairly rough/matte
            bsdf_node.inputs['Specular IOR Level'].default_value = 0.3  # Reduced specular

            # Create output node
            output_node = nodes.new(type='ShaderNodeOutputMaterial')
            output_node.location = (300, 300)

            # Link nodes: Image -> BSDF -> Output
            links.new(img_node.outputs['Color'], bsdf_node.inputs['Base Color'])
            links.new(bsdf_node.outputs['BSDF'], output_node.inputs['Surface'])

            # Load image into Blender (Blender can load PNG natively)
            blender_image = bpy.data.images.load(tmp_texture_path)
            img_node.image = blender_image

            print(f"[Blender FBX Export] Loaded texture: {blender_image.size[0]}x{blender_image.size[1]}")

            # Pack image into blend file (so it's embedded in FBX)
            blender_image.pack()

            print(f"[Blender FBX Export] Packed texture image into blend file")

            # Assign material to mesh
            obj.data.materials.append(mat)

            # Assign material to all faces
            for poly in obj.data.polygons:
                poly.material_index = 0

            print(f"[Blender FBX Export] ✓ Textured material applied: {material_name_from_data}")

            # Clean up temp file
            try:
                os.remove(tmp_texture_path)
            except:
                pass

        except Exception as tex_err:
            print(f"[Blender FBX Export] Warning: Could not apply texture: {tex_err}")
            import traceback
            traceback.print_exc()
    else:
        print(f"[Blender FBX Export] No texture to apply")

# Create armature
print("[Blender FBX Export] Creating armature...")
try:
    bpy.ops.object.armature_add(enter_editmode=True)
    armature = bpy.data.armatures.get('Armature')
    edit_bones = armature.edit_bones

    J = joints.shape[0]
    if tails is None:
        print(f"[Blender FBX Export] Tails not provided, auto-generating...")
        tails = joints.copy()
        tails[:, 2] += extrude_size
        print(f"[Blender FBX Export] Generated tails shape: {tails.shape}")

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

    # Apply identity transforms to prevent FBX coordinate mismatches
    # This ensures mesh and armature have identical world transforms
    # Required for both skeleton-only and skinned exports
    bpy.ops.object.mode_set(mode='OBJECT')

    # Get armature object
    arm_obj = bpy.data.objects.get('Armature')
    if arm_obj:
        arm_obj.location = (0, 0, 0)
        arm_obj.rotation_euler = (0, 0, 0)
        arm_obj.scale = (1, 1, 1)
        print("[Blender FBX Export] Set armature to identity transform")

    # Get mesh object if it exists
    if vertices is not None:
        mesh_obj = bpy.data.objects.get('character')
        if mesh_obj:
            mesh_obj.location = (0, 0, 0)
            mesh_obj.rotation_euler = (0, 0, 0)
            mesh_obj.scale = (1, 1, 1)
            print("[Blender FBX Export] Set mesh to identity transform")

    # Add skinning weights if vertices and skin provided
    if vertices is not None and skin is not None:
        print("[Blender FBX Export] Adding skinning weights...")
        # Already in OBJECT mode from identity transform setup above
        objects = bpy.data.objects
        for o in bpy.context.selected_objects:
            o.select_set(False)
        ob = objects['character']
        arm = bpy.data.objects['Armature']

        # Parent mesh to armature using ARMATURE_NAME method
        # This creates vertex groups AND establishes proper parent-child hierarchy
        # Matches original UniRig implementation for correct FBX structure
        ob.select_set(True)
        arm.select_set(True)
        bpy.ops.object.parent_set(type='ARMATURE_NAME')
        print("[Blender FBX Export] Parented mesh to armature (creates vertex groups)")

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

    print("[Blender FBX Export] ✓ Armature created successfully")

except Exception as e:
    print(f"[Blender FBX Export] Armature creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Export to FBX
print("[Blender FBX Export] Exporting to FBX...")
os.makedirs(os.path.dirname(output_fbx) if os.path.dirname(output_fbx) else '.', exist_ok=True)

try:
    # Export with embedded textures
    # CRITICAL: Disable unit scaling to prevent 100x meter→centimeter conversion
    # The 15.6 unit misalignment was caused by apply_unit_scale defaulting to True
    # This converts Blender meters to FBX centimeters (100x scale factor)
    bpy.ops.export_scene.fbx(
        filepath=output_fbx,
        check_existing=False,
        add_leaf_bones=False,
        path_mode='COPY',  # Copy textures alongside FBX
        embed_textures=True,  # Embed textures in FBX file
        global_scale=1.0,  # Keep 1:1 scale
        apply_unit_scale=False,  # CRITICAL: Don't convert meters→cm (prevents 100x scale)
        apply_scale_options='FBX_SCALE_NONE',  # Don't bake scale into transforms
    )
    print(f"[Blender FBX Export] Saved to: {output_fbx}")
    if texture_data_base64 and len(texture_data_base64) > 0:
        print(f"[Blender FBX Export] ✓ Textures embedded in FBX")
    print("[Blender FBX Export] Done!")
except Exception as e:
    print(f"[Blender FBX Export] Export failed: {e}")
    sys.exit(1)
