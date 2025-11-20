"""
Blender script to export FBX file with a custom pose applied.
This script loads an existing FBX, applies bone transforms, and exports the result.
Usage: blender --background --python blender_export_posed_fbx.py -- <input_fbx> <output_fbx> <transforms_json>
"""

import bpy
import sys
import os
import json
from mathutils import Vector, Quaternion, Euler

# Get arguments after '--'
argv = sys.argv
argv = argv[argv.index("--") + 1:] if "--" in argv else []

if len(argv) < 3:
    print("Usage: blender --background --python blender_export_posed_fbx.py -- <input_fbx> <output_fbx> <transforms_json>")
    sys.exit(1)

input_fbx = argv[0]
output_fbx = argv[1]
transforms_json = argv[2]

print(f"[Blender Posed FBX Export] Input FBX: {input_fbx}")
print(f"[Blender Posed FBX Export] Output FBX: {output_fbx}")
print(f"[Blender Posed FBX Export] Transforms: {transforms_json}")

# Clean default scene
def clean_bpy():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    for c in bpy.data.armatures:
        bpy.data.armatures.remove(c)
    for c in bpy.data.meshes:
        bpy.data.meshes.remove(c)
    for c in bpy.data.materials:
        bpy.data.materials.remove(c)
    for c in bpy.data.images:
        bpy.data.images.remove(c)

clean_bpy()

# Import FBX
try:
    print(f"[Blender Posed FBX Export] Importing FBX...")
    bpy.ops.import_scene.fbx(filepath=input_fbx)
    print(f"[Blender Posed FBX Export] ✓ FBX imported successfully")

    # Recreate materials from scratch to match original export encoding
    # This prevents transparency issues where Three.js interprets re-exported materials differently
    print(f"[Blender Posed FBX Export] Recreating materials from scratch...")

    # Extract texture data from imported materials before deleting them
    texture_images = {}
    old_material_names = {}

    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            for slot_idx, slot in enumerate(obj.material_slots):
                if slot.material and slot.material.use_nodes:
                    mat = slot.material
                    old_material_names[slot_idx] = mat.name

                    # Find texture image in material nodes
                    for node in mat.node_tree.nodes:
                        if node.type == 'TEX_IMAGE' and node.image:
                            # Store the image reference
                            texture_images[slot_idx] = node.image
                            print(f"[Blender Posed FBX Export] Found texture: {node.image.name} ({node.image.size[0]}x{node.image.size[1]})")
                            break

    # Delete all imported materials
    for mat in list(bpy.data.materials):
        bpy.data.materials.remove(mat)
    print(f"[Blender Posed FBX Export] Deleted imported materials")

    # Recreate materials from scratch for each mesh object (same as original export)
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            # Clear existing material slots
            obj.data.materials.clear()

            # Create new material from scratch
            for slot_idx, tex_image in texture_images.items():
                mat_name = old_material_names.get(slot_idx, 'material')

                # Create material with nodes (same as blender_export_fbx.py)
                mat = bpy.data.materials.new(name=mat_name)
                mat.use_nodes = True

                # Clear default nodes
                mat.node_tree.nodes.clear()

                # Create nodes
                nodes = mat.node_tree.nodes
                links = mat.node_tree.links

                # Create image texture node
                img_node = nodes.new(type='ShaderNodeTexImage')
                img_node.location = (-300, 300)
                img_node.image = tex_image

                # Create principled BSDF node
                bsdf_node = nodes.new(type='ShaderNodeBsdfPrincipled')
                bsdf_node.location = (0, 300)

                # Set material properties (same as original export)
                bsdf_node.inputs['Metallic'].default_value = 0.0
                bsdf_node.inputs['Roughness'].default_value = 0.8
                bsdf_node.inputs['Specular IOR Level'].default_value = 0.3

                # Create output node
                output_node = nodes.new(type='ShaderNodeOutputMaterial')
                output_node.location = (300, 300)

                # Link nodes: Image -> BSDF -> Output
                links.new(img_node.outputs['Color'], bsdf_node.inputs['Base Color'])
                links.new(bsdf_node.outputs['BSDF'], output_node.inputs['Surface'])

                # Assign material to mesh
                obj.data.materials.append(mat)

                print(f"[Blender Posed FBX Export] ✓ Recreated material: {mat_name}")

    print(f"[Blender Posed FBX Export] ✓ Recreated {len(texture_images)} materials from scratch")

except Exception as e:
    print(f"[Blender Posed FBX Export] Failed to import FBX: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Find armature object
armature_obj = None
for obj in bpy.data.objects:
    if obj.type == 'ARMATURE':
        armature_obj = obj
        break

if armature_obj is None:
    print(f"[Blender Posed FBX Export] Error: No armature found in FBX file")
    sys.exit(1)

print(f"[Blender Posed FBX Export] Found armature: {armature_obj.name}")
print(f"[Blender Posed FBX Export] Bones: {len(armature_obj.pose.bones)}")

# Load transform data from JSON
try:
    with open(transforms_json, 'r') as f:
        bone_transforms = json.load(f)

    print(f"[Blender Posed FBX Export] Loaded DELTA transforms (pose offsets from rest) for {len(bone_transforms)} bones")

except Exception as e:
    print(f"[Blender Posed FBX Export] Failed to load transforms: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Apply bone transforms in pose mode (using world-space to local-space conversion)
try:
    # Set active object
    bpy.context.view_layer.objects.active = armature_obj
    armature_obj.select_set(True)

    # Enter pose mode
    bpy.ops.object.mode_set(mode='POSE')

    # Apply DELTA transforms from Three.js as pose offsets
    # These are computed as (current - rest) in the viewer, which matches Blender's expectation
    # that pose_bone.location/rotation are OFFSETS from the rest pose

    applied_count = 0
    for bone_name, transform in bone_transforms.items():
        if bone_name in armature_obj.pose.bones:
            pose_bone = armature_obj.pose.bones[bone_name]

            # Apply delta transforms as pose offsets (this is what pose_bone.location expects)
            if 'position' in transform:
                pos = transform['position']
                pose_bone.location = Vector((pos['x'], pos['y'], pos['z']))

            if 'quaternion' in transform:
                quat = transform['quaternion']
                pose_bone.rotation_mode = 'QUATERNION'
                pose_bone.rotation_quaternion = Quaternion((quat['w'], quat['x'], quat['y'], quat['z']))

            if 'scale' in transform:
                scale = transform['scale']
                pose_bone.scale = Vector((scale['x'], scale['y'], scale['z']))

            applied_count += 1
        else:
            print(f"[Blender Posed FBX Export] Warning: Bone '{bone_name}' not found in armature")

    print(f"[Blender Posed FBX Export] ✓ Applied transforms to {applied_count}/{len(bone_transforms)} bones")

    # Return to object mode
    bpy.ops.object.mode_set(mode='OBJECT')

except Exception as e:
    print(f"[Blender Posed FBX Export] Failed to apply transforms: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Export to FBX
print("[Blender Posed FBX Export] Exporting to FBX...")
os.makedirs(os.path.dirname(output_fbx) if os.path.dirname(output_fbx) else '.', exist_ok=True)

try:
    # Use same export options as original to match encoding
    bpy.ops.export_scene.fbx(
        filepath=output_fbx,
        check_existing=False,
        add_leaf_bones=False,
        path_mode='COPY',
        embed_textures=True,
    )

    print(f"[Blender Posed FBX Export] ✓ Saved to: {output_fbx}")
    print("[Blender Posed FBX Export] Done!")
except Exception as e:
    print(f"[Blender Posed FBX Export] Export failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
