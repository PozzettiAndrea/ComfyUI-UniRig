"""
ComfyUI-UniRig

UniRig integration for ComfyUI - State-of-the-art automatic rigging and skeleton extraction.

Based on: One Model to Rig Them All (SIGGRAPH 2025)
Repository: https://github.com/VAST-AI-Research/UniRig
"""

import os
import sys
import traceback

# Track initialization status
INIT_SUCCESS = False
INIT_ERRORS = []

# Set web directory for JavaScript extensions (FBX viewer widget)
WEB_DIRECTORY = "./web"

# Only run initialization and imports when loaded by ComfyUI, not during pytest
# This prevents relative import errors when pytest collects test modules
if 'pytest' not in sys.modules:
    print("[ComfyUI-UniRig] Initializing custom node...")

    # Import node classes
    try:
        from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
        print("[ComfyUI-UniRig] [OK] Node classes imported successfully")
        INIT_SUCCESS = True
    except Exception as e:
        error_msg = f"Failed to import node classes: {str(e)}"
        INIT_ERRORS.append(error_msg)
        print(f"[ComfyUI-UniRig] [WARNING] {error_msg}")
        print(f"[ComfyUI-UniRig] Traceback:\n{traceback.format_exc()}")

        # Set to empty if import failed
        NODE_CLASS_MAPPINGS = {}
        NODE_DISPLAY_NAME_MAPPINGS = {}

    # Add static route for Three.js and other libraries
    try:
        from server import PromptServer
        from aiohttp import web
        import json
        import tempfile
        import subprocess
        import time
        import folder_paths

        static_path = os.path.join(os.path.dirname(__file__), "static")
        if os.path.exists(static_path):
            PromptServer.instance.app.add_routes([
                web.static('/extensions/ComfyUI-UniRig/static', static_path)
            ])

        # Add custom API endpoint for FBX export
        @PromptServer.instance.routes.post('/unirig/export_posed_fbx')
        async def export_posed_fbx(request):
            """API endpoint to export FBX with custom pose."""
            try:
                data = await request.json()
                fbx_filename = data.get('fbx_filename')
                bone_transforms = data.get('bone_transforms', {})
                custom_output_filename = data.get('output_filename')

                if not fbx_filename:
                    return web.json_response({'error': 'Missing fbx_filename'}, status=400)

                # Find the FBX file in the output directory
                output_dir = folder_paths.get_output_directory()
                fbx_path = os.path.join(output_dir, fbx_filename)

                if not os.path.exists(fbx_path):
                    return web.json_response({'error': f'FBX file not found: {fbx_filename}'}, status=404)

                # Save bone transforms to temporary JSON file
                temp_dir = folder_paths.get_temp_directory()
                transforms_json_path = os.path.join(temp_dir, f"bone_transforms_{int(time.time())}.json")
                with open(transforms_json_path, 'w') as f:
                    json.dump(bone_transforms, f)

                # Prepare output path - use custom filename if provided
                if custom_output_filename:
                    output_filename = custom_output_filename
                    # Ensure .fbx extension
                    if not output_filename.lower().endswith('.fbx'):
                        output_filename = output_filename + '.fbx'
                else:
                    output_filename = f"posed_export_{int(time.time())}.fbx"

                output_fbx_path = os.path.join(output_dir, output_filename)

                # Get paths to Blender and script
                from .nodes.base import BLENDER_EXE, NODE_DIR
                from .constants import BLENDER_TIMEOUT

                blender_script = os.path.join(NODE_DIR, 'lib', 'blender_export_posed_fbx.py')

                if not os.path.exists(blender_script):
                    return web.json_response({'error': 'Blender export script not found'}, status=500)

                if not os.path.exists(BLENDER_EXE):
                    return web.json_response({'error': 'Blender executable not found'}, status=500)

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

                # Run Blender
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=BLENDER_TIMEOUT)

                # Print Blender output for debugging
                if result.stdout:
                    print("[UniRig Export API] Blender output:")
                    print(result.stdout)

                # Clean up temporary JSON file
                if os.path.exists(transforms_json_path):
                    os.unlink(transforms_json_path)

                if result.returncode != 0:
                    print(f"[UniRig Export API] Blender export failed: {result.stderr}")
                    return web.json_response({'error': f'Blender export failed: {result.stderr}'}, status=500)

                if not os.path.exists(output_fbx_path):
                    return web.json_response({'error': 'Export completed but output file not found'}, status=500)

                print(f"[UniRig Export API] ✓ Successfully exported to: {output_fbx_path}")

                return web.json_response({
                    'success': True,
                    'filename': output_filename,
                    'message': f'FBX exported successfully: {output_filename}'
                })

            except Exception as e:
                print(f"[UniRig Export API] Error: {e}")
                import traceback
                traceback.print_exc()
                return web.json_response({'error': str(e)}, status=500)

        # Add API endpoint for dynamic FBX file list (for refresh button)
        @PromptServer.instance.routes.get('/unirig/fbx_files')
        async def get_fbx_files(request):
            """API endpoint to fetch FBX file list dynamically."""
            try:
                # Import here to avoid circular dependencies
                from .nodes.skeleton_io import UniRigLoadRiggedMesh

                # Get source from query parameter (default to output)
                source = request.query.get('source_folder', 'output')

                if source == "input":
                    files = UniRigLoadRiggedMesh.get_fbx_files_from_input()
                else:
                    files = UniRigLoadRiggedMesh.get_fbx_files_from_output()

                if not files:
                    files = []

                print(f"[UniRig FBX Files API] Returning {len(files)} files from {source} directory")

                return web.json_response(files)

            except Exception as e:
                print(f"[UniRig FBX Files API] Error: {e}")
                import traceback
                traceback.print_exc()
                return web.json_response({'error': str(e)}, status=500)

    except Exception as e:
        print(f"[ComfyUI-UniRig] Warning: Could not add routes: {e}")

    # Report final status
    if INIT_SUCCESS:
        print("[ComfyUI-UniRig] [OK] Loaded successfully!")
    else:
        print(f"[ComfyUI-UniRig] [ERROR] Failed to load ({len(INIT_ERRORS)} error(s)):")
        for error in INIT_ERRORS:
            print(f"  - {error}")
        print("[ComfyUI-UniRig] Please check the errors above and your installation.")

else:
    # During testing, set dummy values to prevent import errors
    print("[ComfyUI-UniRig] Running in pytest mode - skipping initialization")
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
__version__ = "1.0.0"
