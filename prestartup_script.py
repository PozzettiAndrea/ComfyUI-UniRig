"""
ComfyUI-UniRig Prestartup Script

Handles setup tasks before node loading:
- Copy asset files to input/3d/
- Copy workflow examples to user/default/workflows/ with "UniRig -" prefix
- Create necessary directories
"""

import os
import shutil
import json
from pathlib import Path

# Get paths relative to this script
SCRIPT_DIR = Path(__file__).parent.absolute()
COMFYUI_DIR = SCRIPT_DIR.parent.parent  # custom_nodes/ComfyUI-UniRig -> custom_nodes -> ComfyUI

# Source directories
ASSETS_DIR = SCRIPT_DIR / "assets"
WORKFLOWS_DIR = SCRIPT_DIR / "workflows"

# Target directories (using relative paths from ComfyUI root)
INPUT_3D_DIR = COMFYUI_DIR / "input" / "3d"
USER_WORKFLOWS_DIR = COMFYUI_DIR / "user" / "default" / "workflows"


def copy_asset_files():
    """Copy necessary asset files to input/3d/ directory"""
    try:
        # Create target directory
        INPUT_3D_DIR.mkdir(parents=True, exist_ok=True)

        # Copy FinalBaseMesh.obj if it doesn't exist
        mesh_source = ASSETS_DIR / "FinalBaseMesh.obj"
        mesh_target = INPUT_3D_DIR / "FinalBaseMesh.obj"

        if mesh_source.exists():
            if not mesh_target.exists():
                shutil.copy2(str(mesh_source), str(mesh_target))
                print(f"[UniRig] Copied base mesh to {mesh_target}")
            else:
                print(f"[UniRig] Base mesh already exists at {mesh_target}")
        else:
            print(f"[UniRig] Warning: Source mesh not found at {mesh_source}")

    except Exception as e:
        print(f"[UniRig] Error copying asset files: {e}")
        import traceback
        traceback.print_exc()


def copy_workflow_files():
    """Copy workflow examples to user/default/workflows/ with 'UniRig -' prefix"""
    try:
        # Create target directory
        USER_WORKFLOWS_DIR.mkdir(parents=True, exist_ok=True)

        # Process each workflow JSON file
        if WORKFLOWS_DIR.exists():
            for workflow_file in WORKFLOWS_DIR.glob("*.json"):
                workflow_target = USER_WORKFLOWS_DIR / workflow_file.name

                if not workflow_target.exists():
                    try:
                        # Load the workflow JSON
                        with open(workflow_file, 'r', encoding='utf-8') as f:
                            workflow_data = json.load(f)

                        # Add "UniRig -" prefix to the workflow name if it has one
                        if isinstance(workflow_data, dict):
                            # Check for common workflow name fields
                            if 'name' in workflow_data and workflow_data['name']:
                                if not workflow_data['name'].startswith('UniRig -'):
                                    workflow_data['name'] = f"UniRig - {workflow_data['name']}"
                            elif 'title' in workflow_data and workflow_data['title']:
                                if not workflow_data['title'].startswith('UniRig -'):
                                    workflow_data['title'] = f"UniRig - {workflow_data['title']}"
                            # If no name/title field exists, add one
                            elif 'name' not in workflow_data:
                                workflow_data['name'] = f"UniRig - {workflow_file.stem}"

                        # Save the modified workflow
                        with open(workflow_target, 'w', encoding='utf-8') as f:
                            json.dump(workflow_data, f, indent=2)

                        print(f"[UniRig] Copied workflow '{workflow_file.name}' to {workflow_target}")

                    except json.JSONDecodeError as e:
                        print(f"[UniRig] Warning: Could not parse {workflow_file.name}: {e}")
                        # Fallback: just copy the file as-is
                        shutil.copy2(str(workflow_file), str(workflow_target))
                        print(f"[UniRig] Copied workflow (without modification) to {workflow_target}")
                else:
                    print(f"[UniRig] Workflow already exists at {workflow_target}")
        else:
            print(f"[UniRig] Warning: Workflows directory not found at {WORKFLOWS_DIR}")

    except Exception as e:
        print(f"[UniRig] Error copying workflow files: {e}")
        import traceback
        traceback.print_exc()


# Execute setup tasks
try:
    print("[UniRig] Running prestartup script...")
    copy_asset_files()
    copy_workflow_files()
    print("[UniRig] Prestartup script completed.")
except Exception as e:
    print(f"[UniRig] Error during prestartup: {e}")
    import traceback
    traceback.print_exc()
