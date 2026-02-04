"""ComfyUI-UniRig - Automatic rigging and skeleton extraction."""

from comfy_env import wrap_nodes
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

wrap_nodes()

# FBX files API (used by skeleton_io.py remote dropdown)
try:
    from server import PromptServer
    from aiohttp import web

    @PromptServer.instance.routes.get('/unirig/fbx_files')
    async def get_fbx_files(request):
        try:
            from .nodes.skeleton_io import UniRigLoadRiggedMesh
            source = request.query.get('source_folder', 'output')
            if source == "input":
                files = UniRigLoadRiggedMesh.get_fbx_files_from_input()
            else:
                files = UniRigLoadRiggedMesh.get_fbx_files_from_output()
            return web.json_response(files or [])
        except Exception as e:
            return web.json_response({'error': str(e)}, status=500)
except Exception:
    pass

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
