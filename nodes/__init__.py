"""
UniRig ComfyUI nodes package.
"""

from .base import (
    NODE_DIR,
    LIB_DIR,
    UNIRIG_PATH,
    UNIRIG_MODELS_DIR,
    BLENDER_EXE,
    BLENDER_SCRIPT,
    BLENDER_PARSE_SKELETON,
    BLENDER_EXTRACT_MESH_INFO,
)

from .model_loaders import UniRigLoadSkeletonModel, UniRigLoadSkinningModel
from .skeleton_extraction import UniRigExtractSkeleton
from .skeleton_io import (
    UniRigSaveSkeleton,
    UniRigSaveRiggedMesh,
    UniRigLoadRiggedMesh,
    UniRigPreviewRiggedMesh,
)
from .skinning import UniRigApplySkinningML
from .mesh_io import UniRigLoadMesh, UniRigSaveMesh

NODE_CLASS_MAPPINGS = {
    "UniRigLoadSkeletonModel": UniRigLoadSkeletonModel,
    "UniRigLoadSkinningModel": UniRigLoadSkinningModel,
    "UniRigExtractSkeleton": UniRigExtractSkeleton,
    "UniRigSaveSkeleton": UniRigSaveSkeleton,
    "UniRigSaveRiggedMesh": UniRigSaveRiggedMesh,
    "UniRigLoadRiggedMesh": UniRigLoadRiggedMesh,
    "UniRigPreviewRiggedMesh": UniRigPreviewRiggedMesh,
    "UniRigApplySkinningML": UniRigApplySkinningML,
    "UniRigLoadMesh": UniRigLoadMesh,
    "UniRigSaveMesh": UniRigSaveMesh,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UniRigLoadSkeletonModel": "UniRig: Load Skeleton Model",
    "UniRigLoadSkinningModel": "UniRig: Load Skinning Model",
    "UniRigExtractSkeleton": "UniRig: Extract Skeleton",
    "UniRigSaveSkeleton": "UniRig: Save Skeleton",
    "UniRigSaveRiggedMesh": "UniRig: Save Rigged Mesh",
    "UniRigLoadRiggedMesh": "UniRig: Load Rigged Mesh",
    "UniRigPreviewRiggedMesh": "UniRig: Preview Rigged Mesh",
    "UniRigApplySkinningML": "UniRig: Apply Skinning",
    "UniRigLoadMesh": "UniRig: Load Mesh",
    "UniRigSaveMesh": "UniRig: Save Mesh",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
