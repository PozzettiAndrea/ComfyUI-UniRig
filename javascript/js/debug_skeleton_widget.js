import { createFBXPreviewExtension } from "./mesh_preview_fbx.js";

// Register UniRigDebugSkeleton with the debug viewer
createFBXPreviewExtension({
    nodeName: "UniRigDebugSkeleton",
    extensionName: "unirig.debugskeleton",
    logPrefix: "[UniRig Debug]",
    fbxExportApiPath: "/unirig/export_posed_fbx",
    viewerFile: "viewer_fbx_debug.html"
});
