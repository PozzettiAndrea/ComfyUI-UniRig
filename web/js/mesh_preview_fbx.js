/**
 * ComfyUI UniRig - FBX Rigged Mesh Preview Widget
 * Interactive viewer for FBX files with skeleton manipulation
 */

import { app } from "../../../../scripts/app.js";

console.log("[UniRig] Loading FBX mesh preview extension...");

app.registerExtension({
    name: "unirig.fbxpreview",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "UniRigPreviewRiggedMesh") {
            console.log("[UniRig] Registering Preview Rigged Mesh node");

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                console.log("[UniRig] Node created, adding FBX viewer widget");

                // Create iframe for FBX viewer
                const iframe = document.createElement("iframe");
                iframe.style.width = "100%";
                iframe.style.height = "100%";
                iframe.style.border = "none";
                iframe.style.backgroundColor = "#2a2a2a";
                iframe.style.aspectRatio = "1";

                // Point to our FBX viewer HTML (with cache buster)
                iframe.src = "/extensions/ComfyUI-UniRig/viewer_fbx.html?v=" + Date.now();

                // Add widget
                const widget = this.addDOMWidget("preview", "FBX_PREVIEW", iframe, {
                    getValue() { return ""; },
                    setValue(v) { }
                });

                console.log("[UniRig] Widget created:", widget);

                // Set widget size
                widget.computeSize = function(width) {
                    const size = [width || 512, width || 512];
                    return size;
                };

                widget.element = iframe;

                // Store iframe reference
                this.fbxViewerIframe = iframe;

                // Set initial node size to be square
                this.setSize([512, 512]);

                // Handle execution
                const onExecuted = this.onExecuted;
                this.onExecuted = function(message) {
                    console.log("[UniRig] onExecuted called with message:", message);
                    onExecuted?.apply(this, arguments);

                    // The message contains the FBX file path
                    if (message?.fbx_file && message.fbx_file[0]) {
                        const filename = message.fbx_file[0];
                        console.log(`[UniRig] Loading FBX: ${filename}`);

                        // Try different path formats based on filename
                        let filepath;

                        // If filename is just a basename, it's in temp
                        if (!filename.includes('/') && !filename.includes('\\')) {
                            // Try temp directory first
                            filepath = `/view?filename=${encodeURIComponent(filename)}&type=temp&subfolder=`;
                            console.log(`[UniRig] Using temp path: ${filepath}`);
                        } else {
                            // Full path - extract just the filename
                            const basename = filename.split(/[/\\]/).pop();
                            filepath = `/view?filename=${encodeURIComponent(basename)}&type=temp&subfolder=`;
                            console.log(`[UniRig] Extracted basename: ${basename}, path: ${filepath}`);
                        }

                        // Send message to iframe (with delay to ensure iframe is loaded)
                        setTimeout(() => {
                            if (iframe.contentWindow) {
                                console.log(`[UniRig] Sending postMessage to iframe: ${filepath}`);
                                iframe.contentWindow.postMessage({
                                    type: "LOAD_FBX",
                                    filepath: filepath,
                                    timestamp: Date.now()
                                }, "*");
                            } else {
                                console.error("[UniRig] Iframe contentWindow not available");
                            }
                        }, 100);
                    } else {
                        console.log("[UniRig] No fbx_file in message data. Keys:", Object.keys(message || {}));
                    }
                };

                return r;
            };
        }
    }
});

console.log("[UniRig] FBX mesh preview extension registered");
