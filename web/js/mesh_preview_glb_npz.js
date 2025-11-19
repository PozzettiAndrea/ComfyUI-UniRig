/**
 * ComfyUI UniRig - GLB + NPZ Preview Widget
 * Interactive viewer for GLB mesh and NPZ skeleton data
 */

import { app } from "../../../../scripts/app.js";

console.log("[UniRig] Loading GLB+NPZ preview extension...");

app.registerExtension({
    name: "unirig.glbnpzpreview",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "UniRigPreviewGLBAndNPZ") {
            console.log("[UniRig] Registering Preview GLB+NPZ node");

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                console.log("[UniRig] Node created, adding GLB+NPZ viewer widget");

                // Create iframe for GLB+NPZ viewer
                const iframe = document.createElement("iframe");
                iframe.style.width = "100%";
                iframe.style.flex = "1 1 0";
                iframe.style.minHeight = "0";
                iframe.style.border = "none";
                iframe.style.backgroundColor = "#2a2a2a";

                // Point to our GLB+NPZ viewer HTML (with cache buster)
                iframe.src = "/extensions/ComfyUI-UniRig/viewer_glb_npz.html?v=" + Date.now();
                console.log('[UniRig] Setting iframe src to:', iframe.src);

                // Add load event listener
                iframe.onload = () => {
                    console.log('[UniRig] GLB+NPZ iframe loaded successfully');
                };
                iframe.onerror = (e) => {
                    console.error('[UniRig] GLB+NPZ iframe failed to load:', e);
                };

                // Add widget
                const widget = this.addDOMWidget("preview", "GLB_NPZ_PREVIEW", iframe, {
                    getValue() { return ""; },
                    setValue(v) { }
                });

                console.log("[UniRig] Widget created:", widget);

                // Set widget size - allow flexible height
                widget.computeSize = function(width) {
                    const w = width || 512;
                    const h = w * 1.5;  // Taller than wide to accommodate controls
                    return [w, h];
                };

                widget.element = iframe;

                // Store iframe reference
                this.glbNpzViewerIframe = iframe;
                this.glbNpzViewerReady = false;

                // Listen for ready message from iframe
                const onMessage = (event) => {
                    console.log('[UniRig] Received message from GLB+NPZ iframe:', event.data);
                    if (event.data && event.data.type === 'VIEWER_READY') {
                        console.log('[UniRig] GLB+NPZ viewer iframe is ready!');
                        this.glbNpzViewerReady = true;
                    }
                };
                window.addEventListener('message', onMessage.bind(this));

                // Set initial node size (taller to accommodate controls)
                this.setSize([512, 768]);

                // Handle execution
                const onExecuted = this.onExecuted;
                this.onExecuted = function(message) {
                    console.log("[UniRig] GLB+NPZ onExecuted called with message:", message);
                    onExecuted?.apply(this, arguments);

                    // The message contains the GLB filename and skeleton data
                    if (message?.glb_file && message.glb_file[0] && message?.skeleton_data && message.skeleton_data[0]) {
                        const filename = message.glb_file[0];
                        const skeletonData = message.skeleton_data[0];

                        console.log(`[UniRig] Loading GLB: ${filename}`);
                        console.log(`[UniRig] Skeleton data:`, skeletonData);

                        // Build GLB path
                        let glbPath;
                        if (!filename.includes('/') && !filename.includes('\\')) {
                            // Just a basename - in output directory
                            glbPath = `/view?filename=${encodeURIComponent(filename)}&type=output&subfolder=`;
                            console.log(`[UniRig] Using output path: ${glbPath}`);
                        } else {
                            // Full path - extract basename
                            const basename = filename.split(/[/\\]/).pop();
                            glbPath = `/view?filename=${encodeURIComponent(basename)}&type=output&subfolder=`;
                            console.log(`[UniRig] Extracted basename: ${basename}, path: ${glbPath}`);
                        }

                        // Send message to iframe (wait for ready or use delay)
                        const sendMessage = () => {
                            if (iframe.contentWindow) {
                                console.log(`[UniRig] Sending postMessage to GLB+NPZ iframe`);
                                console.log(`[UniRig] GLB path: ${glbPath}`);
                                console.log(`[UniRig] Skeleton data:`, skeletonData);

                                iframe.contentWindow.postMessage({
                                    type: "LOAD_GLB_AND_SKELETON",
                                    glbPath: glbPath,
                                    skeletonData: skeletonData,
                                    timestamp: Date.now()
                                }, "*");
                            } else {
                                console.error("[UniRig] Iframe contentWindow not available");
                            }
                        };

                        // Wait for iframe to be ready, or use timeout as fallback
                        if (this.glbNpzViewerReady) {
                            sendMessage();
                        } else {
                            const checkReady = setInterval(() => {
                                if (this.glbNpzViewerReady) {
                                    clearInterval(checkReady);
                                    sendMessage();
                                }
                            }, 50);

                            // Fallback timeout after 2 seconds
                            setTimeout(() => {
                                clearInterval(checkReady);
                                if (!this.glbNpzViewerReady) {
                                    console.warn("[UniRig] GLB+NPZ iframe not ready after 2s, sending anyway");
                                    sendMessage();
                                }
                            }, 2000);
                        }
                    } else {
                        console.log("[UniRig] No glb_file or skeleton_data in message. Keys:", Object.keys(message || {}));
                    }
                };

                return r;
            };
        }
    }
});

console.log("[UniRig] GLB+NPZ preview extension registered");
