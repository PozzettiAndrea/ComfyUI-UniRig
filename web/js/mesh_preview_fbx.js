/**
 * ComfyUI UniRig - FBX Rigged Mesh Preview Widget
 * Interactive viewer for FBX files with skeleton manipulation
 */

import { app } from "../../../../scripts/app.js";
import { VIEWER_HTML } from "./viewer_inline.js";

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
                iframe.style.flex = "1 1 0";
                iframe.style.minHeight = "0";
                iframe.style.border = "none";
                iframe.style.backgroundColor = "#2a2a2a";

                // Create blob URL from inline HTML (no external requests!)
                const blob = new Blob([VIEWER_HTML], { type: 'text/html' });
                const blobUrl = URL.createObjectURL(blob);
                iframe.src = blobUrl;
                console.log('[UniRig] Setting iframe src to blob URL (fully self-contained)');

                // Clean up blob URL when iframe is removed
                iframe.addEventListener('load', () => {
                    // Keep blob URL alive while iframe is loaded
                    iframe._blobUrl = blobUrl;
                });

                // Add load event listener
                iframe.onload = () => {
                    console.log('[UniRig] Iframe loaded successfully');
                };
                iframe.onerror = (e) => {
                    console.error('[UniRig] Iframe failed to load:', e);
                };

                // Add widget
                const widget = this.addDOMWidget("preview", "FBX_PREVIEW", iframe, {
                    getValue() { return ""; },
                    setValue(v) { }
                });

                console.log("[UniRig] Widget created:", widget);

                // Set widget size - allow flexible height
                widget.computeSize = function(width) {
                    const w = width || 512;
                    const h = w * 1.5;  // Taller than wide to accommodate controls (increased from 1.25)
                    return [w, h];
                };

                widget.element = iframe;

                // Store iframe reference
                this.fbxViewerIframe = iframe;
                this.fbxViewerReady = false;

                // Listen for ready message from iframe
                const onMessage = (event) => {
                    console.log('[UniRig] Received message from iframe:', event.data);
                    if (event.data && event.data.type === 'VIEWER_READY') {
                        console.log('[UniRig] Viewer iframe is ready!');
                        this.fbxViewerReady = true;
                    }
                };
                window.addEventListener('message', onMessage.bind(this));

                // Set initial node size (taller to accommodate controls)
                this.setSize([512, 768]);

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

                        // If filename is just a basename, it's in output
                        if (!filename.includes('/') && !filename.includes('\\')) {
                            // Try output directory first
                            filepath = `/view?filename=${encodeURIComponent(filename)}&type=output&subfolder=`;
                            console.log(`[UniRig] Using output path: ${filepath}`);
                        } else {
                            // Full path - extract just the filename
                            const basename = filename.split(/[/\\]/).pop();
                            filepath = `/view?filename=${encodeURIComponent(basename)}&type=output&subfolder=`;
                            console.log(`[UniRig] Extracted basename: ${basename}, path: ${filepath}`);
                        }

                        // Send message to iframe (wait for ready or use delay)
                        const sendMessage = () => {
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
                        };

                        // Wait for iframe to be ready, or use timeout as fallback
                        if (this.fbxViewerReady) {
                            sendMessage();
                        } else {
                            const checkReady = setInterval(() => {
                                if (this.fbxViewerReady) {
                                    clearInterval(checkReady);
                                    sendMessage();
                                }
                            }, 50);

                            // Fallback timeout after 2 seconds
                            setTimeout(() => {
                                clearInterval(checkReady);
                                if (!this.fbxViewerReady) {
                                    console.warn("[UniRig] Iframe not ready after 2s, sending anyway");
                                    sendMessage();
                                }
                            }, 2000);
                        }
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
