/**
 * ComfyUI Compare Skeletons Widget
 *
 * Widget for UniRigCompareSkeletons node - displays two FBX skeletons side-by-side
 * with synced camera rotation and cross-bone highlighting.
 */

import { app } from "../../../../scripts/app.js";
import { detectExtensionFolder, getViewerUrl } from "./mesh_preview_fbx.js";

console.log("[UniRig Compare] Loading compare skeleton widget...");

const extensionFolder = detectExtensionFolder();
if (!extensionFolder) {
    console.error("[UniRig Compare] Could not detect extension folder");
}

console.log("[UniRig Compare] Detected extension folder:", extensionFolder);

app.registerExtension({
    name: "unirig.compareskeletons",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "UniRigCompareSkeletons") {
            console.log("[UniRig Compare] Registering Compare Skeletons node");

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                console.log("[UniRig Compare] Node created, adding compare viewer widget");

                // Create iframe for comparison viewer
                const iframe = document.createElement("iframe");
                iframe.style.width = "100%";
                iframe.style.flex = "1 1 0";
                iframe.style.minHeight = "0";
                iframe.style.border = "none";
                iframe.style.backgroundColor = "#2a2a2a";

                // Load viewer from file URL
                const viewerUrl = getViewerUrl(extensionFolder, "viewer_fbx_compare.html");
                iframe.src = viewerUrl;
                console.log("[UniRig Compare] Setting iframe src to:", viewerUrl);

                iframe.onload = () => {
                    console.log("[UniRig Compare] Iframe loaded successfully");
                };
                iframe.onerror = (e) => {
                    console.error("[UniRig Compare] Iframe failed to load:", e);
                };

                // Add widget
                const widget = this.addDOMWidget("preview", "FBX_COMPARE_PREVIEW", iframe, {
                    getValue() { return ""; },
                    setValue(v) { }
                });

                console.log("[UniRig Compare] Widget created:", widget);

                // Set widget size - wider for side-by-side view
                widget.computeSize = function(width) {
                    const w = width || 800;
                    const h = w * 0.75;  // Wider aspect ratio for dual view
                    return [w, h];
                };

                widget.element = iframe;

                // Store iframe reference
                this.compareViewerIframe = iframe;
                this.compareViewerReady = false;

                // Listen for ready message from iframe
                const onMessage = (event) => {
                // Provenance guard: the 'message' event is window-wide, so without
                // this we also handle every OTHER pack's iframe messages.
                if (event.source !== iframe.contentWindow) return;
                    if (event.data && event.data.type === 'VIEWER_READY') {
                        console.log("[UniRig Compare] Viewer iframe is ready!");
                        this.compareViewerReady = true;
                    }
                };
                window.addEventListener('message', onMessage.bind(this));

                const notifyIframeResize = () => {
                    if (iframe.contentWindow) {
                        const rect = iframe.getBoundingClientRect();
                        iframe.contentWindow.postMessage({
                            type: 'RESIZE',
                            width: rect.width,
                            height: rect.height
                        }, '*');
                    }
                };

                this.onResize = function(size) {
                    const isVueNodes = iframe.closest('[data-node-id]') !== null ||
                                       document.querySelector('.vue-graph-canvas') !== null;

                    if (!isVueNodes && size && size[1]) {
                        const nodeHeight = size[1];
                        const headerHeight = 70;
                        const availableHeight = Math.max(200, nodeHeight - headerHeight);
                        iframe.style.height = availableHeight + 'px';
                    }

                    requestAnimationFrame(() => {
                        notifyIframeResize();
                    });
                };

                let resizeTimeout = null;
                let lastSize = { width: 0, height: 0 };
                const resizeObserver = new ResizeObserver((entries) => {
                    const entry = entries[0];
                    const newWidth = entry.contentRect.width;
                    const newHeight = entry.contentRect.height;

                    if (Math.abs(newWidth - lastSize.width) < 1 && Math.abs(newHeight - lastSize.height) < 1) {
                        return;
                    }
                    lastSize = { width: newWidth, height: newHeight };

                    if (resizeTimeout) {
                        clearTimeout(resizeTimeout);
                    }
                    resizeTimeout = setTimeout(() => {
                        notifyIframeResize();
                    }, 50);
                });
                resizeObserver.observe(iframe);

                const originalOnRemoved = this.onRemoved;
                this.onRemoved = function() {
                    resizeObserver.disconnect();
                    if (resizeTimeout) {
                        clearTimeout(resizeTimeout);
                    }
                    window.removeEventListener('message', onMessage);
                    if (originalOnRemoved) {
                        originalOnRemoved.apply(this, arguments);
                    }
                };

                // Set initial node size (wider for dual view)
                this.setSize([800, 600]);

                // Handle execution
                const onExecuted = this.onExecuted;
                this.onExecuted = function(message) {
                    console.log("[UniRig Compare] onExecuted called with message:", message);
                    onExecuted?.apply(this, arguments);

                    // The message contains both FBX file paths
                    const leftFile = message?.fbx_file_left?.[0];
                    const rightFile = message?.fbx_file_right?.[0];

                    if (leftFile || rightFile) {
                        console.log("[UniRig Compare] Loading FBX files - Left:", leftFile, "Right:", rightFile);

                        // Build file paths
                        const buildFilepath = (filename) => {
                            if (!filename) return null;
                            if (!filename.includes('/') && !filename.includes('\\')) {
                                return `${window.location.origin}/view?filename=${encodeURIComponent(filename)}&type=output&subfolder=`;
                            } else {
                                const basename = filename.split(/[/\\]/).pop();
                                return `${window.location.origin}/view?filename=${encodeURIComponent(basename)}&type=output&subfolder=`;
                            }
                        };

                        const filepathLeft = buildFilepath(leftFile);
                        const filepathRight = buildFilepath(rightFile);

                        // Send message to iframe
                        const sendMessage = () => {
                            if (iframe.contentWindow) {
                                console.log("[UniRig Compare] Sending postMessage to iframe");
                                iframe.contentWindow.postMessage({
                                    type: "LOAD_FBX_COMPARE",
                                    filepathLeft: filepathLeft,
                                    filepathRight: filepathRight,
                                    timestamp: Date.now()
                                }, "*");
                            } else {
                                console.error("[UniRig Compare] Iframe contentWindow not available");
                            }
                        };

                        // Wait for iframe to be ready, or use timeout as fallback
                        if (this.compareViewerReady) {
                            sendMessage();
                        } else {
                            const checkReady = setInterval(() => {
                                if (this.compareViewerReady) {
                                    clearInterval(checkReady);
                                    sendMessage();
                                }
                            }, 50);

                            // Fallback timeout after 2 seconds
                            setTimeout(() => {
                                clearInterval(checkReady);
                                if (!this.compareViewerReady) {
                                    console.warn("[UniRig Compare] Iframe not ready after 2s, sending anyway");
                                    sendMessage();
                                }
                            }, 2000);
                        }
                    } else {
                        console.log("[UniRig Compare] No FBX files in message data. Keys:", Object.keys(message || {}));
                    }
                };

                return r;
            };
        }
    }
});

console.log("[UniRig Compare] Compare skeleton widget registered");
