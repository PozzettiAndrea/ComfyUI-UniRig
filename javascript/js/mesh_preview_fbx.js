/**
 * ComfyUI FBX Rigged Mesh Preview Widget
 *
 * Generic widget for FBX preview nodes. Loads viewer from file (not inline).
 * Used by UniRig, SAM3DBody, and other extensions that preview FBX files.
 *
 * This file auto-registers for all known FBX preview node types when loaded.
 * Extensions copy this file from the comfy-3d-viewers package at startup.
 */

import { app } from "../../../../scripts/app.js";

// Auto-register for all known FBX preview node types
// Only this pack's OWN preview nodes. The shared upstream copy of this file
// listed every sibling pack's node too, so installing two of them registered
// the same extension names twice and double-attached the widget.
const FBX_PREVIEW_NODES = [
    {
        nodeName: "UniRigPreviewRiggedMesh",
        extensionName: "unirig.fbxpreview",
        logPrefix: "[UniRig]",
        fbxExportApiPath: "/unirig/export_posed_fbx"
    }
];

/**
 * Detect the extension folder name from the current script URL
 */
function detectExtensionFolder() {
    try {
        // Try to detect from import.meta.url if available
        if (typeof import.meta !== 'undefined' && import.meta.url) {
            const match = import.meta.url.match(/\/extensions\/([^\/]+)\//);
            if (match) {
                return match[1];
            }
        }

        // Fallback: look at currently loading script
        const scripts = document.getElementsByTagName('script');
        for (let i = scripts.length - 1; i >= 0; i--) {
            const src = scripts[i].src;
            if (src) {
                const match = src.match(/\/extensions\/([^\/]+)\//);
                if (match) {
                    return match[1];
                }
            }
        }
    } catch (e) {
        console.warn('[FBX Preview] Could not detect extension folder:', e);
    }
    return null;
}

/**
 * Get the viewer URL for the FBX viewer HTML file
 * @param {string} extensionFolder - The extension folder name
 * @returns {string} The URL to the viewer HTML
 */
function getViewerUrl(extensionFolder, viewerFile = "viewer_fbx.html") {
    // Add cache-busting timestamp
    return `/extensions/${extensionFolder}/${viewerFile}?v=` + Date.now();
}

/**
 * Create and register an FBX preview extension
 * @param {Object} config - Configuration object
 * @param {string} config.extensionName - Unique extension name (e.g., "unirig.fbxpreview")
 * @param {string} config.nodeName - Node type name to hook (e.g., "UniRigPreviewRiggedMesh")
 * @param {string} config.logPrefix - Log prefix for console messages (e.g., "[UniRig]")
 * @param {string} [config.extensionFolder] - Override extension folder detection
 * @param {string} [config.fbxExportApiPath] - API path for FBX export (e.g., "/unirig/export_posed_fbx")
 */
export function createFBXPreviewExtension(config) {
    const {
        extensionName,
        nodeName,
        logPrefix,
        extensionFolder: configExtensionFolder,
        fbxExportApiPath
    } = config;

    console.log(`${logPrefix} Loading FBX mesh preview extension...`);

    // Detect extension folder if not provided
    const extensionFolder = configExtensionFolder || detectExtensionFolder();
    if (!extensionFolder) {
        console.error(`${logPrefix} Could not detect extension folder`);
        return;
    }
    console.log(`${logPrefix} Detected extension folder: ${extensionFolder}`);

    app.registerExtension({
        name: extensionName,

        async beforeRegisterNodeDef(nodeType, nodeData, app) {
            if (nodeData.name === nodeName) {
                console.log(`${logPrefix} Registering Preview Rigged Mesh node`);

                const onNodeCreated = nodeType.prototype.onNodeCreated;
                nodeType.prototype.onNodeCreated = function() {
                    const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                    console.log(`${logPrefix} Node created, adding FBX viewer widget`);

                    // Create iframe for FBX viewer
                    const iframe = document.createElement("iframe");
                    iframe.style.width = "100%";
                    iframe.style.flex = "1 1 0";
                    iframe.style.minHeight = "0";
                    iframe.style.border = "none";
                    iframe.style.backgroundColor = "#2a2a2a";

                    // Load viewer from file URL (not inline)
                    const viewerUrl = getViewerUrl(extensionFolder);
                    iframe.src = viewerUrl;
                    console.log(`${logPrefix} Setting iframe src to: ${viewerUrl}`);

                    // Add load event listener
                    iframe.onload = () => {
                        console.log(`${logPrefix} Iframe loaded successfully`);
                    };
                    iframe.onerror = (e) => {
                        console.error(`${logPrefix} Iframe failed to load:`, e);
                    };

                    // Add widget
                    const widget = this.addDOMWidget("preview", "FBX_PREVIEW", iframe, {
                        getValue() { return ""; },
                        setValue(v) { }
                    });

                    console.log(`${logPrefix} Widget created:`, widget);

                    // Set widget size - allow flexible height
                    widget.computeSize = function(width) {
                        const w = width || 512;
                        const h = w * 1.5;  // Taller than wide to accommodate controls
                        return [w, h];
                    };

                    widget.element = iframe;

                    // Store iframe reference
                    this.fbxViewerIframe = iframe;
                    this.fbxViewerReady = false;

                    // Listen for ready message from iframe
                    const onMessage = (event) => {
                    // Provenance guard: the 'message' event is window-wide, so without
                    // this we also handle every OTHER pack's iframe messages.
                    if (event.source !== iframe.contentWindow) return;
                        if (event.data && event.data.type === 'VIEWER_READY') {
                            console.log(`${logPrefix} Viewer iframe is ready!`);
                            this.fbxViewerReady = true;
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

                    // Set initial node size (taller to accommodate controls)
                    this.setSize([512, 768]);

                    // Handle execution
                    const onExecuted = this.onExecuted;
                    this.onExecuted = function(message) {
                        console.log(`${logPrefix} onExecuted called with message:`, message);
                        onExecuted?.apply(this, arguments);

                        // The message contains the FBX file path
                        if (message?.fbx_file && message.fbx_file[0]) {
                            const filename = message.fbx_file[0];
                            console.log(`${logPrefix} Loading FBX: ${filename}`);

                            // Try different path formats based on filename
                            let filepath;
                            let folderType = 'output';  // default

                            // Detect folder type from path
                            if (filename.includes('/input/') || filename.includes('\\input\\')) {
                                folderType = 'input';
                            }

                            // Extract basename if full path provided
                            const basename = filename.includes('/') || filename.includes('\\')
                                ? filename.split(/[/\\]/).pop()
                                : filename;

                            filepath = `${window.location.origin}/view?filename=${encodeURIComponent(basename)}&type=${folderType}&subfolder=`;

                            // Send message to iframe
                            const sendMessage = () => {
                                if (iframe.contentWindow) {
                                    console.log(`${logPrefix} Sending postMessage to iframe: ${filepath}`);
                                    const messageData = {
                                        type: "LOAD_FBX",
                                        filepath: filepath,
                                        timestamp: Date.now()
                                    };
                                    // Include custom FBX export API path if configured
                                    if (fbxExportApiPath) {
                                        messageData.fbxExportApiPath = fbxExportApiPath;
                                    }
                                    iframe.contentWindow.postMessage(messageData, "*");
                                } else {
                                    console.error(`${logPrefix} Iframe contentWindow not available`);
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
                                        console.warn(`${logPrefix} Iframe not ready after 2s, sending anyway`);
                                        sendMessage();
                                    }
                                }, 2000);
                            }
                        } else {
                            console.log(`${logPrefix} No fbx_file in message data. Keys:`, Object.keys(message || {}));
                        }
                    };

                    return r;
                };
            }
        }
    });

    console.log(`${logPrefix} FBX mesh preview extension registered`);
}

// Export detectExtensionFolder for use by other modules
export { detectExtensionFolder, getViewerUrl };

// Self-executing: register all known FBX preview node types
console.log("[FBX Preview] Auto-registering for all known FBX preview node types...");
FBX_PREVIEW_NODES.forEach(config => {
    try {
        createFBXPreviewExtension(config);
    } catch (e) {
        // Extension already registered by another custom node copy - skip
    }
});
