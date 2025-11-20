#!/usr/bin/env node
// Build script to create a self-contained viewer HTML with inlined JavaScript

const fs = require('fs');
const path = require('path');

const viewerHtmlPath = path.join(__dirname, 'web/viewer_fbx.html');
const bundlePath = path.join(__dirname, 'static/three/viewer-bundle.js');
const outputPath = path.join(__dirname, 'web/js/viewer_inline.js');

console.log('Reading viewer HTML...');
let html = fs.readFileSync(viewerHtmlPath, 'utf-8');

console.log('Reading bundle JavaScript...');
const bundleJs = fs.readFileSync(bundlePath, 'utf-8');

console.log('Bundle size:', (bundleJs.length / 1024 / 1024).toFixed(2), 'MB');

// Replace the script src with inline script
html = html.replace(
    /<script src="\/extensions\/ComfyUI-UniRig\/static\/three\/viewer-bundle\.js"><\/script>/,
    `<script>${bundleJs}</script>`
);

// Escape backticks and dollar signs for template literal
const escapedHtml = html
    .replace(/\\/g, '\\\\')
    .replace(/`/g, '\\`')
    .replace(/\${/g, '\\${');

// Create JavaScript module that exports the HTML
const jsModule = `// Auto-generated inline viewer - DO NOT EDIT
// Generated on ${new Date().toISOString()}

export const VIEWER_HTML = \`${escapedHtml}\`;
`;

console.log('Writing output to:', outputPath);
fs.writeFileSync(outputPath, jsModule, 'utf-8');

console.log('Done! Inline viewer created successfully.');
console.log('Output size:', (jsModule.length / 1024 / 1024).toFixed(2), 'MB');
