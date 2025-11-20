// Temporary entry file for bundling
import * as THREE from './three.module.js';
import { OrbitControls } from './examples/jsm/controls/OrbitControls.js';
import { TransformControls } from './examples/jsm/controls/TransformControls.js';
import { FBXLoader } from './examples/jsm/loaders/FBXLoader.js';
import { GLTFExporter } from './examples/jsm/exporters/GLTFExporter.js';

// Export to window for global access
window.THREE = THREE;
window.OrbitControls = OrbitControls;
window.TransformControls = TransformControls;
window.FBXLoader = FBXLoader;
window.GLTFExporter = GLTFExporter;
