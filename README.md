WORK IN PROGRESS

# ComfyUI-UniRig

Automatic skeleton extraction for ComfyUI using UniRig (SIGGRAPH 2025). Self-contained with bundled Blender.

## Installation

1. Clone into ComfyUI custom nodes
2. Run `bash install_unirig.sh` to set up the unirig conda environment
3. Blender auto-installs on first use

## Usage

**UniRig: Extract Skeleton** - Extracts skeleton from any 3D mesh
- Input: TRIMESH mesh
- Output: SKELETON (joints + bones, normalized to [-1,1])

## Requirements

- CUDA GPU (8GB+ VRAM)
- Conda (for unirig environment)

## Links

- [UniRig Paper](https://zjp-shadow.github.io/works/UniRig/)
- [UniRig GitHub](https://github.com/VAST-AI-Research/UniRig)
