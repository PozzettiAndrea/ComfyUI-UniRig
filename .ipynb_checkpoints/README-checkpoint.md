# ComfyUI-UniRig

State-of-the-art automatic skeleton extraction and rigging for ComfyUI, powered by UniRig (SIGGRAPH 2025).

## Features

- **High-quality skeleton extraction** using ML-based approach
- **Works on any mesh type**: humans, animals, objects, cameras, furniture, etc.
- **Multiple extraction modes**:
  - Skeleton-only extraction (fast, for pose control)
  - Full rig extraction (skeleton + skinning weights for animation)
- **Completely local** - no API calls or cloud services

## What is UniRig?

UniRig is a unified framework for automatic 3D model rigging developed by Tsinghua University and Tripo. It uses large autoregressive models to predict topologically valid skeleton hierarchies and skinning weights.

**Paper**: [One Model to Rig Them All](https://zjp-shadow.github.io/works/UniRig/)
**Repository**: [VAST-AI-Research/UniRig](https://github.com/VAST-AI-Research/UniRig)

## Installation

### 1. Clone UniRig

```bash
cd /workspace
git clone https://github.com/VAST-AI-Research/UniRig.git
```

### 2. Create Separate Python 3.11 Environment for UniRig

**Important**: UniRig requires Python 3.11 (for `bpy==4.2.0`), but ComfyUI runs on Python 3.10.
We handle this by running UniRig in a separate conda environment via subprocess.

```bash
# Create Python 3.11 environment for UniRig
conda create -n unirig python=3.11 -y
conda activate unirig

cd /workspace/UniRig

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install base requirements
pip install -r requirements.txt

# Install CUDA-specific packages (IMPORTANT: match your CUDA version!)
# For CUDA 11.8:
pip install spconv-cu118

# For CUDA 12.1:
# pip install spconv-cu121

# For PyG packages (adjust torch and CUDA versions):
pip install torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.3.1+cu118.html --no-cache-dir

# Downgrade numpy for compatibility
pip install numpy==1.26.4

# Optional but recommended (may require manual compilation):
pip install flash-attn --no-build-isolation
```

**How it works**: The ComfyUI-UniRig node uses `conda run -n unirig` to execute UniRig scripts in the Python 3.11 environment, while ComfyUI itself continues running in Python 3.10. Communication happens via temp files (GLB in, FBX out).

### 3. Download Model Checkpoint

The model will be automatically downloaded from HuggingFace on first use. Alternatively, download manually:

```bash
# Model is hosted at: https://huggingface.co/VAST-AI/UniRig
```

## Usage

### Node: UniRig: Extract Skeleton

Extracts skeletal structure from any 3D mesh.

**Inputs**:
- `trimesh`: Input 3D mesh
- `seed`: Random seed for variation (try different values for different skeleton topologies)
- `checkpoint`: HuggingFace model ID (default: "VAST-AI/UniRig")

**Outputs**:
- `skeleton`: SKELETON format (joints + bones)

**Output format**: Normalized to [-1, 1] range, compatible with Hunyuan3D-Omni pose control.

### Node: UniRig: Extract Full Rig

Extracts complete rig with skeleton and skinning weights for animation.

**Inputs**:
- `trimesh`: Input 3D mesh
- `seed`: Random seed

**Outputs**:
- `rigged_mesh`: TRIMESH with armature and skinning weights

## Workflow Examples

### Skeleton Extraction for Pose Control

```
LoadMesh → UniRig: Extract Skeleton → SkeletonToPose → Hy3DOmniGenerateLatents
```

### Full Rigging for Animation

```
LoadMesh → UniRig: Extract Full Rig → SaveGLB
```

## Comparison with Other Methods

| Method | Quality | Speed | Semantic | Works on All Shapes |
|--------|---------|-------|----------|-------------------|
| **UniRig** | Excellent | Slow | Yes | Yes |
| Skeletor | Good | Fast | No | Yes |
| Pinocchio | Good | Fast | Partial | Humanoid only |
| RigNet | Good | Medium | Yes | Humanoid-focused |

## Tips

- **Seed variation**: Try different seed values (0-1000) to get different skeleton topologies
- **Speed**: Skeleton extraction takes ~30-60 seconds on GPU, skinning adds ~30-60 seconds more
- **Quality**: UniRig produces semantic skeletons that understand anatomy, unlike geometric methods
- **Use cases**:
  - Skeleton-only → For pose control in Hunyuan3D-Omni generation
  - Full rig → For animation export to Blender/Unity/Unreal

## Troubleshooting

### "UniRig not found" error
Make sure UniRig is cloned to `/workspace/UniRig`. Adjust the `UNIRIG_PATH` in `unirig_nodes.py` if needed.

### CUDA errors during installation
UniRig requires CUDA. Make sure to install `spconv`, `torch_scatter`, and `torch_cluster` with the correct CUDA version.

### "flash_attn" installation fails
flash_attn is optional but improves performance. If installation fails, you can skip it.

### Inference fails or times out
- Check GPU memory (needs ~8GB VRAM)
- Check that all dependencies are installed correctly
- Try a simpler mesh first to verify installation

## Requirements

- **GPU**: CUDA-enabled GPU with 8GB+ VRAM
- **ComfyUI**: Python 3.10 environment
- **UniRig**: Separate Python 3.11 conda environment (named `unirig`)
- **PyTorch**: >= 2.3.1
- **CUDA**: 11.8 or 12.1 recommended
- **conda**: For managing separate environments

**Architecture**: ComfyUI (Python 3.10) → subprocess → UniRig (Python 3.11)

## License

This wrapper: MIT License
UniRig: MIT License (see [UniRig repository](https://github.com/VAST-AI-Research/UniRig))

## Credits

- **UniRig** by VAST-AI Research and Tsinghua University
- **Paper**: Zhang et al., "One Model to Rig Them All: Diverse Skeleton Rigging with UniRig" (SIGGRAPH 2025)

## Links

- [UniRig Project Page](https://zjp-shadow.github.io/works/UniRig/)
- [UniRig GitHub](https://github.com/VAST-AI-Research/UniRig)
- [UniRig Model (HuggingFace)](https://huggingface.co/VAST-AI/UniRig)
