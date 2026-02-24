# ComfyUI-UniRig

<div align="center">
<a href="https://pozzettiandrea.github.io/ComfyUI-UniRig/">
<img src="https://pozzettiandrea.github.io/ComfyUI-UniRig/gallery-preview.png" alt="Workflow Test Gallery" width="800">
</a>
<br>
<b><a href="https://pozzettiandrea.github.io/ComfyUI-UniRig/">View Live Test Gallery →</a></b>
</div>

## About UniRig

This node uses [UniRig](https://zjp-shadow.github.io/works/UniRig/) (SIGGRAPH'25 / ACM TOG), a unified framework for automatic 3D rigging: it predicts a topologically valid skeleton and per-vertex skinning weights from mesh geometry in one pipeline. In ComfyUI, **Load Mesh → UniRig Load Model → UniRig Auto Rig** produces a single rigged FBX—equivalent to the upstream CLI steps `generate_skeleton` → `generate_skin` → `merge`. No separate merge step is required.

- **Project page:** [zjp-shadow.github.io/works/UniRig](https://zjp-shadow.github.io/works/UniRig/)
- **Paper (arXiv):** [arxiv.org/abs/2504.12451](https://arxiv.org/abs/2504.12451)
- **Models:** [Hugging Face — VAST-AI/UniRig](https://huggingface.co/VAST-AI/UniRig)

**Model sources (Load Model nodes):** By default, weights are downloaded from **apozz/UniRig-safetensors** (safetensors format). You can instead choose **VAST-AI/UniRig** in the Load Model / Load Skeleton Model / Load Skinning Model dropdown to use the official Hugging Face `.ckpt` checkpoints.

**Key features (from UniRig):**
- Unified model for diverse 3D assets (humans, animals, objects)
- Automated skeleton generation and skinning prediction
- Skeleton Tree Tokenization and Bone–Point Cross Attention

**Supported input formats:** Load Mesh supports `.obj`, `.ply`, `.stl`, `.glb`, `.gltf`, `.fbx`. For **.vrm**, the optional Blender VRM addon must be installed—see [nodes/unirig/README.md](nodes/unirig/README.md) (Installation, step 6).

**System requirements:** CUDA-enabled GPU with at least 8GB VRAM for generation.

Automatic skeleton extraction for ComfyUI using UniRig (SIGGRAPH 2025) or Make it Animatable (CVPR 2025).
Self-contained with bundled Blender and UniRig/MIA code.

It is recommended to use MIA for humanoid characters.

Rig your character mesh and skin it!
![rigging_and_skinning](docs/rigging_and_skinning.png)

Change their pose, export a new one
![rigging_manipulation](docs/rigging_manipulation.png)

## Video demos

Rigging/skinning workflow (video is sped up for documentation purposes):


https://github.com/user-attachments/assets/6d06a3cd-db63-4e3a-b13b-78ff7868a162


Manipulation/saving/export:


https://github.com/user-attachments/assets/f320db66-4323-4993-a46e-87e2717748ef

## Installation

### Via ComfyUI Manager (Recommended)
1. Open ComfyUI Manager
2. Search for "UniRig"
3. Click Install
4. Restart ComfyUI

## Citation

If you use UniRig in your research, please cite:

```bibtex
@article{10.1145/3730930,
  author = {Zhang, Jia-Peng and Pu, Cheng-Feng and Guo, Meng-Hao and Cao, Yan-Pei and Hu, Shi-Min},
  title = {One Model to Rig Them All: Diverse Skeleton Rigging with UniRig},
  year = {2025},
  issue_date = {August 2025},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  volume = {44},
  number = {4},
  issn = {0730-0301},
  url = {https://doi.org/10.1145/3730930},
  doi = {10.1145/3730930},
  journal = {ACM Trans. Graph.},
  month = jul,
  articleno = {123},
  numpages = {18}
}
```

## Community

Questions or feature requests? Open a [Discussion](https://github.com/PozzettiAndrea/ComfyUI-UniRig/discussions) on GitHub.

Join the [Comfy3D Discord](https://discord.gg/bcdQCUjnHE) for help, updates, and chat about 3D workflows in ComfyUI.

## Credits

- Based on [UniRig](https://github.com/VAST-AI-Research/UniRig) by VAST-AI-Research, Tsinghua University, and Tripo.
- [UniRig Project Page](https://zjp-shadow.github.io/works/UniRig/)
- [UniRig GitHub](https://github.com/VAST-AI-Research/UniRig)

For full upstream content (training, dataset, CLI usage), see [nodes/unirig/README.md](nodes/unirig/README.md).
