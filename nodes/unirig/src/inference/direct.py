"""
Direct inference for UniRig models - no subprocess, no Lightning Trainer.

This module provides simple Python functions to run inference directly,
keeping models loaded in memory for fast repeated inference.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from safetensors.torch import load_file as load_safetensors

def _load_checkpoint_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """Load state dict from .safetensors or .ckpt file."""
    if checkpoint_path.endswith('.ckpt'):
        torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
        if torch_version >= (2, 6):
            ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        else:
            ckpt = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            return ckpt['state_dict']
        return ckpt
    return load_safetensors(checkpoint_path)


def sample_mesh_surface(
    vertices: np.ndarray,
    faces: np.ndarray,
    num_samples: int = 2048,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample points uniformly on mesh surface using barycentric coordinates.

    Args:
        vertices: Mesh vertices (V, 3)
        faces: Face indices (F, 3)
        num_samples: Number of points to sample
        seed: Random seed

    Returns:
        Tuple of (sampled_points, sampled_normals) each (num_samples, 3)
    """
    rng = np.random.RandomState(seed)

    # Compute face areas for weighted sampling
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # Cross product gives 2x area
    cross = np.cross(v1 - v0, v2 - v0)
    face_areas = np.linalg.norm(cross, axis=1) / 2

    # Normalize face normals
    face_normals = cross / (np.linalg.norm(cross, axis=1, keepdims=True) + 1e-8)

    # Sample faces weighted by area
    probs = face_areas / face_areas.sum()
    face_indices = rng.choice(len(faces), size=num_samples, p=probs)

    # Sample barycentric coordinates
    r1 = rng.random(num_samples)
    r2 = rng.random(num_samples)

    # Ensure point is inside triangle
    sqrt_r1 = np.sqrt(r1)
    u = 1 - sqrt_r1
    v = sqrt_r1 * (1 - r2)
    w = sqrt_r1 * r2

    # Get sampled points
    sampled_v0 = vertices[faces[face_indices, 0]]
    sampled_v1 = vertices[faces[face_indices, 1]]
    sampled_v2 = vertices[faces[face_indices, 2]]

    points = (
        u[:, None] * sampled_v0 +
        v[:, None] * sampled_v1 +
        w[:, None] * sampled_v2
    )

    # Get normals for sampled faces
    normals = face_normals[face_indices]

    return points.astype(np.float32), normals.astype(np.float32)


def normalize_vertices(vertices: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Normalize vertices to [-1, 1] range.

    Args:
        vertices: Input vertices (N, 3)

    Returns:
        Tuple of (normalized_vertices, normalization_params)
    """
    min_coords = vertices.min(axis=0)
    max_coords = vertices.max(axis=0)
    center = (min_coords + max_coords) / 2
    scale = (max_coords - min_coords).max() / 2

    if scale > 0:
        normalized = (vertices - center) / scale
    else:
        normalized = vertices - center

    params = {
        'center': center,
        'scale': scale,
        'min_coords': min_coords,
        'max_coords': max_coords,
    }

    return normalized.astype(np.float32), params


def _get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def _load_skeleton_model(checkpoint_path: str, device: Optional[torch.device] = None):
    """
    Load the skeleton (AR) model directly.

    Args:
        checkpoint_path: Path to skeleton.safetensors
        device: Device to load model on (default: auto-detect)

    Returns:
        Loaded model ready for inference
    """
    import sys
    import os
    from pathlib import Path

    # Add unirig to path if needed
    unirig_path = Path(__file__).parents[2]
    if str(unirig_path) not in sys.path:
        sys.path.insert(0, str(unirig_path))

    from box import Box
    import yaml

    if device is None:
        device = _get_device()

    # Change to unirig directory so relative config paths resolve correctly
    # (configs reference paths like ./configs/skeleton/vroid.yaml)
    original_cwd = os.getcwd()
    os.chdir(unirig_path)

    try:
        # Load configs
        config_dir = unirig_path / 'configs'

        with open(config_dir / 'model' / 'unirig_ar_350m_1024_81920_float32.yaml') as f:
            model_config = Box(yaml.safe_load(f))

        with open(config_dir / 'tokenizer' / 'tokenizer_parts_articulationxl_256.yaml') as f:
            tokenizer_config = Box(yaml.safe_load(f))

        # Create tokenizer
        from src.tokenizer.parse import get_tokenizer
        from src.tokenizer.spec import TokenizerConfig
        tokenizer = get_tokenizer(config=TokenizerConfig.parse(config=tokenizer_config))

        # Create model
        from src.model.parse import get_model
        model = get_model(tokenizer=tokenizer, **model_config)
    finally:
        # Restore original working directory
        os.chdir(original_cwd)

    # Load weights from safetensors or .ckpt
    print(f"[UniRig Direct] Loading skeleton weights from {checkpoint_path}")
    state_dict = _load_checkpoint_state_dict(checkpoint_path)

    # Remove 'model.' prefix if present
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        new_key = k[6:] if k.startswith('model.') else k
        cleaned_state_dict[new_key] = v

    model.load_state_dict(cleaned_state_dict, strict=True)
    model = model.to(device)
    model.eval()

    print(f"[UniRig Direct] Skeleton model loaded on {device}")
    return model, tokenizer


def _remap_skin_state_dict_flash_to_mha(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Remap skin checkpoint from Flash attention (Wq, Wkv, out_proj) to PyTorch
    MultiheadAttention (in_proj_weight, in_proj_bias, out_proj.weight/bias).

    Checkpoint was saved with Flash MHA; when FLASH_ATTENTION=0 the model uses
    nn.MultiheadAttention, which expects different parameter names and layout.
    """
    import re
    out = {k: v for k, v in state_dict.items() if ".attention.Wq" not in k and ".attention.Wkv" not in k and ".attention.out_proj" not in k}
    # Find all attention block prefixes: e.g. bone_encoder.attn.0.attention, mesh_bone_attn.5.attention
    prefixes = set()
    for k in state_dict:
        m = re.match(r"^(.+\.attention)\.(Wq|Wkv|out_proj)\.", k)
        if m:
            prefixes.add(m.group(1))
    for prefix in prefixes:
        base = prefix + ".attn."
        wq_w = state_dict.get(prefix + ".Wq.weight")
        wq_b = state_dict.get(prefix + ".Wq.bias")
        wkv_w = state_dict.get(prefix + ".Wkv.weight")
        wkv_b = state_dict.get(prefix + ".Wkv.bias")
        out_w = state_dict.get(prefix + ".out_proj.weight")
        out_b = state_dict.get(prefix + ".out_proj.bias")
        if wq_w is None or wkv_w is None or out_w is None:
            continue
        # PyTorch MHA: in_proj = [Q; K; V] stacked; Flash has Wq (Q) and Wkv (K,V)
        in_proj_weight = torch.cat([wq_w, wkv_w], dim=0)
        out[base + "in_proj_weight"] = in_proj_weight
        if wq_b is not None and wkv_b is not None:
            in_proj_bias = torch.cat([wq_b, wkv_b], dim=0)
            out[base + "in_proj_bias"] = in_proj_bias
        out[base + "out_proj.weight"] = out_w
        if out_b is not None:
            out[base + "out_proj.bias"] = out_b
    return out


def _load_skin_model(checkpoint_path: str, device: Optional[torch.device] = None):
    """
    Load the skinning model directly.

    Args:
        checkpoint_path: Path to skin.safetensors
        device: Device to load model on (default: auto-detect)

    Returns:
        Loaded model ready for inference
    """
    import sys
    import os
    from pathlib import Path

    # Add unirig to path if needed
    unirig_path = Path(__file__).parents[2]
    if str(unirig_path) not in sys.path:
        sys.path.insert(0, str(unirig_path))

    from box import Box
    import yaml

    if device is None:
        device = _get_device()

    # Change to unirig directory so relative config paths resolve correctly
    original_cwd = os.getcwd()
    os.chdir(unirig_path)

    try:
        # Load config
        config_dir = unirig_path / 'configs'

        with open(config_dir / 'model' / 'unirig_skin.yaml') as f:
            model_config = Box(yaml.safe_load(f))

        # Create model
        from src.model.parse import get_model
        model = get_model(tokenizer=None, **model_config)
    finally:
        # Restore original working directory
        os.chdir(original_cwd)

    # Load weights from safetensors or .ckpt
    print(f"[UniRig Direct] Loading skin weights from {checkpoint_path}")
    state_dict = _load_checkpoint_state_dict(checkpoint_path)

    # Remove 'model.' prefix if present
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        new_key = k[6:] if k.startswith('model.') else k
        cleaned_state_dict[new_key] = v

    # If model expects PyTorch MHA but checkpoint uses Flash-style keys, remap up front.
    try:
        model_keys = model.state_dict().keys()
        expects_mha = any(".attention.attn.in_proj_weight" in k for k in model_keys)
    except Exception:
        expects_mha = False
    has_flash_keys = any(".attention.Wq." in k or ".attention.Wkv." in k for k in cleaned_state_dict.keys())
    if expects_mha and has_flash_keys:
        print("[UniRig Direct] Detected Flash-style attention keys; remapping to PyTorch MHA format...")
        cleaned_state_dict = _remap_skin_state_dict_flash_to_mha(cleaned_state_dict)

    # Try loading; remap Flash checkpoint -> PyTorch MHA when FLASH_ATTENTION=0
    try:
        model.load_state_dict(cleaned_state_dict, strict=True)
    except RuntimeError as e:
        err_str = str(e)
        # Checkpoint has Flash keys (*.attention.Wq/Wkv/out_proj), model expects PyTorch MHA (*.attention.attn.in_proj_*)
        if "Unexpected key(s)" in err_str and ("attention.Wq" in err_str or "attention.Wkv" in err_str):
            print("[UniRig Direct] Checkpoint uses Flash attention keys; remapping to PyTorch MHA for compatibility...")
            cleaned_state_dict = _remap_skin_state_dict_flash_to_mha(cleaned_state_dict)
            model.load_state_dict(cleaned_state_dict, strict=True)
        elif "attention.attn.Wq" in err_str:
            # Model expects Flash keys but checkpoint has flat .attention.Wq (add .attn)
            print("[UniRig Direct] Remapping attention keys for compatibility...")
            remapped_dict = {}
            for k, v in cleaned_state_dict.items():
                new_key = k
                for suffix in [".Wq.", ".Wkv.", ".out_proj."]:
                    old_pattern = ".attention" + suffix
                    new_pattern = ".attention.attn" + suffix
                    if old_pattern in new_key:
                        new_key = new_key.replace(old_pattern, new_pattern)
                remapped_dict[new_key] = v
            model.load_state_dict(remapped_dict, strict=True)
        else:
            raise

    model = model.to(device)
    model.eval()

    print(f"[UniRig Direct] Skin model loaded on {device}")
    return model


def get_skeleton_model(checkpoint_path: str, device: Optional[torch.device] = None, force_reload: bool = False):
    """Get or load cached skeleton model."""
    cache_key = f"skeleton:{checkpoint_path}"
    if cache_key not in _MODEL_CACHE or force_reload:
        _MODEL_CACHE[cache_key] = _load_skeleton_model(checkpoint_path, device)
    return _MODEL_CACHE[cache_key]


def get_skin_model(checkpoint_path: str, device: Optional[torch.device] = None, force_reload: bool = False):
    """Get or load cached skin model."""
    cache_key = f"skin:{checkpoint_path}"
    if cache_key not in _MODEL_CACHE or force_reload:
        _MODEL_CACHE[cache_key] = _load_skin_model(checkpoint_path, device)
    return _MODEL_CACHE[cache_key]


def clear_model_cache():
    """Clear all cached models to free memory."""
    global _MODEL_CACHE
    _MODEL_CACHE.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.no_grad()
def predict_skeleton(
    vertices: np.ndarray,
    normals: np.ndarray,
    checkpoint_path: str,
    cls: str = "articulationxl",
    max_new_tokens: int = 2048,
    device: Optional[torch.device] = None,
    **generate_kwargs,
) -> Dict[str, np.ndarray]:
    """
    Predict skeleton from mesh vertices and normals.

    Args:
        vertices: Mesh vertices (N, 3), should be normalized to [-1, 1]
        normals: Vertex normals (N, 3)
        checkpoint_path: Path to skeleton.safetensors
        cls: Class name for conditioning (default: "articulationxl")
        max_new_tokens: Maximum tokens to generate
        device: Device to run on
        **generate_kwargs: Additional kwargs for generation (num_beams, etc.)

    Returns:
        Dict with 'joints', 'parents', 'names' etc from detokenization
    """
    if device is None:
        device = _get_device()

    model, tokenizer = get_skeleton_model(checkpoint_path, device)

    # Convert to tensors
    vertices_t = torch.from_numpy(vertices).float().to(device)
    normals_t = torch.from_numpy(normals).float().to(device)

    # Default generation kwargs (matching ar_inference_articulationxl.yaml)
    default_kwargs = {
        'max_new_tokens': max_new_tokens,
        'num_beams': 15,
        'do_sample': True,
        'top_k': 5,
        'top_p': 0.95,
        'repetition_penalty': 3.0,
        'temperature': 1.5,
    }
    default_kwargs.update(generate_kwargs)

    # Run generation
    result = model.generate(
        vertices=vertices_t,
        normals=normals_t,
        cls=cls,
        **default_kwargs,
    )

    # Convert result to dict of numpy arrays
    output = {
        'joints': np.array(result.joints) if result.joints is not None else None,
        'parents': np.array(result.parents) if result.parents is not None else None,
        'names': result.names if hasattr(result, 'names') else None,
    }

    # Add any other attributes from DetokenizeOutput
    for attr in ['tails', 'bone_names', 'root_idx']:
        if hasattr(result, attr):
            val = getattr(result, attr)
            if val is not None:
                output[attr] = np.array(val) if isinstance(val, (list, tuple)) else val

    return output


@torch.no_grad()
def predict_skinning(
    vertices: np.ndarray,
    normals: np.ndarray,
    joints: np.ndarray,
    parents: np.ndarray,
    checkpoint_path: str,
    faces: Optional[np.ndarray] = None,
    tails: Optional[np.ndarray] = None,
    voxel_grid_size: int = 196,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """
    Predict skinning weights from mesh and skeleton.

    Args:
        vertices: Mesh vertices (N, 3)
        normals: Vertex normals (N, 3)
        joints: Joint positions (J, 3)
        parents: Parent indices (J,)
        checkpoint_path: Path to skin.safetensors
        faces: Mesh faces (F, 3) - required for voxel_skin computation
        tails: Bone tail positions (J, 3) - if None, computed from joints
        voxel_grid_size: Grid size for voxel_skin (default 196)
        device: Device to run on

    Returns:
        Skinning weights (N, J)
    """
    if device is None:
        device = _get_device()

    model = get_skin_model(checkpoint_path, device)

    num_joints = len(joints)
    num_vertices = len(vertices)

    # Compute tails if not provided (use joint positions as tails)
    if tails is None:
        # Default: tails = joints (bone has zero length, point bone)
        tails = joints.copy()

    # Compute voxel_skin if faces provided
    if faces is not None:
        # Import voxelization functions
        import sys
        from pathlib import Path
        unirig_path = Path(__file__).parents[2]
        if str(unirig_path) not in sys.path:
            sys.path.insert(0, str(unirig_path))

        from src.data.vertex_group import voxelization, voxel_skin

        print(f"[UniRig Direct] Computing voxel_skin (grid={voxel_grid_size})...")

        # Voxelize the mesh
        grid_coords = voxelization(
            vertices=vertices,
            faces=faces,
            grid=voxel_grid_size,
            backend='trimesh',
        )

        # Compute voxel skin weights
        voxel_skin_weights = voxel_skin(
            grid=voxel_grid_size,
            grid_coords=grid_coords,
            joints=joints,
            vertices=vertices,
            faces=faces,
            alpha=0.5,
            link_dis=0.00001,
            grid_query=7,
            vertex_query=1,
            grid_weight=3.0,
        )
        voxel_skin_weights = np.nan_to_num(voxel_skin_weights, nan=0., posinf=0., neginf=0.)
        print(f"[UniRig Direct] voxel_skin shape: {voxel_skin_weights.shape}")
    else:
        # Fallback: zero voxel_skin
        voxel_skin_weights = np.zeros((num_joints, num_vertices), dtype=np.float32)

    # Prepare batch
    # Note: offset is for PTv3 sparse convolutions - indicates where each batch element ends
    # For a single batch with N vertices, offset = [N]
    batch = {
        'vertices': torch.from_numpy(vertices).float().unsqueeze(0).to(device),
        'normals': torch.from_numpy(normals).float().unsqueeze(0).to(device),
        'joints': torch.from_numpy(joints).float().unsqueeze(0).to(device),
        'tails': torch.from_numpy(tails).float().unsqueeze(0).to(device),
        'voxel_skin': torch.from_numpy(voxel_skin_weights).float().unsqueeze(0).to(device),
        'parents': torch.from_numpy(parents).long().unsqueeze(0).to(device),
        'num_bones': torch.tensor([num_joints], dtype=torch.long, device=device),
        'offset': torch.tensor([num_vertices], dtype=torch.long, device=device),  # PTv3 batch offset for single sample
        'path': ['direct_inference'],
    }

    # Run prediction
    result = model.predict_step(batch)

    # Extract skin weights - result is a list of tensors, one per batch
    skin_weights = result[0]  # Get first (only) batch item
    if isinstance(skin_weights, torch.Tensor):
        skin_weights = skin_weights.cpu().numpy()

    return skin_weights


@torch.no_grad()
def predict_skeleton_from_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    skeleton_checkpoint: str,
    num_samples: int = 2048,
    cls: str = "articulationxl",
    max_new_tokens: int = 2048,
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    High-level function: predict skeleton from raw mesh.

    Handles all preprocessing (normalization, sampling) internally.

    Args:
        vertices: Raw mesh vertices (V, 3)
        faces: Face indices (F, 3)
        skeleton_checkpoint: Path to skeleton.safetensors
        num_samples: Number of surface points to sample
        cls: Class name for conditioning
        max_new_tokens: Maximum tokens to generate
        seed: Random seed for sampling
        device: Device to run on

    Returns:
        Tuple of (skeleton_dict, normalization_params)
        - skeleton_dict: Dict with 'joints', 'parents', 'names', etc.
        - normalization_params: Dict with 'center', 'scale' for denormalization
    """
    # 1. Normalize mesh vertices
    norm_vertices, norm_params = normalize_vertices(vertices)

    # 2. Sample surface points
    sampled_points, sampled_normals = sample_mesh_surface(
        norm_vertices,
        faces,
        num_samples=num_samples,
        seed=seed,
    )

    # 3. Run skeleton prediction
    skeleton = predict_skeleton(
        vertices=sampled_points,
        normals=sampled_normals,
        checkpoint_path=skeleton_checkpoint,
        cls=cls,
        max_new_tokens=max_new_tokens,
        device=device,
    )

    return skeleton, norm_params


@torch.no_grad()
def rig_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    skeleton_checkpoint: str,
    skin_checkpoint: str,
    num_samples: int = 2048,
    cls: str = "articulationxl",
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Complete rigging pipeline: mesh -> skeleton -> skinning weights.

    Args:
        vertices: Raw mesh vertices (V, 3)
        faces: Face indices (F, 3)
        skeleton_checkpoint: Path to skeleton.safetensors
        skin_checkpoint: Path to skin.safetensors
        num_samples: Number of surface points to sample
        cls: Class name for conditioning
        seed: Random seed
        device: Device to run on

    Returns:
        Dict with:
            - 'joints': Joint positions (J, 3) in normalized space
            - 'parents': Parent indices (J,)
            - 'names': Joint names
            - 'skin_weights': Skinning weights (num_samples, J)
            - 'sampled_vertices': Sampled surface points (num_samples, 3)
            - 'sampled_normals': Sampled normals (num_samples, 3)
            - 'normalization': Dict with center/scale for denormalization
    """
    # 1. Normalize mesh vertices
    norm_vertices, norm_params = normalize_vertices(vertices)

    # 2. Sample surface points
    sampled_points, sampled_normals = sample_mesh_surface(
        norm_vertices,
        faces,
        num_samples=num_samples,
        seed=seed,
    )

    # 3. Predict skeleton
    print("[UniRig Direct] Predicting skeleton...")
    skeleton = predict_skeleton(
        vertices=sampled_points,
        normals=sampled_normals,
        checkpoint_path=skeleton_checkpoint,
        cls=cls,
        device=device,
    )

    if skeleton['joints'] is None:
        raise RuntimeError("Skeleton prediction failed - no joints generated")

    # 4. Predict skinning weights
    print("[UniRig Direct] Predicting skinning weights...")
    skin_weights = predict_skinning(
        vertices=sampled_points,
        normals=sampled_normals,
        joints=skeleton['joints'],
        parents=skeleton['parents'],
        checkpoint_path=skin_checkpoint,
        device=device,
    )

    return {
        'joints': skeleton['joints'],
        'parents': skeleton['parents'],
        'names': skeleton.get('names'),
        'tails': skeleton.get('tails'),
        'skin_weights': skin_weights,
        'sampled_vertices': sampled_points,
        'sampled_normals': sampled_normals,
        'normalization': norm_params,
    }
