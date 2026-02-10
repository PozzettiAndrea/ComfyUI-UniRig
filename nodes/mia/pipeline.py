"""
Make-It-Animatable inference pipeline.

Standalone implementation without Gradio dependencies.
Adapted from the original MIA app.py for use in ComfyUI.
"""

import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
import trimesh

from .utils import sample_mesh
from .dataset_mixamo import MIXAMO_PREFIX
from . import BONES_IDX_DICT, KINEMATIC_TREE


@dataclass
class InferenceData:
    """Data container for inference pipeline."""
    mesh: trimesh.Trimesh = None
    is_mesh: bool = True
    verts: torch.Tensor = None
    verts_normal: torch.Tensor = None
    faces: np.ndarray = None
    pts: torch.Tensor = None
    pts_normal: torch.Tensor = None
    gs: torch.Tensor = None
    sample_mask: np.ndarray = None
    global_transform: Any = None
    bw: torch.Tensor = None
    joints: torch.Tensor = None
    pose: torch.Tensor = None
    output_dir: str = None
    original_visual: Any = None  # Preserve original mesh visual (textures/materials) for export


def get_conflict_mask(dominant_idx, match_fn, conflict_fn, bones_idx_dict):
    """Get mask for conflicting bone weights."""
    match_indices = [i for k, i in bones_idx_dict.items() if match_fn(k)]
    conflict_indices = [i for k, i in bones_idx_dict.items() if conflict_fn(k)]
    match_mask = sum(dominant_idx == i for i in match_indices).bool()
    conflict_mask = sum(dominant_idx == i for i in conflict_indices).bool()
    return match_mask & conflict_mask


def get_hips_transform(hips, rightupleg, leftupleg):
    """Compute transform to align hips coordinate system."""
    # Get the up vector (Y axis) - from hips toward head
    up = torch.tensor([0.0, 1.0, 0.0], device=hips.device, dtype=hips.dtype)

    # Get the right vector from left to right leg
    right = rightupleg - leftupleg
    right = right / (torch.norm(right, dim=-1, keepdim=True) + 1e-8)

    # Compute forward as cross product
    forward = torch.cross(up.expand_as(right), right, dim=-1)
    forward = forward / (torch.norm(forward, dim=-1, keepdim=True) + 1e-8)

    # Recompute right to ensure orthogonality
    right = torch.cross(up.expand_as(forward), forward, dim=-1)

    # Build rotation matrix
    rotation = torch.stack([right, up.expand_as(right), forward], dim=-1)

    # Add translation
    transform = torch.eye(4, device=hips.device, dtype=hips.dtype).unsqueeze(0).expand(hips.shape[0], -1, -1).clone()
    transform[..., :3, :3] = rotation

    return transform


class SimpleTransform:
    """Simple transform class to replace pytorch3d Transform3d."""

    def __init__(self, matrix: torch.Tensor = None, scale: float = 1.0, translate: torch.Tensor = None):
        if matrix is not None:
            self.matrix = matrix  # (B, 4, 4)
        else:
            self.matrix = torch.eye(4).unsqueeze(0)
            if scale != 1.0:
                self.matrix[..., :3, :3] *= scale
            if translate is not None:
                self.matrix[..., :3, 3] = translate

    def transform_points(self, points: torch.Tensor) -> torch.Tensor:
        """Transform points: (B, N, 3) -> (B, N, 3)"""
        # Add homogeneous coordinate
        ones = torch.ones(*points.shape[:-1], 1, device=points.device, dtype=points.dtype)
        points_h = torch.cat([points, ones], dim=-1)  # (B, N, 4)

        # Apply transform
        transformed = torch.einsum('bij,bnj->bni', self.matrix, points_h)

        return transformed[..., :3]

    def transform_normals(self, normals: torch.Tensor) -> torch.Tensor:
        """Transform normals using inverse transpose of rotation."""
        rot = self.matrix[..., :3, :3]
        # For normals, use inverse transpose (for uniform scale, just use rotation)
        return torch.einsum('bij,bnj->bni', rot, normals)

    def compose(self, other: 'SimpleTransform') -> 'SimpleTransform':
        """Compose with another transform."""
        new_matrix = torch.bmm(other.matrix, self.matrix)
        result = SimpleTransform()
        result.matrix = new_matrix
        return result

    def inverse(self) -> 'SimpleTransform':
        """Compute inverse transform."""
        result = SimpleTransform()
        result.matrix = torch.inverse(self.matrix)
        return result


def get_normalize_transform(points: torch.Tensor, keep_ratio: bool = True, recenter: bool = True) -> SimpleTransform:
    """
    Create a normalization transform that scales points to [-1, 1] range.

    Args:
        points: Input points (B, N, 3)
        keep_ratio: If True, use uniform scaling
        recenter: If True, center the points

    Returns:
        SimpleTransform that normalizes the points
    """
    B = points.shape[0]
    device = points.device
    dtype = points.dtype

    min_vals = torch.min(points, dim=1, keepdim=True)[0]  # (B, 1, 3)
    max_vals = torch.max(points, dim=1, keepdim=True)[0]  # (B, 1, 3)

    if keep_ratio:
        scale = 2.0 / torch.max(max_vals - min_vals, dim=2, keepdim=True)[0]  # (B, 1, 1)
        scale = scale.squeeze(-1).squeeze(-1)  # (B,)
    else:
        scale = 2.0 / (max_vals - min_vals).squeeze(1)  # (B, 3)

    if recenter:
        center = ((max_vals + min_vals) / 2.0).squeeze(1)  # (B, 3)
    else:
        center = torch.zeros(B, 3, device=device, dtype=dtype)

    # Build transform matrix: first translate by -center, then scale
    matrix = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1).clone()

    if keep_ratio:
        # Uniform scale
        matrix[:, 0, 0] = scale
        matrix[:, 1, 1] = scale
        matrix[:, 2, 2] = scale
        matrix[:, :3, 3] = -center * scale.unsqueeze(-1)
    else:
        # Non-uniform scale
        matrix[:, 0, 0] = scale[:, 0]
        matrix[:, 1, 1] = scale[:, 1]
        matrix[:, 2, 2] = scale[:, 2]
        matrix[:, :3, 3] = -center * scale

    return SimpleTransform(matrix=matrix)


def prepare_input(
    mesh: trimesh.Trimesh,
    N: int = 32768,
    hands_resample_ratio: float = 0.5,
    geo_resample_ratio: float = 0.0,
    get_normals: bool = True,
) -> InferenceData:
    """
    Prepare mesh for inference.

    Args:
        mesh: Input trimesh object
        N: Number of points to sample
        hands_resample_ratio: Ratio for hand region oversampling
        geo_resample_ratio: Ratio for geometric feature sampling
        get_normals: Whether to compute normals

    Returns:
        InferenceData object with prepared data
    """
    data = InferenceData()
    data.mesh = mesh
    # Store original visual before any vertex mutations destroy it
    data.original_visual = mesh.visual if hasattr(mesh, 'visual') else None

    verts = np.array(mesh.vertices).astype(np.float32)
    faces = np.array(mesh.faces) if hasattr(mesh, 'faces') else None
    data.faces = faces
    data.is_mesh = faces is not None

    if data.is_mesh:
        verts_normal = np.array(mesh.vertex_normals).astype(np.float32)
        pts = sample_mesh(mesh, N, get_normals=True).astype(np.float32)
        pts, pts_normal = np.split(pts, 2, axis=-1)
    else:
        verts_normal = None
        pts = sample_mesh(mesh, N, get_normals=False).astype(np.float32)
        pts_normal = None

    data.verts = torch.from_numpy(verts).unsqueeze(0)
    data.verts_normal = torch.from_numpy(verts_normal).unsqueeze(0) if verts_normal is not None else None
    data.pts = torch.from_numpy(pts).unsqueeze(0)
    data.pts_normal = torch.from_numpy(pts_normal).unsqueeze(0) if pts_normal is not None else None

    return data


def preprocess(
    data: InferenceData,
    model_coarse: torch.nn.Module,
    device: torch.device,
    hands_resample_ratio: float = 0.5,
    geo_resample_ratio: float = 0.0,
    N: int = 32768,
) -> InferenceData:
    """
    Preprocess data: normalize and run coarse joint localization.

    Args:
        data: InferenceData from prepare_input
        model_coarse: Coarse joint prediction model
        device: Torch device
        hands_resample_ratio: Ratio for hand oversampling
        geo_resample_ratio: Ratio for geometric sampling
        N: Number of sample points

    Returns:
        Updated InferenceData
    """
    pts = data.pts
    verts = data.verts
    pts_normal = data.pts_normal
    verts_normal = data.verts_normal

    # Normalize to unit cube centered at origin
    norm = get_normalize_transform(pts, keep_ratio=True, recenter=True)
    pts = norm.transform_points(pts)
    verts = norm.transform_points(verts)

    # Run coarse joint localization
    with torch.no_grad():
        pts_device = pts.to(device)
        joints_out = model_coarse(pts_device).joints.cpu()

    joints, joints_tail = joints_out[..., :3], joints_out[..., 3:]

    # Get key joint positions for alignment
    hips = joints[:, BONES_IDX_DICT[f"{MIXAMO_PREFIX}Hips"]]
    rightupleg = joints[:, BONES_IDX_DICT[f"{MIXAMO_PREFIX}RightUpLeg"]]
    leftupleg = joints[:, BONES_IDX_DICT[f"{MIXAMO_PREFIX}LeftUpLeg"]]

    # Compute rotation to align with standard orientation
    rotate_matrix = get_hips_transform(hips, rightupleg, leftupleg).transpose(-1, -2)
    rotate = SimpleTransform(matrix=rotate_matrix)

    # Compose transforms
    global_transform = norm.compose(rotate)

    # Apply rotation
    pts = rotate.transform_points(pts)
    verts = rotate.transform_points(verts)

    if data.is_mesh and pts_normal is not None:
        pts_normal = F.normalize(rotate.transform_normals(pts_normal), dim=-1)
        verts_normal = F.normalize(rotate.transform_normals(verts_normal), dim=-1)

    # Update mesh vertices for resampling
    data.mesh.vertices = verts.squeeze(0).numpy()

    # Resample with hand attention if requested
    if hands_resample_ratio > 0:
        joints_tail_aligned = rotate.transform_points(joints_tail).squeeze(0).numpy()
        hands_centers = [
            joints_tail_aligned[BONES_IDX_DICT[f"{MIXAMO_PREFIX}LeftHand"]],
            joints_tail_aligned[BONES_IDX_DICT[f"{MIXAMO_PREFIX}RightHand"]],
        ]
        pts_np = sample_mesh(
            data.mesh,
            N,
            get_normals=data.is_mesh,
            attn_ratio=hands_resample_ratio,
            attn_centers=hands_centers,
            attn_geo_ratio=geo_resample_ratio,
        ).astype(np.float32)

        pts = torch.from_numpy(pts_np).unsqueeze(0)
        if data.is_mesh:
            pts, pts_normal = torch.chunk(pts, 2, dim=-1)

    data.verts = verts
    data.verts_normal = verts_normal
    data.pts = pts
    data.pts_normal = pts_normal
    data.global_transform = global_transform

    return data


_BW_CHUNK_SIZE = 32768


def _chunked_bw(model, pts_device, verts_device, chunk_size=_BW_CHUNK_SIZE):
    """Run bone weight model in chunks over vertices to avoid OOM on large meshes."""
    n_verts = verts_device.shape[1]
    if n_verts <= chunk_size:
        return model(pts_device, verts_device).bw
    chunks = []
    for i in range(0, n_verts, chunk_size):
        chunk = verts_device[:, i:i + chunk_size]
        chunks.append(model(pts_device, chunk).bw)
    return torch.cat(chunks, dim=1)


def infer(
    data: InferenceData,
    model_bw: torch.nn.Module,
    model_bw_normal: torch.nn.Module,
    model_joints: torch.nn.Module,
    model_pose: torch.nn.Module,
    device: torch.device,
    use_normal: bool = False,
) -> InferenceData:
    """
    Run main inference: predict blend weights, joints, and pose.

    Args:
        data: Preprocessed InferenceData
        model_bw: Blend weights model
        model_bw_normal: Blend weights with normals model
        model_joints: Joint prediction model
        model_pose: Pose prediction model
        device: Torch device
        use_normal: Whether to use normal-aware blend weights

    Returns:
        Updated InferenceData with predictions
    """
    pts = data.pts
    pts_normal = data.pts_normal
    verts = data.verts
    verts_normal = data.verts_normal

    if use_normal and not data.is_mesh:
        raise ValueError("Normals are not available for point clouds")

    # Normalize for inference
    norm = get_normalize_transform(pts, keep_ratio=True, recenter=False)
    pts = norm.transform_points(pts)
    verts = norm.transform_points(verts)

    with torch.no_grad():
        # Blend weights inference
        pts_device = pts.to(device)
        verts_device = verts.to(device)

        bw = _chunked_bw(model_bw, pts_device, verts_device)

        if use_normal and pts_normal is not None:
            pts_normal_device = pts_normal.to(device)
            verts_normal_device = verts_normal.to(device) if verts_normal is not None else None

            pts_with_normal = torch.cat([pts_device, pts_normal_device], dim=-1)
            verts_with_normal = (
                torch.cat([verts_device, verts_normal_device], dim=-1)
                if verts_normal_device is not None
                else verts_device
            )
            bw_normal = _chunked_bw(model_bw_normal, pts_with_normal, verts_with_normal)

            # Blend normal weights for spine/shoulder/arm regions
            mask = get_conflict_mask(
                torch.argmax(bw, dim=-1),
                lambda k: True,
                lambda k: any(x in k for x in ("Spine", "Shoulder", "Arm")),
                BONES_IDX_DICT,
            )
            bw_normal[mask] = bw[mask]
            bw = bw_normal

        bw = bw.cpu()

        # Joints and pose inference
        joints_out = model_joints(pts_device).joints.cpu()

        # Prepare joints for pose model
        joints_for_pose = joints_out.clone().to(device)
        pose = model_pose(pts_device, joints=joints_for_pose).pose_trans.cpu()

    data.mesh.vertices = verts.squeeze(0).cpu().numpy()
    data.pts = pts
    data.verts = verts
    data.bw = bw
    data.joints = joints_out
    data.pose = pose
    data.global_transform = data.global_transform.compose(norm)

    return data


def bw_post_process(
    bw: torch.Tensor,
    bones_idx_dict: Dict[str, int],
    above_head_mask: torch.Tensor = None,
    no_fingers: bool = False,
) -> torch.Tensor:
    """
    Post-process blend weights.

    Args:
        bw: Blend weights tensor (B, N, num_bones)
        bones_idx_dict: Bone name to index mapping
        above_head_mask: Mask for vertices above head
        no_fingers: Whether to merge finger weights to hand

    Returns:
        Processed blend weights
    """
    def _edit_bw(mask, bw, bones_idx_dict, target_bone_name, value=1e5):
        mask = mask.unsqueeze(-1).tile((bw.shape[-1],))
        head_bone_mask = torch.zeros_like(mask)
        head_bone_mask[..., bones_idx_dict[f"{MIXAMO_PREFIX}{target_bone_name}"]] = True
        mask = mask & head_bone_mask
        bw[mask] = value
        return bw

    assert bw.shape[-1] == len(bones_idx_dict)

    if above_head_mask is not None and all("Ear" not in b for b in bones_idx_dict):
        bw = _edit_bw(above_head_mask, bw, bones_idx_dict, "Head")

    hands = {"Left", "Right"}
    fingers = {"Thumb", "Index", "Middle", "Ring", "Pinky"}

    if no_fingers:
        dominant_idx = torch.argmax(bw, dim=-1)
        for hand in hands:
            bw[
                get_conflict_mask(
                    dominant_idx,
                    lambda k, h=hand: h in k and any(x in k for x in fingers),
                    lambda k, h=hand: k.endswith(f"{h}Hand"),
                    bones_idx_dict,
                )
            ] = 1e5
        bw[get_conflict_mask(dominant_idx, lambda k: True, lambda k: any(x in k for x in fingers), bones_idx_dict)] = 0

    # Refine conflicting left/right limbs
    dominant_idx = torch.argmax(bw, dim=-1)
    if not no_fingers:
        for hand in hands:
            other_hand = next(iter(hands - {hand}))
            bw[get_conflict_mask(dominant_idx, lambda k, h=hand: h in k, lambda k, oh=other_hand: oh in k, bones_idx_dict)] = 0
            for finger in fingers:
                other_fingers = fingers - {finger}
                bw[
                    get_conflict_mask(
                        dominant_idx,
                        lambda k, h=hand, f=finger: h in k and f in k,
                        lambda k, h=hand, ofs=other_fingers: h in k and any(of in k for of in ofs),
                        bones_idx_dict,
                    )
                ] = 0

    # Normalize weights to sum to 1
    bw = bw / (bw.sum(dim=-1, keepdim=True) + 1e-10)

    # Only keep weights from the top joints_per_point joints
    joints_per_point = 4
    thresholds = torch.topk(bw, k=joints_per_point, dim=-1, sorted=True).values[..., -1:]
    bw[bw < thresholds] = 0

    return bw
