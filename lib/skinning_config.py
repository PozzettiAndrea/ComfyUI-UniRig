"""
Configuration dataclasses for UniRig skinning quality parameters.

These provide a clean, programmatic way to override YAML config values at runtime.
"""

from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class SkinningConfig:
    """
    Configuration for skinning quality parameters.

    These parameters control the quality vs performance trade-off for skinning inference.
    Higher values generally mean better quality but more VRAM usage and slower processing.
    """

    # Voxel grid resolution for spatial weight distribution
    # Higher = more accurate weight distribution, more VRAM
    # Default: 196, Recommended: 256 for quality, 512 for best quality
    voxel_grid_size: int = 196

    # Number of surface samples for weight calculation
    # Higher = more accurate sampling, slower
    # Default: 32768
    num_samples: int = 32768

    # Number of vertex samples
    # Higher = more accurate vertex processing
    # Default: 8192
    vertex_samples: int = 8192

    # Grid size for point cloud spatial features
    # Lower = finer detail, more compute
    # Default: 0.005, try 0.002 for finer detail
    grid_size: float = 0.005

    # Power applied to voxel mask for weight sharpness
    # Higher = sharper weight transitions
    # Default: 3.0, try 2.0 for smoother transitions
    voxel_mask_power: float = 3.0

    # Number of vertices processed per batch during inference
    # Higher = faster but more VRAM
    # Default: 512
    num_train_vertex: int = 512

    # Voxel grid query neighbors
    # Higher = more context, slightly slower
    # Default: 7
    grid_query: int = 7

    # Vertex query neighbors
    # Higher = more context
    # Default: 1
    vertex_query: int = 1

    # Grid weight for voxel influence
    # Default: 3.0
    grid_weight: float = 3.0

    # Alpha blending factor
    # Default: 0.5
    alpha: float = 0.5

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'SkinningConfig':
        """Create config from dictionary, ignoring unknown keys."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)


@dataclass
class SkeletonConfig:
    """
    Configuration for skeleton extraction parameters.
    """

    # Target face count for mesh decimation
    # Higher = preserve more detail, slower processing
    # Default: 50000
    target_face_count: int = 50000

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'SkeletonConfig':
        """Create config from dictionary, ignoring unknown keys."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)
