import numpy as np
import trimesh
import os
import json
import math
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    o3d = None
    OPEN3D_AVAILABLE = False
import torch


def sample_surface(mesh, count, face_weight=None, sample_color=False, seed=147):

    if face_weight is None:
        # len(mesh.faces) float, array of the areas
        # of each face of the mesh
        face_weight = mesh.area_faces

    # cumulative sum of weights (len(mesh.faces))
    weight_cum = np.cumsum(face_weight)

    # seed the random number generator as requested
    random = np.random.default_rng(seed).random

    # last value of cumulative sum is total summed weight/area
    face_pick = random(count) * weight_cum[-1]
    # get the index of the selected faces
    face_index = np.searchsorted(weight_cum, face_pick)

    # pull triangles into the form of an origin + 2 vectors
    tri_origins = mesh.vertices[mesh.faces[:, 0]]
    tri_vectors = mesh.vertices[mesh.faces[:, 1:]].copy()
    tri_vectors -= np.tile(tri_origins, (1, 2)).reshape((-1, 2, 3))

    # pull the vectors for the faces we are going to sample from
    tri_origins = tri_origins[face_index]
    tri_vectors = tri_vectors[face_index]

    if sample_color and hasattr(mesh.visual, "uv"):
        uv_origins = mesh.visual.uv[mesh.faces[:, 0]]
        uv_vectors = mesh.visual.uv[mesh.faces[:, 1:]].copy()
        uv_origins_tile = np.tile(uv_origins, (1, 2)).reshape((-1, 2, 2))
        uv_vectors -= uv_origins_tile
        uv_origins = uv_origins[face_index]
        uv_vectors = uv_vectors[face_index]

    # randomly generate two 0-1 scalar components to multiply edge vectors b
    random_lengths = random((len(tri_vectors), 2, 1))

    # points will be distributed on a quadrilateral if we use 2 0-1 samples
    # if the two scalar components sum less than 1.0 the point will be
    # inside the triangle, so we find vectors longer than 1.0 and
    # transform them to be inside the triangle
    random_test = random_lengths.sum(axis=1).reshape(-1) > 1.0
    random_lengths[random_test] -= 1.0
    random_lengths = np.abs(random_lengths)

    # multiply triangle edge vectors by the random lengths and sum
    sample_vector = (tri_vectors * random_lengths).sum(axis=1)

    # finally, offset by the origin to generate
    # (n,3) points in space on the triangle
    samples = sample_vector + tri_origins

    if sample_color:
        if hasattr(mesh.visual, "uv"):
            sample_uv_vector = (uv_vectors * random_lengths).sum(axis=1)
            uv_samples = sample_uv_vector + uv_origins
            try:
                texture = mesh.visual.material.baseColorTexture
            except:
                texture = mesh.visual.material.image
            colors = trimesh.visual.color.uv_to_interpolated_color(uv_samples, texture)
        else:
            colors = mesh.visual.face_colors[face_index]

        return samples, face_index, colors

    return samples, face_index


def get_ray_directions(W, H, fx, fy, cx, cy, use_pixel_centers=True):
    pixel_center = 0.5 if use_pixel_centers else 0
    i, j = np.meshgrid(
        np.arange(W, dtype=np.float32) + pixel_center,
        np.arange(H, dtype=np.float32) + pixel_center,
        indexing="xy",
    )
    directions = np.stack(
        [(i - cx) / fx, -(j - cy) / fy, -np.ones_like(i)], -1
    ) 

    return directions


def gen_pcd(depth, c2w_opengl, camera_angle_x):

    h, w = depth.shape
    
    depth_valid = depth < 65500.0
    depth = depth[depth_valid]
    focal = (
        0.5 * w / math.tan(0.5 * camera_angle_x)
    )  # scaled focal length
    ray_directions = get_ray_directions(w, h, focal, focal, w // 2, h // 2)
    points_c = ray_directions[depth_valid] * depth[:, None]
    points_c_homo = np.concatenate(
        [points_c, np.ones_like(points_c[..., :1])], axis=-1
    )
    org_points = (points_c_homo @ c2w_opengl.T)[..., :3]

    return org_points


def save_point_cloud(coord, color=None, file_path="pc.ply", logger=None):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    coord = np.array(coord)
    if color is not None:
        color = np.array(color)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord)
    pcd.colors = o3d.utility.Vector3dVector(np.ones_like(coord) if color is None else color)
    o3d.io.write_point_cloud(file_path, pcd)
    if logger is not None:
        logger.info(f"Save Point Cloud to: {file_path}")


def vis_pcd_feat(coord, point_feat, save_path):
    class TorchPCA(object):

        def __init__(self, n_components):
            self.n_components = n_components

        def fit(self, X):
            self.mean_ = X.mean(dim=0)
            unbiased = X - self.mean_.unsqueeze(0)
            U, S, V = torch.pca_lowrank(unbiased, q=self.n_components, center=False, niter=4)
            self.components_ = V.T
            self.singular_values_ = S
            return self

        def transform(self, X):
            t0 = X - self.mean_.unsqueeze(0)
            projected = t0 @ self.components_.T
            return projected
        
    fit_pca = TorchPCA(n_components=3).fit(point_feat)
    x_red = fit_pca.transform(point_feat)
    if isinstance(x_red, np.ndarray):
        x_red = torch.from_numpy(x_red)
    x_red -= x_red.min(dim=0, keepdim=True).values
    x_red /= x_red.max(dim=0, keepdim=True).values

    save_point_cloud(coord.detach().cpu(), x_red.detach().cpu(), save_path)