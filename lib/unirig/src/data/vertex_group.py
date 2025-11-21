import platform
import os
if platform.system() == "Linux":
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

from typing import Dict, List, Tuple, Literal
from dataclasses import dataclass
from collections import defaultdict
from abc import ABC, abstractmethod
import numpy as np
from numpy import ndarray

from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path, connected_components

from .asset import Asset
from .spec import ConfigSpec

@dataclass
class VertexGroupConfig(ConfigSpec):
    '''
    Config to sample vertex group.
    '''
    
    # names
    names: List[str]
    
    # kwargs
    kwargs: Dict[str, Dict]
    
    @classmethod
    def parse(cls, config) -> 'VertexGroupConfig':
        cls.check_keys(config)
        return VertexGroupConfig(
            names=config.get('names', []),
            kwargs=config.get('kwargs', {}),
        )

class VertexGroup(ABC):
    
    @abstractmethod
    def __init__(self, **kwargs):
        pass
    
    @abstractmethod
    def get_vertex_group(self, asset: Asset) -> Dict[str, ndarray]:
        pass
    
class VertexGroupSkin(VertexGroup):
    '''
    Capture skin.
    '''
    
    def __init__(self, **kwargs):
        pass
    
    def get_vertex_group(self, asset: Asset) -> Dict[str, ndarray]:
        return {
            'skin': asset.skin / (asset.skin.sum(axis=-1, keepdims=True) + 1e-6),
        }

class VertexGroupGeodesicDistance(VertexGroup):
    '''
    Calculate geodesic distance.
    '''
    def __init__(self, **kwargs):
        self.deterministic = kwargs.get('deterministic', False)
        self.soft_mask = kwargs.get('soft_mask', False)
    
    def _prepare(
        self,
        joints: ndarray, # (J, 3)
        edges: List[Tuple[int, int]],
    ) -> Tuple[ndarray, ndarray]:
        J = joints.shape[0]
        dis_matrix = np.ones((J, J)) * 100.0
        step_matrix = np.ones((J, J)) * 100.0
        def dis(x: ndarray, y: ndarray):
            return np.linalg.norm(x-y)
        for i in range(J):
            dis_matrix[i, i] = 0.
            step_matrix[i, i] = 0.
        for edge in edges:
            dis_matrix[edge[0], edge[1]] = dis(joints[edge[0]], joints[edge[1]])
            dis_matrix[edge[1], edge[0]] = dis(joints[edge[0]], joints[edge[1]])
            step_matrix[edge[0], edge[1]] = 1
            step_matrix[edge[1], edge[0]] = 1
        # floyd
        for k in range(J):
            dis_matrix = np.minimum(dis_matrix, dis_matrix[:, k][:, np.newaxis] + dis_matrix[k, :][np.newaxis, :])
            step_matrix = np.minimum(step_matrix, step_matrix[:, k][:, np.newaxis] + step_matrix[k, :][np.newaxis, :])
        return dis_matrix, step_matrix
    
    def get_vertex_group(self, asset: Asset) -> Dict[str, ndarray]:
        children = defaultdict(list)
        edges = []
        for (id, p) in enumerate(asset.parents):
            if p is not None:
                edges.append((id, p))
                children[p].append(id)
        child = []
        tails = asset.tails.copy()
        for id in range(asset.J):
            if len(children[id]) == 1:
                child.append(children[id][0])
            else:
                child.append(id)
                if self.deterministic:
                    tails[id] = asset.joints[id]
        child = np.array(child)
        dis_matrix, step_matrix = self._prepare(
            joints=asset.joints,
            edges=edges,
        )
        geo_dis, geo_mask = get_geodesic_distance(
            vertices=asset.vertices,
            joints=asset.joints,
            tails=tails,
            dis_matrix=dis_matrix,
            step_matrix=step_matrix,
            child=child,
            soft_mask=self.soft_mask,
        )
        return {
            'geodesic_distance': geo_dis,
            'geodesic_mask': geo_mask,
        }

class VertexGroupVoxelSkin(VertexGroup):
    '''
    Capture voxel skin.
    '''
    
    def __init__(self, **kwargs):
        self.grid = kwargs.get('grid', 64)
        self.alpha = kwargs.get('alpha', 0.5)
        self.link_dis = kwargs.get('link_dis', 0.00001)
        self.grid_query = kwargs.get('grid_query', 27)
        self.vertex_query = kwargs.get('vertex_query', 27)
        self.grid_weight = kwargs.get('grid_weight', 3.0)
        self.mode = kwargs.get('mode', 'square')
        self.backend = kwargs.get('backend', 'pyrender')
    
    def get_vertex_group(self, asset: Asset) -> Dict[str, ndarray]:
        
        # normalize into [-1, 1] first
        min_vals = np.min(asset.vertices, axis=0)
        max_vals = np.max(asset.vertices, axis=0)
        
        center = (min_vals + max_vals) / 2
        
        scale = np.max(max_vals - min_vals) / 2
        
        normalized_vertices = (asset.vertices - center) / scale
        normalized_joints = (asset.joints - center) / scale
        
        grid_coords = voxelization(
            vertices=normalized_vertices,
            faces=asset.faces,
            grid=self.grid,
            backend=self.backend,
        )
        skin = voxel_skin(
            grid=self.grid,
            grid_coords=grid_coords,
            joints=normalized_joints,
            vertices=normalized_vertices,
            faces=asset.faces,
            alpha=self.alpha,
            link_dis=self.link_dis,
            grid_query=self.grid_query,
            vertex_query=self.vertex_query,
            grid_weight=self.grid_weight,
            mode=self.mode,
        )
        skin = np.nan_to_num(skin, nan=0., posinf=0., neginf=0.)
        return {
            'voxel_skin': skin,
        }

class VertexGroupMeshPartDistance(VertexGroup):
    def __init__(self, **kwargs):
        self.part_dim = kwargs['part_dim']
        self.dis_dim = kwargs['dis_dim']
    
    def get_vertex_group(self, asset: Asset) -> Dict[str, ndarray]:
        tot, vertex_labels, face_labels = find_connected_components(asset.vertices, asset.faces)
        # (N, dis_dim)
        part_distances = compute_distances_in_components(asset.vertices, asset.faces, vertex_labels, tot, self.dis_dim)
        # (tot, part_dim)
        part_vectors = generate_spread_vectors(tot, self.part_dim)
        # (N, part_dim)
        part_vectors = np.zeros((asset.vertices.shape[0], self.part_dim))
        for i in range(tot):
            part_vectors[labels == i] = part_vectors[i]
        return {
            'num_parts': tot,
            'part_vectors': part_vectors,
            'part_distances': part_distances,
        }

# TODO: move this into a new file
class VertexGroupMeshParts(VertexGroup):
    def __init__(self, **kwargs):
        pass
    
    def get_vertex_group(self, asset: Asset) -> Dict[str, ndarray]:
        tot, vertex_labels, face_labels = find_connected_components(asset.vertices, asset.faces)
        asset.meta['num_parts'] = tot
        asset.meta['vertex_labels'] = vertex_labels
        asset.meta['face_labels'] = face_labels
        return {}

def get_geodesic_distance(
    vertices: ndarray, # (N, 3)
    joints: ndarray, # (J, 3)
    tails: ndarray, # (J, 3)
    dis_matrix: ndarray, # (J, J)
    step_matrix: ndarray, # (J, J)
    child: ndarray,
    eps: float=1e-4,
    soft_mask: bool=False,
) -> Tuple[ndarray, ndarray]:
    # (J, 3)
    offset = tails - joints
    inv = (1./(offset * offset + eps).sum(axis=-1))[np.newaxis, ...]
    # head
    g0 = tails[np.newaxis, ...] - vertices[:, np.newaxis, :]
    c0 = (g0 * offset[np.newaxis, ...]).sum(axis=-1) * inv
    # tail
    g1 = vertices[:, np.newaxis, :] - joints[np.newaxis, ...]
    c1 = (g1 * offset[np.newaxis, ...]).sum(axis=-1) * inv
    # (N, J)
    scale0 = (np.clip(c0, 0., 1.) + eps) / (np.clip(c0, 0., 1.) + np.clip(c1, 0., 1.) + eps * 2)
    scale1 = -scale0 + 1
    # (N, J, 3)
    nearest = scale0[..., np.newaxis] * joints[np.newaxis, ...] + scale1[..., np.newaxis] * tails[np.newaxis, ...]
    # (N, J)
    dis = np.linalg.norm(vertices[:, np.newaxis, :] - nearest, axis=-1)
    # (N)
    index = np.argmin(dis, axis=1)
    # (N)
    r = np.arange(dis.shape[0])
    # (N, J)
    res = (
        dis_matrix[index] * scale0[r[:, np.newaxis], index[:, np.newaxis]] +
        dis_matrix[child[index]] * scale1[r[:, np.newaxis], index[:, np.newaxis]]
    )
    if soft_mask:
        mask = (1.0 - (
            step_matrix[index] * scale0[r[:, np.newaxis], index[:, np.newaxis]] +
            step_matrix[child[index]] * scale1[r[:, np.newaxis], index[:, np.newaxis]]
        )).clip(0., 1.).astype(np.float32)
    else:
        mask = ((
            step_matrix[index] * scale0[r[:, np.newaxis], index[:, np.newaxis]] +
            step_matrix[child[index]] * scale1[r[:, np.newaxis], index[:, np.newaxis]]
        ) <= 1.).astype(np.float32)
    
    # normalize geo dis
    row_min = np.min(res, axis=0, keepdims=True)
    row_max = np.max(res, axis=0, keepdims=True)
    res = (res - row_min) / (row_max - row_min)
    res = np.nan_to_num(res, nan=0., posinf=0., neginf=0.)
    return res, mask

def get_vertex_groups(config: VertexGroupConfig) -> List[VertexGroup]:
    vertex_groups = []
    MAP = {
        'geodesic_distance': VertexGroupGeodesicDistance,
        'skin': VertexGroupSkin,
        'voxel_skin': VertexGroupVoxelSkin,
        'mesh_part_distance': VertexGroupMeshPartDistance,
        'mesh_parts': VertexGroupMeshParts,
    }
    for name in config.names:
        assert name in MAP, f"expect: [{','.join(MAP.keys())}], found: {name}"
        vertex_groups.append(MAP[name](**config.kwargs.get(name, {})))
    return vertex_groups

def voxelization(
    vertices: ndarray,
    faces: ndarray,
    grid: int=256,
    scale: float=1.0,
    backend: Literal['pyrender', 'trimesh']='trimesh',
):
    assert backend in ['pyrender', 'trimesh']
    if backend == 'pyrender':
        import pyrender
        znear = 0.05
        zfar = 4.0
        eye_dis = 2.0 # distance from eye to origin
        r_faces = np.stack([faces[:, 0], faces[:, 2], faces[:, 1]], axis=-1)
        # get zbuffers
        mesh = pyrender.Mesh(
            primitives=[
                pyrender.Primitive(
                    positions=vertices,
                    indices=np.concatenate([faces, r_faces]), # double sided
                    mode=pyrender.GLTF.TRIANGLES,
                )
            ]
        )
        scene = pyrender.Scene(bg_color=[0, 0, 0, 0])
        scene.add(mesh)
        
        camera = pyrender.OrthographicCamera(xmag=scale, ymag=scale, znear=znear, zfar=zfar)
        camera_poses = {}
        # coordinate:
        # see https://pyrender.readthedocs.io/en/latest/examples/cameras.html
        camera_poses['+z'] = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, eye_dis],
            [0, 0, 0, 1],
        ], dtype=np.float32) # look at +z (bottom to top)
        camera_poses['-z'] = np.array([
            [-1, 0, 0, 0],
            [ 0, 1, 0, 0],
            [ 0, 0,-1, -eye_dis],
            [ 0, 0, 0, 1],
        ], dtype=np.float32) # look at -z (top to bottom)
        camera_poses['+y'] = np.array([
            [1, 0, 0, 0],
            [0, 0,-1, -eye_dis],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32) # look at +y (because model is looking at -y)(front to back)
        camera_poses['-y'] = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, eye_dis],
            [0,-1, 0, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32) # look at -y (back to front)
        camera_poses['+x'] = np.array([
            [0, 0,-1, -eye_dis],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32) # look at +x (left to right)
        camera_poses['-x'] = np.array([
            [ 0, 0, 1, eye_dis],
            [ 0, 1, 0, 0],
            [-1, 0, 0, 0],
            [ 0, 0, 0, 1],
        ], dtype=np.float32) # look at -x (righy to left)
        for name, pose in camera_poses.items():
            scene.add(camera, name=name, pose=pose)
        camera_nodes = [node for node in scene.get_nodes() if isinstance(node, pyrender.Node) and node.camera is not None]
        # if you are having issues with pyrender, change `backend` to 'trimesh' in configs/transform/<name>.yaml
        renderer = pyrender.OffscreenRenderer(viewport_width=grid, viewport_height=grid)

        i, j, k = np.indices((grid, grid, grid))
        grid_indices = np.stack((i.ravel(), j.ravel(), k.ravel()), axis=1, dtype=np.int64)
        grid_coords = np.stack((i.ravel(), j.ravel(), grid-1-k.ravel()), axis=1, dtype=np.float32) * 2 / grid - 1.0 + 1.0 / grid # every position is in the middle of the grid
        depths = {}
        for cam_node in camera_nodes:
            # a = time.time()
            scene.main_camera_node = cam_node
            name = cam_node.name
            proj_depth = renderer.render(scene, flags=pyrender.constants.RenderFlags.DEPTH_ONLY | pyrender.constants.RenderFlags.OFFSCREEN)
            proj_depth[proj_depth<znear] = zfar
            proj_depth = znear + zfar - (znear * zfar) / proj_depth # back to origin
            depths[name] = proj_depth
        
        mask_z = -grid_coords[:, 2] + depths['+z'][grid-1-grid_indices[:, 1], grid_indices[:, 0]] <= eye_dis

        mask_z &= grid_coords[:, 2] + depths['-z'][grid-1-grid_indices[:, 1], grid-1-grid_indices[:, 0]] <= eye_dis
        
        mask_x = -grid_coords[:, 0] + depths['+x'][grid-1-grid_indices[:, 1], grid-1-grid_indices[:, 2]] <= eye_dis
        
        mask_x &= grid_coords[:, 0] + depths['-x'][grid-1-grid_indices[:, 1], grid_indices[:, 2]] <= eye_dis
        
        mask_y = -grid_coords[:, 1] + depths['+y'][grid_indices[:, 2], grid_indices[:, 0]] <= eye_dis
        
        mask_y &= grid_coords[:, 1] + depths['-y'][grid-1-grid_indices[:, 2], grid_indices[:, 0]] <= eye_dis
        
        mask = (mask_x & mask_y) | (mask_x & mask_z) | (mask_y & mask_z)
        grid_coords = grid_coords[mask]
        return grid_coords
    elif backend == 'trimesh':
        import trimesh
        # Create trimesh object
        mesh_tri = trimesh.Trimesh(vertices=vertices, faces=faces)
        voxel_size = 2 / grid

        # Voxelize the mesh using trimesh
        voxel_grid = mesh_tri.voxelized(pitch=voxel_size)

        # Get the voxel matrix
        voxel_matrix = voxel_grid.matrix

        # Fill interior voxels using projection method
        grids = np.indices(voxel_matrix.shape)
        x_coord = grids[0, ...]
        y_coord = grids[1, ...]
        z_coord = grids[2, ...]

        INF = 2147483647
        x_tmp = x_coord.copy()
        x_tmp[~voxel_matrix] = INF
        x_min = x_tmp.min(axis=0)
        x_tmp[~voxel_matrix] = -1
        x_max = x_tmp.max(axis=0)

        y_tmp = y_coord.copy()
        y_tmp[~voxel_matrix] = INF
        y_min = y_tmp.min(axis=1)
        y_tmp[~voxel_matrix] = -1
        y_max = y_tmp.max(axis=1)

        z_tmp = z_coord.copy()
        z_tmp[~voxel_matrix] = INF
        z_min = z_tmp.min(axis=2)
        z_tmp[~voxel_matrix] = -1
        z_max = z_tmp.max(axis=2)

        in_x = (x_coord >= x_min[None, :, :]) & (x_coord <= x_max[None, :, :])
        in_y = (y_coord >= y_min[:, None, :]) & (y_coord <= y_max[:, None, :])
        in_z = (z_coord >= z_min[:, :, None]) & (z_coord <= z_max[:, :, None])

        count = in_x.astype(int) + in_y.astype(int) + in_z.astype(int)
        fill_mask = count >= 2
        voxel_matrix = voxel_matrix | fill_mask

        # Get voxel coordinates
        x, y, z = np.where(voxel_matrix)
        grid_indices = np.stack([x, y, z], axis=1)

        # Convert to world coordinates
        # Trimesh uses a transform matrix to get the origin
        origin = voxel_grid.transform[:3, 3]  # Extract translation from 4x4 transform
        grid_coords = origin + (grid_indices + 0.5) * voxel_size
        return grid_coords

def voxel_skin(
    grid: int,
    grid_coords: ndarray, # (M, 3)
    joints: ndarray, # (J, 3)
    vertices: ndarray, # (N, 3)
    faces: ndarray, # (F, 3)
    alpha: float=0.5,
    link_dis: float=0.00001,
    grid_query: int=27,
    vertex_query: int=27,
    grid_weight: float=3.0,
    mode: str='square',
):  
    
    # https://dl.acm.org/doi/pdf/10.1145/2485895.2485919
    assert mode in ['square', 'exp']
    J = joints.shape[0]
    M = grid_coords.shape[0]
    N = vertices.shape[0]
    
    grid_tree = cKDTree(grid_coords)
    vertex_tree = cKDTree(vertices)
    joint_tree = cKDTree(joints)
    
    # make combined vertices
    # 0   ~ N-1: mesh vertices
    # N   ~ N+M-1: grid vertices
    combined_vertices = np.concatenate([vertices, grid_coords], axis=0)
    
    # link adjacent grids
    dist, idx = grid_tree.query(grid_coords, grid_query) # 3*3*3
    dist = dist[:, 1:]
    idx = idx[:, 1:]
    mask = (0 < dist) & (dist < 2/grid*1.001)
    source_grid2grid = np.repeat(np.arange(M), grid_query-1)[mask.ravel()] + N
    to_grid2grid = idx[mask] + N
    weight_grid2grid = dist[mask] * grid_weight
    
    # link very close vertices
    dist, idx = vertex_tree.query(vertices, 4)
    dist = dist[:, 1:]
    idx = idx[:, 1:]
    mask = (0 < dist) & (dist < link_dis*1.001)
    source_close = np.repeat(np.arange(N), 3)[mask.ravel()]
    to_close = idx[mask]
    weight_close = dist[mask]
    
    # link grids to mesh vertices
    dist, idx = vertex_tree.query(grid_coords, vertex_query)
    mask = (0 < dist) & (dist < 2/grid*1.001) # sqrt(3)
    source_grid2vertex = np.repeat(np.arange(M), vertex_query)[mask.ravel()] + N
    to_grid2vertex = idx[mask]
    weight_grid2vertex = dist[mask]
    
    # build combined vertices tree
    combined_tree = cKDTree(combined_vertices)
    # link joints to the neartest vertices
    _, joint_indices = combined_tree.query(joints)
    
    # build graph
    source_vertex2vertex = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]], axis=0)
    to_vertex2vertex = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]], axis=0)
    weight_vertex2vertex = np.sqrt(((vertices[source_vertex2vertex] - vertices[to_vertex2vertex])**2).sum(axis=-1))
    graph = csr_matrix(
        (np.concatenate([weight_close, weight_vertex2vertex, weight_grid2grid, weight_grid2vertex]),
        (
            np.concatenate([source_close, source_vertex2vertex, source_grid2grid, source_grid2vertex], axis=0),
            np.concatenate([to_close, to_vertex2vertex, to_grid2grid, to_grid2vertex], axis=0)),
        ),
        shape=(N+M, N+M),
    )
    
    # get shortest path (J, N+M)
    dist_matrix = shortest_path(graph, method='D', directed=False, indices=joint_indices)
    
    # (J, N)
    dis_vertex2joint = dist_matrix[:, :N]
    unreachable = np.isinf(dis_vertex2joint).all(axis=0)
    k = min(J, 3)
    dist, idx = joint_tree.query(vertices[unreachable], k)
    
    # make sure at least one value in dis is not inf
    unreachable_indices = np.where(unreachable)[0]
    row_indices = idx
    col_indices = np.repeat(unreachable_indices, k).reshape(-1, k)
    dis_vertex2joint[row_indices, col_indices] = dist
    
    finite_vals = dis_vertex2joint[np.isfinite(dis_vertex2joint)]
    max_dis = np.max(finite_vals)
    dis_vertex2joint = np.nan_to_num(dis_vertex2joint, nan=max_dis, posinf=max_dis, neginf=max_dis)
    dis_vertex2joint = np.maximum(dis_vertex2joint, 1e-6)
    # (J, N)
    if mode == 'exp':
        skin = np.exp(-dis_vertex2joint / max_dis * 20.0)
    elif mode == 'square':
        skin = (1./((1-alpha)*dis_vertex2joint + alpha*dis_vertex2joint**2))**2
    else:
        assert False, f'invalid mode: {mode}'
    skin = skin / skin.sum(axis=0)
    # (N, J)
    skin = skin.transpose()
    return skin

def find_connected_components(vertices: ndarray, faces: ndarray) -> Tuple[int, ndarray]:
    '''
    Find connected components of a mesh.
    
    Returns:
        int: number of connected components
        ndarray: labels of connected components
    '''
    N = vertices.shape[0]
    edges = []
    for face in faces:
        v0, v1, v2 = face
        edges.append([v0, v1])
        edges.append([v1, v2])
        edges.append([v2, v0])
    
    edges = np.array(edges)
    row = edges[:, 0]
    col = edges[:, 1]
    data = np.ones(len(edges), dtype=int)
    adj_matrix = csr_matrix((data, (row, col)), shape=(N, N))
    adj_matrix = adj_matrix + adj_matrix.T
    
    tot, vertex_labels = connected_components(adj_matrix, directed=False, return_labels=True)
    face_labels = vertex_labels[faces[:, 0]]
    return tot, vertex_labels, face_labels

def compute_distances_in_components(vertices: ndarray, faces: ndarray, vertex_labels: ndarray, tot: int, k: int) -> ndarray:
    N = vertices.shape[0]
    edges = []
    weights = []
    for face in faces:
        v0, v1, v2 = face
        w01 = np.linalg.norm(vertices[v0] - vertices[v1])
        w12 = np.linalg.norm(vertices[v1] - vertices[v2])
        w20 = np.linalg.norm(vertices[v2] - vertices[v0])
        edges.extend([[v0, v1], [v1, v2], [v2, v0]])
        weights.extend([w01, w12, w20])
    
    edges = np.array(edges)
    weights = np.array(weights)
    row = edges[:, 0]
    col = edges[:, 1]
    adj_matrix = csr_matrix((weights, (row, col)), shape=(N, N))
    adj_matrix = adj_matrix + adj_matrix.T

    distance_matrix = np.full((N, k), np.inf)  # (N, k)

    for component_id in range(tot):
        component_mask = (vertex_labels == component_id)
        component_vertices_idx = np.where(component_mask)[0]
        n_component = len(component_vertices_idx)
        
        if n_component == 0:
            continue
        
        if n_component >= k:
            sampled_indices = np.random.permutation(n_component)[:k]
        else:
            sampled_indices = np.concatenate([
                np.random.permutation(n_component),
                np.random.randint(0, n_component, k - n_component)
            ])
        sampled_vertices = component_vertices_idx[sampled_indices]

        dist_matrix = shortest_path(adj_matrix, indices=sampled_vertices, directed=False)
        dist_matrix = dist_matrix[:, component_mask].T
        # normalize into [0, 1]
        max_value = dist_matrix.max()
        min_value = dist_matrix.min()
        if max_value < min_value + 1e-6:
            dist_matrix[...] = 0.
        else:
            dist_matrix = (dist_matrix - min_value) / (max_value - min_value)
        
        distance_matrix[component_mask, :] = dist_matrix
    
    return distance_matrix

def generate_spread_vectors(tot: int, dim: int, iterations: int=100, lr: float=1.0) -> ndarray:
    if tot <= 0:
        return None
    
    vectors = np.random.randn(tot, dim)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = np.nan_to_num(vectors, nan=1.0, posinf=1.0, neginf=1.0)
    
    for _ in range(iterations):
        diff = vectors[np.newaxis, :, :] - vectors[:, np.newaxis, :]
        norm_sq = np.sum(diff ** 2, axis=2)
        weight = 1. / (norm_sq + 1.)
        vectors += np.sum(diff * weight[:, :, np.newaxis] * lr, axis=1)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    
    return vectors
