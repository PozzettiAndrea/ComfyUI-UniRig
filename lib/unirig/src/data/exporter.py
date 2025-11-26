import numpy as np
from numpy import ndarray
from typing import List, Union, Tuple
from collections import defaultdict
import os
import subprocess
import pickle
import tempfile

try:
    import open3d as o3d
    OPEN3D_EQUIPPED = True
except ImportError:
    o3d = None
    OPEN3D_EQUIPPED = False

class Exporter():
    
    def _safe_make_dir(self, path):
        if os.path.dirname(path) == '':
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
    
    def _export_skeleton(self, joints: ndarray, parents: List[Union[int, None]], path: str):
        format = path.split('.')[-1]
        assert format in ['obj']
        name = path.removesuffix('.obj')
        path = name + ".obj"
        self._safe_make_dir(path)
        J = joints.shape[0]
        with open(path, 'w') as file:
            file.write("o spring_joint\n")
            _joints = []
            for id in range(J):
                pid = parents[id]
                if pid is None or pid == -1:
                    continue
                bx, by, bz = joints[id]
                ex, ey, ez = joints[pid]
                _joints.extend([
                    f"v {bx} {bz} {-by}\n",
                    f"v {ex} {ez} {-ey}\n",
                    f"v {ex} {ez} {-ey + 0.00001}\n"
                ])
            file.writelines(_joints)
            
            _faces = [f"f {id*3+1} {id*3+2} {id*3+3}\n" for id in range(J)]
            file.writelines(_faces)
    
    def _export_bones(self, bones: ndarray, path: str):
        format = path.split('.')[-1]
        assert format in ['obj']
        name = path.removesuffix('.obj')
        path = name + ".obj"
        self._safe_make_dir(path)
        J = bones.shape[0]
        with open(path, 'w') as file:
            file.write("o bones\n")
            _joints = []
            for bone in bones:
                bx, by, bz = bone[:3]
                ex, ey, ez = bone[3:]
                _joints.extend([
                    f"v {bx} {bz} {-by}\n",
                    f"v {ex} {ez} {-ey}\n",
                    f"v {ex} {ez} {-ey + 0.00001}\n"
                ])
            file.writelines(_joints)
            
            _faces = [f"f {id*3+1} {id*3+2} {id*3+3}\n" for id in range(J)]
            file.writelines(_faces)
    
    def _export_skeleton_sequence(self, joints: ndarray, parents: List[Union[int, None]], path: str):
        format = path.split('.')[-1]
        assert format in ['obj']
        name = path.removesuffix('.obj')
        path = name + ".obj"
        self._safe_make_dir(path)
        J = joints.shape[0]
        for i in range(J):
            file = open(name + f"_{i}.obj", 'w')
            file.write("o spring_joint\n")
            _joints = []
            for id in range(i + 1):
                pid = parents[id]
                if pid is None:
                    continue
                bx, by, bz = joints[id]
                ex, ey, ez = joints[pid]
                _joints.extend([
                    f"v {bx} {bz} {-by}\n",
                    f"v {ex} {ez} {-ey}\n",
                    f"v {ex} {ez} {-ey + 0.00001}\n"
                ])
            file.writelines(_joints)
            
            _faces = [f"f {id*3+1} {id*3+2} {id*3+3}\n" for id in range(J)]
            file.writelines(_faces)
            file.close()
    
    def _export_mesh(self, vertices: ndarray, faces: ndarray, path: str):
        format = path.split('.')[-1]
        assert format in ['obj', 'ply']
        if path.endswith('ply'):
            if not OPEN3D_EQUIPPED:
                raise RuntimeError("open3d is not available")
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            mesh.triangles = o3d.utility.Vector3iVector(faces)
            self._safe_make_dir(path)
            o3d.io.write_triangle_mesh(path, mesh)
            return
        name = path.removesuffix('.obj')
        path = name + ".obj"
        self._safe_make_dir(path)
        with open(path, 'w') as file:
            file.write("o mesh\n")
            _vertices = []
            for co in vertices:
                _vertices.append(f"v {co[0]} {co[2]} {-co[1]}\n")
            file.writelines(_vertices)
            _faces = []
            for face in faces:
                _faces.append(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
            file.writelines(_faces)
            
    def _export_pc(self, vertices: ndarray, path: str, vertex_normals: Union[ndarray, None]=None, normal_size: float=0.01):
        if path.endswith('.ply'):
            if vertex_normals is not None:
                print("normal result will not be displayed in .ply format")
            name = path.removesuffix('.ply')
            path = name + ".ply"
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(vertices)
            # segment fault when numpy >= 2.0 !! use torch environment
            self._safe_make_dir(path)
            o3d.io.write_point_cloud(path, pc)
            return
        name = path.removesuffix('.obj')
        path = name + ".obj"
        self._safe_make_dir(path)
        with open(path, 'w') as file:
            file.write("o pc\n")
            _vertex = []
            for co in vertices:
                _vertex.append(f"v {co[0]} {co[2]} {-co[1]}\n")
            file.writelines(_vertex)
            if vertex_normals is not None:
                new_path = path.replace('.obj', '_normal.obj')
                nfile = open(new_path, 'w')
                nfile.write("o normal\n")
                _normal = []
                for i in range(vertices.shape[0]):
                    co = vertices[i]
                    x = vertex_normals[i, 0]
                    y = vertex_normals[i, 1]
                    z = vertex_normals[i, 2]
                    _normal.extend([
                        f"v {co[0]} {co[2]} {-co[1]}\n",
                        f"v {co[0]+0.0001} {co[2]} {-co[1]}\n",
                        f"v {co[0]+x*normal_size} {co[2]+z*normal_size} {-(co[1]+y*normal_size)}\n",
                        f"f {i*3+1} {i*3+2} {i*3+3}\n",
                    ])
                nfile.writelines(_normal)
    
    def _make_armature(
        self,
        vertices: Union[ndarray, None],
        joints: ndarray,
        skin: Union[ndarray, None],
        parents: List[Union[int, None]],
        names: List[str],
        faces: Union[ndarray, None]=None,
        extrude_size: float=0.03,
        group_per_vertex: int=-1,
        add_root: bool=False,
        do_not_normalize: bool=False,
        use_extrude_bone: bool=True,
        use_connect_unique_child: bool=True,
        extrude_from_parent: bool=True,
        tails: Union[ndarray, None]=None,
    ):
        '''
        This method has been replaced with subprocess-based approach.
        Call _export_fbx instead, which uses external Blender script.
        '''
        raise NotImplementedError(
            "Direct bpy usage removed. Use _export_fbx with blender_exe parameter instead."
        )

    def _clean_bpy(self):
        '''
        This method has been replaced with subprocess-based approach.
        Blender cleanup is handled by the wrapper script.
        '''
        raise NotImplementedError(
            "Direct bpy usage removed. Cleanup is handled by wrapper scripts."
        )
    
    def _export_fbx(
        self,
        path: str,
        vertices: Union[ndarray, None],
        joints: ndarray,
        skin: Union[ndarray, None],
        parents: List[Union[int, None]],
        names: List[str],
        faces: Union[ndarray, None]=None,
        extrude_size: float=0.03,
        group_per_vertex: int=-1,
        add_root: bool=False,
        do_not_normalize: bool=False,
        use_extrude_bone: bool=True,
        use_connect_unique_child: bool=True,
        extrude_from_parent: bool=True,
        tails: Union[ndarray, None]=None,
        blender_exe: Union[str, None]=None,
        uv_coords: Union[ndarray, None]=None,
        uv_faces: Union[ndarray, None]=None,
        texture_data_base64: Union[str, None]=None,
        texture_format: Union[str, None]=None,
        material_name: Union[str, None]=None,
    ):
        '''
        Export skeleton to FBX using external Blender executable.
        No longer requires bpy to be installed in current environment.

        Args:
            blender_exe: Path to Blender executable. If None, will try to get from
                        environment variable BLENDER_EXE or sys.BLENDER_EXE attribute.
        '''
        self._safe_make_dir(path)

        # Find Blender executable
        if blender_exe is None:
            blender_exe = os.environ.get('BLENDER_EXE')
        if blender_exe is None:
            import sys
            blender_exe = getattr(sys, 'BLENDER_EXE', None)
        if blender_exe is None:
            raise RuntimeError(
                "Blender executable not found. Pass blender_exe parameter, "
                "set BLENDER_EXE environment variable, or set sys.BLENDER_EXE"
            )

        # Find wrapper script (lib/blender_export_fbx.py)
        # Assume it's in lib/ relative to this file's parent directories
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # From src/data/ go up to unirig/, then up to lib/
        wrapper_script = os.path.join(current_dir, '..', '..', '..', 'blender_export_fbx.py')
        wrapper_script = os.path.abspath(wrapper_script)

        if not os.path.exists(wrapper_script):
            raise RuntimeError(
                f"Blender wrapper script not found at {wrapper_script}. "
                "Make sure lib/blender_export_fbx.py exists."
            )

        # Prepare data - convert ALL numpy types to plain Python types to avoid version conflicts
        # This is critical because Blender's bundled numpy may be older/incompatible
        def convert_to_python(obj):
            """Recursively convert numpy types to Python native types"""
            if obj is None:
                return None
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()  # Convert numpy scalar to Python int/float
            elif isinstance(obj, list):
                return [convert_to_python(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_to_python(value) for key, value in obj.items()}
            else:
                return obj

        data = {
            'joints': convert_to_python(joints),
            'parents': convert_to_python(parents),
            'names': convert_to_python(names),
            'vertices': convert_to_python(vertices),
            'faces': convert_to_python(faces),
            'skin': convert_to_python(skin),
            'tails': convert_to_python(tails),
            'uv_coords': convert_to_python(uv_coords),
            'uv_faces': convert_to_python(uv_faces),
            'texture_data_base64': texture_data_base64 if texture_data_base64 else "",
            'texture_format': texture_format if texture_format else "",
            'material_name': material_name if material_name else "Material",
            'group_per_vertex': int(group_per_vertex) if isinstance(group_per_vertex, (np.integer, np.floating)) else group_per_vertex,
            'do_not_normalize': bool(do_not_normalize),
        }

        print(f"[Exporter] Prepared FBX export data (all numpy types converted to Python native types)")
        print(f"[Exporter] Data summary: joints={len(data['joints']) if data['joints'] else 0}, "
              f"vertices={len(data['vertices']) if data['vertices'] else 0}, "
              f"faces={len(data['faces']) if data['faces'] else 0}")

        # Save to temporary pickle file
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
            pickle_path = f.name
            pickle.dump(data, f)

        print(f"[Exporter] Saved pickle data to: {pickle_path}")

        try:
            # Build command
            cmd = [
                blender_exe,
                '--background',
                '--python', wrapper_script,
                '--',
                pickle_path,
                path,
            ]

            # Add optional flags
            if add_root:
                cmd.append('--add_root')
            if not use_extrude_bone:
                cmd.append('--no_extrude_bone')
            if not use_connect_unique_child:
                cmd.append('--no_connect_unique_child')
            if not extrude_from_parent:
                cmd.append('--no_extrude_from_parent')
            cmd.append(f'--extrude_size={extrude_size}')

            # Run Blender
            print(f"[Exporter] Running Blender FBX export to: {path}")
            print(f"[Exporter] Blender command: {' '.join(cmd[:3])} ... {cmd[-2:]}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            # Always print output for debugging
            if result.stdout:
                print(f"[Exporter] Blender stdout:\n{result.stdout}")
            if result.stderr:
                print(f"[Exporter] Blender stderr:\n{result.stderr}")

            if result.returncode != 0:
                raise RuntimeError(f"FBX export failed with return code {result.returncode}")

            print(f"[Exporter] Blender completed with return code: {result.returncode}")

            # Check if output file was created
            if not os.path.exists(path):
                print(f"[Exporter] ERROR: Output file not found: {path}")
                print(f"[Exporter] Pickle input was: {pickle_path}")
                print(f"[Exporter] Wrapper script: {wrapper_script}")
                raise RuntimeError(f"FBX export completed but output file not found: {path}")
            else:
                file_size = os.path.getsize(path)
                print(f"[Exporter] ✓ FBX export successful: {path} ({file_size} bytes)")

        finally:
            # Clean up pickle file
            if os.path.exists(pickle_path):
                print(f"[Exporter] Cleaning up temporary pickle file: {pickle_path}")
                os.unlink(pickle_path)
    
    def _export_render(
        self,
        path: str,
        vertices: Union[ndarray, None],
        faces: Union[ndarray, None],
        bones: Union[ndarray, None],
        resolution: Tuple[float, float]=[256, 256],
    ):
        '''
        Rendering requires bpy. This method has been disabled.
        Create a wrapper script (lib/blender_render.py) if rendering is needed.
        '''
        raise NotImplementedError(
            "Direct bpy usage removed. Rendering functionality not implemented with subprocess yet. "
            "Create lib/blender_render.py wrapper script if needed."
        )

def _trans_to_m(v: ndarray):
    m = np.eye(4)
    m[0:3, 3] = v
    return m

def _scale_to_m(r: ndarray):
    m = np.zeros((4, 4))
    m[0, 0] = r
    m[1, 1] = r
    m[2, 2] = r
    m[3, 3] = 1.
    return m