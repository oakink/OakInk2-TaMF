from __future__ import annotations

from copy import deepcopy
import typing

import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import trimesh
import numpy as np

from .console_io import RedirectStream
from .vis_cv2_util import edge_list_bbox

if typing.TYPE_CHECKING:
    from typing import Callable


def create_pc(xyz, rgb):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    return pcd


class VizContext:
    def __init__(self, non_block=False) -> None:
        self.vis = o3d.visualization.VisualizerWithKeyCallback()

        self.running = True

        def shutdown_callback(vis):
            self.running = False

        self.vis.register_key_callback(ord("Q"), shutdown_callback)

        self.non_block = non_block

    # note: this two-stage initialization is because python resource management is by garbage collection
    # which means that you cannot easily determine when resource will be destructed
    # thus need init / deinit method to manually control the timepoint of resource creation / destruction
    # if migrate to c++, for sure RAII
    def init(self, width=1920, height=1080):
        self.vis.create_window(width=width, height=height)

    def deinit(self):
        self.vis.close()
        self.vis.destroy_window()

    def add_geometry(self, pc, reset_bounding_box=True):
        self.vis.add_geometry(pc, reset_bounding_box=reset_bounding_box)

    def add_geometry_list(self, pc_list, reset_bounding_box=False):
        for pc in pc_list:
            self.vis.add_geometry(pc, reset_bounding_box=reset_bounding_box)

    def update_geometry(self, pc):
        self.vis.update_geometry(pc)

    def update_geometry_list(self, pc_list):
        for pc in pc_list:
            self.vis.update_geometry(pc)

    def step(self):
        self.vis.poll_events()
        self.vis.update_renderer()

    def remove_geometry(self, pc, reset_bounding_box=True):
        self.vis.remove_geometry(pc, reset_bounding_box=reset_bounding_box)

    def remove_geometry_list(self, pc_list, reset_bounding_box=False):
        for pc in pc_list:
            self.vis.remove_geometry(pc, reset_bounding_box=reset_bounding_box)

    def reset(self):
        self.running = True

    def condition(self):
        return self.running and (not self.non_block)

    def register_key_callback(self, keyord, callback):
        self.vis.register_key_callback(keyord, callback)


def cvt_from_trimesh(mesh: trimesh.Trimesh, use_vertex_color=True):
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    if use_vertex_color and mesh.visual.kind == "vertex":
        o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(np.array(mesh.visual.vertex_colors[:, :3]) / 255.0)
    else:
        o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(
            np.array(
                [
                    [0.8, 0.8, 0.8],
                ]
                * len(vertices)
            )
        )
    o3d_mesh.compute_vertex_normals()
    return o3d_mesh


def transf_o3d_mesh(mesh: o3d.geometry.TriangleMesh, transf):
    mesh = deepcopy(mesh)
    mesh.transform(transf)
    return mesh


def caculate_align_mat(vec):
    vec = vec / np.linalg.norm(vec)
    z_unit_Arr = np.array([0, 0, 1])

    z_mat = np.array(
        [
            [0, -z_unit_Arr[2], z_unit_Arr[1]],
            [z_unit_Arr[2], 0, -z_unit_Arr[0]],
            [-z_unit_Arr[1], z_unit_Arr[0], 0],
        ]
    )

    z_c_vec = np.matmul(z_mat, vec)
    z_c_vec_mat = np.array(
        [
            [0, -z_c_vec[2], z_c_vec[1]],
            [z_c_vec[2], 0, -z_c_vec[0]],
            [-z_c_vec[1], z_c_vec[0], 0],
        ]
    )

    if np.dot(z_unit_Arr, vec) == -1:
        qTrans_Mat = -np.eye(3, 3)
    elif np.dot(z_unit_Arr, vec) == 1:
        qTrans_Mat = np.eye(3, 3)
    else:
        qTrans_Mat = np.eye(3, 3) + z_c_vec_mat + np.matmul(z_c_vec_mat, z_c_vec_mat) / (1 + np.dot(z_unit_Arr, vec))

    return qTrans_Mat


def create_coord_system_can(scale=1, transf=None, merge=True):
    axis_list = []
    cylinder_radius = 0.0015 * scale
    cone_radius = 0.002 * scale
    cylinder_height = 0.05 * scale
    cone_height = 0.008 * scale
    resolution = int(20 * scale)
    cylinder_split = 4
    cone_split = 1

    x = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=cylinder_radius,
        cone_radius=cone_radius,
        cylinder_height=cylinder_height,
        cone_height=cone_height,
        resolution=resolution,
        cylinder_split=cylinder_split,
        cone_split=cone_split,
    )
    x.paint_uniform_color([255 / 255.0, 0 / 255.0, 0 / 255.0])
    align_x = caculate_align_mat(np.array([1, 0, 0]))
    x = x.rotate(align_x, center=(0, 0, 0))
    x.compute_vertex_normals()
    axis_list.append(x)

    y = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=cylinder_radius,
        cone_radius=cone_radius,
        cylinder_height=cylinder_height,
        cone_height=cone_height,
        resolution=resolution,
        cylinder_split=cylinder_split,
        cone_split=cone_split,
    )
    y.paint_uniform_color([0 / 255.0, 255 / 255.0, 0 / 255.0])

    align_y = caculate_align_mat(np.array([0, 1, 0]))
    y = y.rotate(align_y, center=(0, 0, 0))
    y.compute_vertex_normals()
    axis_list.append(y)

    z = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=cylinder_radius,
        cone_radius=cone_radius,
        cylinder_height=cylinder_height,
        cone_height=cone_height,
        resolution=resolution,
        cylinder_split=cylinder_split,
        cone_split=cone_split,
    )
    z.paint_uniform_color([0 / 255.0, 0 / 255.0, 255 / 255.0])
    align_z = caculate_align_mat(np.array([0, 0, 1]))
    z = z.rotate(align_z, center=(0, 0, 0))
    z.compute_vertex_normals()
    axis_list.append(z)

    if transf is not None:
        assert transf.shape == (4, 4), "transf must be 4x4 Transformation matrix"
        for i, axis in enumerate(axis_list):
            axis.rotate(transf[:3, :3], center=(0, 0, 0))
            axis.translate(transf[:3, 3].T)
            axis_list[i] = axis

    if not merge:
        res = axis_list
    else:
        res = o3d_merge_mesh(axis_list, merge_color=True)
    return res


def o3d_merge_mesh(mesh_list, merge_color=False):
    # Collect all vertices and triangles
    all_vertices = []
    all_triangles = []
    if merge_color:
        all_vertex_colors = []
    vertex_offset = 0

    for mesh in mesh_list:
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        vertex_colors = np.asarray(mesh.vertex_colors)

        all_vertices.append(vertices)
        all_triangles.append(triangles + vertex_offset)
        if merge_color:
            all_vertex_colors.append(vertex_colors)
        vertex_offset += vertices.shape[0]

    # Create new mesh
    merged_mesh = o3d.geometry.TriangleMesh()
    merged_mesh.vertices = o3d.utility.Vector3dVector(np.vstack(all_vertices))
    merged_mesh.triangles = o3d.utility.Vector3iVector(np.vstack(all_triangles))
    if merge_color:
        merged_mesh.vertex_colors = o3d.utility.Vector3dVector(np.vstack(all_vertex_colors))

    return merged_mesh


class VizContext2:
    def __init__(self) -> None:
        app = gui.Application.instance
        app.initialize()

        self._reset()
        self.running = True

        # storage for key callback
        self._callback_store: dict[int, Callable] = {}

        def shutdown_callback():
            self.running = False

        self.register_key_callback(ord("q"), shutdown_callback)

    def _reset(self):
        self._window = None
        self._scene_widget = None

        # storage for object handling
        self._name_store = None
        self._geometry_store = None
        self._material_store = None

    def init(self):
        with RedirectStream():
            self._window = gui.Application.instance.create_window("VizContext2", width=1080, height=768)
        self._scene_widget = gui.SceneWidget()
        self._scene_widget.scene = rendering.Open3DScene(self._window.renderer)
        self._scene_widget.scene.camera.look_at([0, 0, 0], [2, 2, 2], [0, 1, 0])
        self._window.add_child(self._scene_widget)

        # install key callback
        def _handle_key(event):
            ret_type = gui.Widget.EventCallbackResult.IGNORED
            if event.type == gui.MouseEvent.Type.BUTTON_DOWN:
                key = int(event.key)
                if key in self._callback_store:
                    self._callback_store[key]()
                    ret_type = gui.Widget.EventCallbackResult.HANDLED
            return ret_type

        self._scene_widget.set_on_key(_handle_key)

        self._name_store = {}
        self._geometry_store = {}
        self._material_store = {}

    def deinit(self):
        self._window.close()
        gui.Application.instance.run_one_tick()

    def add_geometry(self, pc):
        _addr = id(pc)
        if _addr in self._name_store:
            return

        _name = f"Geometry_{_addr}"
        self._name_store[_addr] = _name
        self._geometry_store[_addr] = pc

        _mat = rendering.MaterialRecord()
        _mat.shader = "defaultLit"
        _mat.point_size = 3 * self._window.scaling
        self._material_store[_addr] = _mat

        def _add_geometry():
            self._scene_widget.scene.add_geometry(_name, pc, _mat)

        gui.Application.instance.post_to_main_thread(self._window, _add_geometry)

    def add_geometry_list(self, pc_list):
        for pc in pc_list:
            self.add_geometry(pc)

    def update_geometry(self, pc):
        _addr = id(pc)
        if _addr not in self._name_store:
            return

        _name = self._name_store[_addr]
        self._geometry_store[_addr] = pc
        _mat = self._material_store[_addr]

        def _update_geometry():
            self._scene_widget.scene.remove_geometry(_name)
            self._scene_widget.scene.add_geometry(_name, pc, _mat)

        gui.Application.instance.post_to_main_thread(self._window, _update_geometry)

    def step(self):
        status = gui.Application.instance.run_one_tick()
        if not status:
            self.running = False

    def remove_geometry(self, pc):
        _addr = id(pc)
        if _addr not in self._name_store:
            return

        _name = self._name_store[_addr]

        def _remove_geometry():
            self._scene_widget.scene.remove_geometry(_name)

        gui.Application.instance.post_to_main_thread(self._window, _remove_geometry)

        # del
        self._name_store.pop(_addr)
        self._geometry_store.pop(_addr)
        self._material_store.pop(_addr)

    def remove_geometry_list(self, pc_list):
        for pc in pc_list:
            self.remove_geometry(pc)

    def reset(self):
        self._scene_widget.scene.clear_geometry()
        self._reset()
        self.running = True

    def condition(self):
        return self.running

    def register_key_callback(self, keyord, callback):
        self._callback_store[keyord] = callback


class VizContext2MultiWin:
    def __init__(self) -> None:
        pass


# geometry function
def o3d_bbox(corner, color=None):
    # if color is none, use blue as default
    if color is None:
        color = np.array([0, 0, 1.0], dtype=np.float32)
        # repeat to edge_list_bbox
        color = np.repeat(color[np.newaxis, :], len(edge_list_bbox), axis=0)
    elif isinstance(color, np.ndarray) and color.ndim == 1:
        color = np.repeat(color[np.newaxis, :], len(edge_list_bbox), axis=0)
    bbox_lineset = o3d.geometry.LineSet()
    bbox_lineset.points = o3d.utility.Vector3dVector(corner)
    bbox_lineset.lines = o3d.utility.Vector2iVector(edge_list_bbox)
    bbox_lineset.colors = o3d.utility.Vector3dVector(color)
    return bbox_lineset


def o3d_lineset_set_color(lineset, color=None):
    nc = np.asarray(lineset.lines).shape[0]
    if color is None:
        color = np.array([0, 0, 1.0], dtype=np.float32)
        # repeat to edge_list_bbox
        color = np.repeat(color[np.newaxis, :], nc, axis=0)
    elif isinstance(color, np.ndarray) and color.ndim == 1:
        color = np.repeat(color[np.newaxis, :], nc, axis=0)
    lineset.colors = o3d.utility.Vector3dVector(color)
    return lineset


def o3d_load_camera_param(filepath):
    return o3d.io.read_pinhole_camera_parameters(filepath)
