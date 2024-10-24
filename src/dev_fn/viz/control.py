from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from typing import Optional

import numpy as np
import open3d as o3d
import matplotlib

from ..util.vis_o3d_util import VizContext, o3d_load_camera_param
from ..transform.transform_np import inv_transf_np


class VizControl:
    def __init__(self):
        self.viz_ctx = None
        self.viz_ctx_internal = False

        self.geometry_store = dict()
        self.geometry_status = dict()
        self.geometry_curr_pose = dict()

        self.multiframe_enable = False
        self.multiframe_mode = dict()
        self.multiframe_store = dict()

    def attach_viz_ctx(self, viz_ctx: Optional[VizContext] = None, viz_ctx_args: Optional[dict] = None):
        if self.viz_ctx is not None:
            self.detach_viz_ctx()

        if viz_ctx is not None:
            self.viz_ctx = viz_ctx
            self.viz_ctx_internal = False
        else:
            if viz_ctx_args is None:
                viz_ctx_args = {}
            _viz_ctx = VizContext(**viz_ctx_args)
            _viz_ctx.init()
            self.viz_ctx = _viz_ctx
            self.viz_ctx_internal = True

    def detach_viz_ctx(self):
        if self.viz_ctx is None:
            return

        self.reset()
        if self.viz_ctx_internal:
            self.viz_ctx.deinit()

        self.viz_ctx = None
        self.viz_ctx_internal = False

    def reset(self):
        for k, v in self.geometry_store.items():
            if self.geometry_status[k]:  # otherwise no geometry to be removed
                self.viz_ctx.remove_geometry(v)
        self.geometry_store = dict()
        self.geometry_status = dict()
        self.geometry_curr_pose = dict()
        self.multiframe_mode = dict()
        self.multiframe_store = dict()

    def step(self):
        self.viz_ctx.step()

    def condition(self):
        return self.viz_ctx.condition()

    def condition_reset(self):
        self.viz_ctx.reset()

    def load_pinhole_param(self, filepath):
        param = o3d_load_camera_param(filepath)
        ctr = self.viz_ctx.vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(param)

    def capture_frame(self):
        image = np.asarray(self.viz_ctx.vis.capture_screen_float_buffer(True))
        image = (image * 255).astype(np.uint8)
        image = image[:, :, ::-1]
        return image

    def register_key_callback(self, keyord, callback):
        self.viz_ctx.register_key_callback(keyord, callback)

    # geometry management
    def _paint_color_on(self, pts, colors=None):
        if colors is None:
            colors = np.ones_like(pts) * [0.9, 0.9, 0.9]
        elif isinstance(colors, (list, tuple)) and len(colors) == 3:
            colors = np.ones_like(pts) * colors
            # if any of colors greater than 1
            if np.any(colors) > 1:
                colors = colors / 255.0
        elif isinstance(colors, str):
            colors = np.ones_like(pts) * matplotlib.colors.to_rgb(colors)
        elif isinstance(colors, np.ndarray):
            if colors.ndim == 1 and colors.shape[0] == 3:
                colors = np.ones_like(pts) * colors.reshape(1, 3)
            elif colors.ndim == 2 and colors.shape[1] == 3:  # (NPts, 3)
                pass
        else:
            raise ValueError(
                "Unknown color type or shape, "
                "support str #ffffff, list/tuple of 3, np.ndarray of shape (3,) or (NPts, 3)"
            )
        return colors

    def _construct_mesh(self, verts, faces, normals=None, vcolors=None):
        mesh_to_create = o3d.geometry.TriangleMesh()
        mesh_to_create.vertices = o3d.utility.Vector3dVector(verts)
        mesh_to_create.triangles = o3d.utility.Vector3iVector(faces)
        vcolors = self._paint_color_on(verts, vcolors)
        mesh_to_create.vertex_colors = o3d.utility.Vector3dVector(vcolors)
        if normals is not None:
            mesh_to_create.vertex_normals = o3d.utility.Vector3dVector(normals)
        else:
            mesh_to_create.compute_vertex_normals()
            mesh_to_create.compute_triangle_normals()
        return mesh_to_create

    def update_by_mesh(
        self,
        geo_key: str,
        geo=None,
        verts=None,
        faces=None,
        normals=None,
        vcolors=None,
        pose=None,
        update=True,
        reset_bounding_box=True,
    ):
        # already exist and not update
        if self.geometry_store.get(geo_key) and not update:
            return

        if self.geometry_store.get(geo_key) is None:  # create content
            # if geo is not None, use user provided geo
            if geo is not None:
                mesh_to_create = geo
            else:
                assert len(verts.shape) == 2 and verts.shape[1] == 3, f"verts.shape: {verts.shape}"
                assert len(faces.shape) == 2 and faces.shape[1] == 3, f"faces.shape: {faces.shape}"
                if normals is not None:
                    assert len(normals.shape) == 2 and normals.shape[1] == 3, f"normals.shape: {normals.shape}"
                mesh_to_create = self._construct_mesh(verts, faces, normals, vcolors)

            if pose is not None:
                mesh_to_create.transform(pose)
            self.geometry_store[geo_key] = mesh_to_create  # save to dict
            self.geometry_status[geo_key] = True
            self.geometry_curr_pose[geo_key] = pose if pose is not None else np.eye(4)
            self.viz_ctx.add_geometry(mesh_to_create, reset_bounding_box=reset_bounding_box)
        else:  # update content
            mesh_to_update = self.geometry_store[geo_key]  # retrieve from dict, type: o3d.geometry.TriangleMesh
            if geo is not None:
                if mesh_to_update != geo:
                    if self.geometry_status[geo_key]:
                        self.viz_ctx.remove_geometry(mesh_to_update, reset_bounding_box=False)
                    self.geometry_store[geo_key] = geo
                    if self.geometry_status[geo_key]:
                        self.viz_ctx.add_geometry(geo, reset_bounding_box=False)
            else:
                if verts is not None:
                    mesh_to_update.vertices = o3d.utility.Vector3dVector(verts)
                if faces is not None:
                    mesh_to_update.triangles = o3d.utility.Vector3iVector(faces)
                if vcolors is not None:
                    _v = verts if verts is not None else np.asarray(mesh_to_update.vertices)
                    vcolors = self._paint_color_on(_v, vcolors)
                    mesh_to_update.vertex_colors = o3d.utility.Vector3dVector(vcolors)
                if normals is not None:
                    mesh_to_update.vertex_normals = o3d.utility.Vector3dVector(normals)
                else:
                    mesh_to_update.compute_vertex_normals()
                    mesh_to_update.compute_triangle_normals()
                if pose is not None:
                    pose_prev = self.geometry_curr_pose[geo_key]
                    _delta_pose = pose @ inv_transf_np(pose_prev)
                    mesh_to_update.transform(_delta_pose)
                    self.geometry_curr_pose[geo_key] = pose

                if self.geometry_status[geo_key]:
                    self.viz_ctx.update_geometry(mesh_to_update)

    def _construct_pc(self, pcs, normals=None, pcolors=None):
        pc_to_creat = o3d.geometry.PointCloud()
        pc_to_creat.points = o3d.utility.Vector3dVector(pcs)
        pcolors = self._paint_color_on(pcs, pcolors)
        pc_to_creat.colors = o3d.utility.Vector3dVector(pcolors)
        if normals is not None:
            pc_to_creat.normals = o3d.utility.Vector3dVector(normals)
        return pc_to_creat

    def update_by_pc(
        self,
        geo_key: str,
        geo=None,
        points=None,
        normals=None,
        pcolors=None,
        update=True,
        reset_bounding_box=True,
    ):
        # already exist and not update
        if self.geometry_store.get(geo_key) and not update:
            return

        if self.geometry_store.get(geo_key) is None:  # create content
            if geo is not None:
                pc_to_creat = geo
            else:
                assert len(points.shape) == 2 and points.shape[1] == 3, f"pcs.shape: {points.shape}"
                if normals is not None:
                    assert len(normals.shape) == 2 and normals.shape[1] == 3, f"normals.shape: {normals.shape}"
                pc_to_creat = self._construct_pc(points, normals, pcolors)

            self.geometry_store[geo_key] = pc_to_creat  # save to dict
            self.geometry_status[geo_key] = True
            self.geometry_curr_pose[geo_key] = np.eye(4)
            self.viz_ctx.add_geometry(pc_to_creat, reset_bounding_box=reset_bounding_box)
        else:  # update content
            pcs_to_update = self.geometry_store[geo_key]  # retrieve from dict, type: o3d.geometry.PointCloud
            if geo is not None:
                if pcs_to_update != geo:
                    if self.geometry_status[geo_key]:
                        self.viz_ctx.remove_geometry(pcs_to_update, reset_bounding_box=False)
                    self.geometry_store[geo_key] = geo
                    if self.geometry_status[geo_key]:
                        self.viz_ctx.add_geometry(geo, reset_bounding_box=False)
            else:
                if points is not None:
                    pcs_to_update.points = o3d.utility.Vector3dVector(points)
                if pcolors is not None:
                    _pts = points if points is not None else np.asarray(pcs_to_update.points)
                    pcolors = self._paint_color_on(_pts, pcolors)
                    pcs_to_update.colors = o3d.utility.Vector3dVector(pcolors)
                if normals is not None:
                    pcs_to_update.normals = o3d.utility.Vector3dVector(normals)

                if self.geometry_status[geo_key]:
                    self.viz_ctx.update_geometry(pcs_to_update)

    # def update_by_lineset

    # remove geo
    def remove_geo(self, geo_key: str):
        if self.geometry_store.get(geo_key) is None:
            return

        geo = self.geometry_store.pop(geo_key)
        status = self.geometry_status.pop(geo_key)
        self.geometry_curr_pose.pop(geo_key)
        if status:
            self.viz_ctx.remove_geometry(geo, reset_bounding_box=False)

    # low-lvl management
    def ctx_add_geometry(self, pc, reset_bounding_box=True):
        self.viz_ctx.add_geometry(pc, reset_bounding_box=reset_bounding_box)

    def ctx_add_geometry_list(self, pc_list, reset_bounding_box=False):
        for pc in pc_list:
            self.viz_ctx.add_geometry(pc, reset_bounding_box=reset_bounding_box)

    def ctx_update_geometry(self, pc):
        self.viz_ctx.update_geometry(pc)

    def ctx_update_geometry_list(self, pc_list):
        for pc in pc_list:
            self.viz_ctx.update_geometry(pc)

    # TODO multiframe
    # def multiframe_init(self):

    # def multiframe_deinit(self):

    # def multiframe_setup_geo(self, geo_key, mode):

    # def multiframe_update_geo(self, geo_key, arr, frame_id):

    # def multiframe_defered_update(self):

    # def multiframe_prev(self):

    # def multiframe_next(self):
