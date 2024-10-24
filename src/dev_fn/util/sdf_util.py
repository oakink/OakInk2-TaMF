import numpy as np
import trimesh
from pysdf import SDF
import pickle
import skimage
import skimage.measure

import dataclasses
from dataclasses import dataclass, fields


@dataclass
class NamedData:

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __iter__(self):
        return self.values()

    def keys(self):
        keys = [t.name for t in fields(self)]
        return iter(keys)

    def values(self):
        values = [getattr(self, t.name) for t in fields(self)]
        return iter(values)

    def items(self):
        data = [(t.name, getattr(self, t.name)) for t in fields(self)]
        return iter(data)


@dataclasses.dataclass
class SDFData(NamedData):
    mesh_center: np.ndarray
    bbox: np.ndarray
    bbox_centered: np.ndarray
    bbox_centered_expanded: np.ndarray
    bbox_expanded: np.ndarray

    bbox_expand_ratio: float
    resolution: int

    extent: np.ndarray
    extent_expanded: np.ndarray
    tick_unit: np.ndarray

    point: np.ndarray
    sdf: np.ndarray


def process_sdf(obj_mesh: trimesh.Trimesh, bbox_expand_ratio=1.2, resolution=100):
    obj_mesh_aabb = obj_mesh.bounding_box.vertices
    obj_mesh_center = np.mean(obj_mesh_aabb, axis=0)
    obj_mesh_aabb_centered = obj_mesh_aabb - obj_mesh_center
    obj_mesh_aabb_centered_expanded = obj_mesh_aabb_centered * bbox_expand_ratio
    obj_mesh_aabb_expanded = obj_mesh_aabb_centered_expanded + obj_mesh_center

    # centerize the mesh
    obj_mesh.vertices -= obj_mesh_center
    obj_mesh_extent = np.asarray(obj_mesh.bounding_box.extents)
    obj_mesh_extent_expanded = obj_mesh_extent * bbox_expand_ratio
    obj_tick_unit = obj_mesh_extent_expanded / resolution

    # get tick
    tick = np.linspace(-obj_mesh_extent_expanded / 2.0, obj_mesh_extent_expanded / 2.0, resolution)
    x, y, z = np.meshgrid(tick[:, 0], tick[:, 1], tick[:, 2], indexing="ij")
    query_point = np.vstack((x.flatten(), y.flatten(), z.flatten())).T

    # compute sdf
    f = SDF(obj_mesh.vertices, obj_mesh.faces, robust=True)  # negative outside
    query_sdf = f(query_point)

    # offset to object frame
    obj_frame_query_point = query_point + obj_mesh_center

    # construct res
    res = SDFData(
        mesh_center=obj_mesh_center,
        bbox=np.asarray(obj_mesh_aabb),
        bbox_centered=np.asarray(obj_mesh_aabb_centered),
        bbox_centered_expanded=np.asarray(obj_mesh_aabb_centered_expanded),
        bbox_expanded=np.asarray(obj_mesh_aabb_expanded),
        bbox_expand_ratio=bbox_expand_ratio,
        resolution=resolution,
        extent=obj_mesh_extent,
        extent_expanded=obj_mesh_extent_expanded,
        tick_unit=obj_tick_unit,
        point=np.asarray(obj_frame_query_point),
        sdf=np.asarray(query_sdf),
    )
    return res


@dataclasses.dataclass
class SDFReconData(NamedData):
    vert: np.ndarray
    face: np.ndarray
    normal: np.ndarray
    value: np.ndarray


def reconstruct_sdf(sdf, obj_mesh_center, obj_mesh_extent_expanded, resolution):
    obj_tick_unit = obj_mesh_extent_expanded / resolution
    vert, face, normal, value = skimage.measure.marching_cubes(
        sdf.reshape(resolution, resolution, resolution),
        0.0,
        spacing=obj_tick_unit,
        step_size=1,
        allow_degenerate=False,
        method="lewiner",
    )
    vert = vert - obj_mesh_extent_expanded / 2.0 + obj_mesh_center
    # invert order of face vertex
    face = face[:, [2, 1, 0]]

    res = SDFReconData(
        vert=vert,
        face=face,
        normal=normal,
        value=value,
    )
    return res


def load_sdf_data(filepath):
    with open(filepath, "rb") as bytestream:
        dict_sdf_data = pickle.load(bytestream)
    sdf_data = SDFData(**dict_sdf_data)
    return sdf_data
