from __future__ import annotations
import typing

if typing.TYPE_CHECKING:
    from typing import Optional

import os
import numpy as np
import pickle
import logging

import torch

from oakink2_toolkit.dataset import OakInk2__Dataset
from oakink2_toolkit.meta import FPS_MOCAP, HAND_SIDE, HAND_SIDE_MAP
from dev_fn.transform.rotation import quat_to_rotmat
from dev_fn.transform.rotation_np import rotmat_to_rot6d_np, rot6d_to_rotmat_np
from dev_fn.transform.transform_np import transf_to_tslrot6d_np
import manotorch
import itertools

import tqdm
from dev_fn.util import pbar_util

from .setment_slice import SegmentSlice

_logger = logging.getLogger(__name__)


class InteractionSegmentData(torch.utils.data.Dataset):

    def collect_obj(self, primitive_task_data, src_obj_list, seg_beg, seg_end, task_beg):
        seg_obj_traj_store = {}
        for obj_id in src_obj_list:
            _off_beg, _off_end = seg_beg - task_beg, seg_end - task_beg
            seg_obj_traj_store[obj_id] = primitive_task_data.obj_transf[obj_id][_off_beg:_off_end].astype(np.float32)
        return seg_obj_traj_store

    def collect_mano(self, primitive_task_data, hand_side, seg_beg, seg_end, task_beg):
        _in_range_mask = primitive_task_data[f"{hand_side}_in_range_mask"]
        _pose = primitive_task_data[f"{hand_side}_param"]["pose_coeffs"][_in_range_mask]
        _tsl = primitive_task_data[f"{hand_side}_param"]["tsl"][_in_range_mask]
        _shape = primitive_task_data[f"{hand_side}_param"]["betas"][_in_range_mask]
        assert _pose.shape[0] == seg_end - seg_beg
        assert _tsl.shape[0] == seg_end - seg_beg
        assert _shape.shape[0] == seg_end - seg_beg
        seg_mano_pose_traj = _pose
        seg_mano_tsl_traj = _tsl
        seg_mano_shape_traj = _shape
        seg_mano_pose_traj = quat_to_rotmat(seg_mano_pose_traj)
        seg_mano_pose_traj = seg_mano_pose_traj.numpy().astype(np.float32)
        seg_mano_tsl_traj = seg_mano_tsl_traj.numpy().astype(np.float32)
        seg_mano_shape_traj = seg_mano_shape_traj.numpy().astype(np.float32)
        return seg_mano_pose_traj, seg_mano_tsl_traj, seg_mano_shape_traj

    def load_dataset(self, rank: Optional[int] = None):
        interaction_segment_info_list = []
        interaction_segment_len_list = []
        interaction_segment_pose_list = []
        interaction_segment_tsl_list = []
        interaction_segment_shape_list = []
        interaction_segment_hand_side_list = []
        interaction_segment_text_list = []
        interaction_segment_obj_traj_list = []
        interaction_segment_frame_id_list = []
        interaction_object_list = None
        object_list = set()
        pbar = (tqdm.tqdm(
            total=len(self.process_range_list), position=0, bar_format=pbar_util.fmt, desc="collect segment:")
                if not rank else pbar_util.dummy_pbar())
        for process_key in self.process_range_list:
            complex_task_data = self.dataset.load_complex_task(seq_key=process_key)
            primitive_task_data_list = self.dataset.load_primitive_task(complex_task_data=complex_task_data)
            from dev_fn.util.console_io import pprint

            for primitive_identifier, primitive_task_data in zip(complex_task_data.exec_path, primitive_task_data_list):
                task_beg = primitive_task_data.frame_range[0]

                for hand_side in HAND_SIDE:
                    if primitive_task_data.hand_involved not in ["bh", hand_side]:
                        continue

                    text_desc = primitive_task_data.task_desc
                    seg_beg, seg_end = primitive_task_data[f"frame_range_{hand_side}"]
                    seg_info = (process_key, primitive_identifier, hand_side)
                    seg_len = seg_end - seg_beg

                    src_obj_list = primitive_task_data[f"{hand_side}_obj_list"]
                    if len(src_obj_list) == 0:
                        continue

                    # collect obj
                    object_list.update(src_obj_list)
                    seg_obj_traj_store = self.collect_obj(primitive_task_data, src_obj_list, seg_beg, seg_end, task_beg)
                    # collect mano
                    seg_mano_pose_traj, seg_mano_tsl_traj, seg_mano_shape_traj = self.collect_mano(
                        primitive_task_data, hand_side, seg_beg, seg_end, task_beg)
                    # slice with target fps
                    seg_mano_pose_traj_sliced, seg_len_sliced = SegmentSlice.from_gap(
                        seg_mano_pose_traj, self.target_gap, self.slice_max_len, self.slice_min_len)
                    seg_mano_tsl_traj_sliced, _ = SegmentSlice.from_gap(seg_mano_tsl_traj, self.target_gap,
                                                                        self.slice_max_len, self.slice_min_len)
                    seg_mano_shape_traj_sliced, _ = SegmentSlice.from_gap(seg_mano_shape_traj, self.target_gap,
                                                                          self.slice_max_len, self.slice_min_len)
                    seg_obj_traj_store_sliced = {}
                    for obj_id in src_obj_list:
                        seg_obj_traj_store_sliced[obj_id], _ = SegmentSlice.from_gap(
                            seg_obj_traj_store[obj_id], self.target_gap, self.slice_max_len, self.slice_min_len)
                    # reformat with seg_len_sliced
                    seg_obj_traj_list_sliced = []
                    for _offset in range(len(seg_len_sliced)):
                        _item = {}
                        for obj_id in src_obj_list:
                            _item[obj_id] = seg_obj_traj_store_sliced[obj_id][_offset]
                        seg_obj_traj_list_sliced.append(_item)
                    # frame_id
                    seg_frame_id_list = np.array(list(range(seg_beg, seg_end)))
                    seg_frame_id_list_sliced, _ = SegmentSlice.from_gap(seg_frame_id_list, self.target_gap,
                                                                        self.slice_max_len, self.slice_min_len)
                    _seg_frame_id_list_sliced = []
                    for _len, _fid_list in zip(seg_len_sliced, seg_frame_id_list_sliced):
                        _new_list = _fid_list[:_len].tolist()
                        _seg_frame_id_list_sliced.append(_new_list)
                    seg_frame_id_list_sliced = _seg_frame_id_list_sliced
                    # extend storage list
                    interaction_segment_info_list.extend([seg_info] * len(seg_len_sliced))
                    interaction_segment_len_list.extend(seg_len_sliced)
                    interaction_segment_pose_list.extend(seg_mano_pose_traj_sliced)
                    interaction_segment_tsl_list.extend(seg_mano_tsl_traj_sliced)
                    interaction_segment_shape_list.extend(seg_mano_shape_traj_sliced)
                    interaction_segment_hand_side_list.extend([hand_side] * len(seg_len_sliced))
                    interaction_segment_text_list.extend([text_desc] * len(seg_len_sliced))
                    interaction_segment_obj_traj_list.extend(seg_obj_traj_list_sliced)
                    interaction_segment_frame_id_list.extend(seg_frame_id_list_sliced)

            pbar.update()
        pbar.close()
        interaction_object_list = sorted(object_list)

        # debug
        # print(len(interaction_segment_len_list), interaction_segment_len_list[:10])
        # print(len(interaction_segment_hand_side_list), interaction_segment_len_list[:10])
        # print(len(interaction_segment_obj_traj_list))
        # for el in interaction_segment_tsl_list[:10]:
        #     print(el.shape)

        return (
            interaction_segment_info_list,
            interaction_segment_len_list,
            interaction_segment_pose_list,
            interaction_segment_tsl_list,
            interaction_segment_shape_list,
            interaction_segment_hand_side_list,
            interaction_segment_text_list,
            interaction_segment_obj_traj_list,
            interaction_segment_frame_id_list,
            interaction_object_list,
        )

    def load_reverse_segment(self):
        (
            interaction_segment_info_list,
            interaction_segment_len_list,
            interaction_segment_pose_list,
            interaction_segment_tsl_list,
            interaction_segment_shape_list,
            interaction_segment_hand_side_list,
            interaction_segment_text_list,
            interaction_segment_obj_traj_list,
            interaction_segment_frame_id_list,
            interaction_object_list,
        ) = (
            self.interaction_segment_info_list,
            self.interaction_segment_len_list,
            self.interaction_segment_pose_list,
            self.interaction_segment_tsl_list,
            self.interaction_segment_shape_list,
            self.interaction_segment_hand_side_list,
            self.interaction_segment_text_list,
            self.interaction_segment_obj_traj_list,
            self.interaction_segment_frame_id_list,
            self.interaction_object_list,
        )

        def rev_prefix(arr, length):
            res_arr = arr.copy()
            avai_arr = arr[:length]
            rev_avai_arr = avai_arr[::-1]
            res_arr[:length] = rev_avai_arr
            return res_arr

        (
            _info_list,
            _len_list,
            _pose_list,
            _tsl_list,
            _shape_list,
            _hand_side_list,
            _text_list,
            _obj_traj_list,
            _fid_list,
        ) = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for seg_info, seg_len, seg_pose, seg_tsl, seg_shape, seg_hand_side, seg_text, seg_obj_traj, seg_fid_list in zip(
                self.interaction_segment_info_list,
                self.interaction_segment_len_list,
                self.interaction_segment_pose_list,
                self.interaction_segment_tsl_list,
                self.interaction_segment_shape_list,
                self.interaction_segment_hand_side_list,
                self.interaction_segment_text_list,
                self.interaction_segment_obj_traj_list,
                self.interaction_segment_frame_id_list,
        ):
            # reverse pose, tsl, shape, obj_traj within seg_len
            rev_seg_pose = rev_prefix(seg_pose, seg_len)
            rev_seg_tsl = rev_prefix(seg_tsl, seg_len)
            rev_seg_shape = rev_prefix(seg_shape, seg_len)
            rev_seg_obj_traj = {}
            for obj_id in seg_obj_traj:
                rev_seg_obj_traj[obj_id] = rev_prefix(seg_obj_traj[obj_id], seg_len)
            rev_seg_fid_list = seg_fid_list[::-1]

            # append
            _info_list.append(seg_info)
            _len_list.append(seg_len)
            _pose_list.append(rev_seg_pose)
            _tsl_list.append(rev_seg_tsl)
            _shape_list.append(rev_seg_shape)
            _hand_side_list.append(seg_hand_side)
            _text_list.append(seg_text)
            _obj_traj_list.append(rev_seg_obj_traj)
            _fid_list.append(rev_seg_fid_list)

        interaction_segment_info_list = interaction_segment_info_list + _info_list
        interaction_segment_len_list = interaction_segment_len_list + _len_list
        interaction_segment_pose_list = interaction_segment_pose_list + _pose_list
        interaction_segment_tsl_list = interaction_segment_tsl_list + _tsl_list
        interaction_segment_shape_list = interaction_segment_shape_list + _shape_list
        interaction_segment_hand_side_list = interaction_segment_hand_side_list + _hand_side_list
        interaction_segment_text_list = interaction_segment_text_list + _text_list
        interaction_segment_obj_traj_list = interaction_segment_obj_traj_list + _obj_traj_list
        interaction_segment_frame_id_list = interaction_segment_frame_id_list + _fid_list

        return (
            interaction_segment_info_list,
            interaction_segment_len_list,
            interaction_segment_pose_list,
            interaction_segment_tsl_list,
            interaction_segment_shape_list,
            interaction_segment_hand_side_list,
            interaction_segment_text_list,
            interaction_segment_obj_traj_list,
            interaction_segment_frame_id_list,
            interaction_object_list,
        )

    def load_object_embedding(self):
        obj_embedding_store = {}
        for obj_id in self.interaction_object_list:
            _filepath = os.path.join(self.obj_embedding_prefix, f"{obj_id}.pt")
            _embed = torch.load(_filepath, map_location=torch.device("cpu"))
            _embed_np = _embed.numpy().astype(np.float32).copy()
            obj_embedding_store[obj_id] = _embed_np
        return obj_embedding_store

    def load_object_pointcloud(self):
        obj_pointcloud_store = {}
        for obj_id in self.interaction_object_list:
            _filepath = os.path.join(self.obj_pointcloud_prefix, f"{obj_id}.npz")
            with np.load(_filepath) as npz_stream:
                pointcloud = npz_stream["point"].astype(np.float32).copy()
            obj_pointcloud_store[obj_id] = pointcloud
        return obj_pointcloud_store

    def __init__(
        self,
        process_range_list: list[str],
        data_prefix: str,
        target_fps: float = 10.0,
        slice_min_len: int = 16,
        slice_max_len: int = 160,
        rank: Optional[int] = None,
        enable_obj_model: bool = False,
        obj_embedding_prefix: Optional[str] = None,
        obj_pointcloud_prefix: Optional[str] = None,
        cache_dict: Optional[dict] = None,
        append_reverse_segment: bool = False,
    ):
        # splits control by process_range_list
        self.process_range_list = process_range_list
        self.data_prefix = data_prefix

        # dataset
        self.dataset = OakInk2__Dataset(dataset_prefix=self.data_prefix, return_instantiated=True)

        # deal with slicing
        self.origin_fps = FPS_MOCAP
        self.target_fps = target_fps
        self.target_gap = int(self.origin_fps // self.target_fps)
        self.slice_min_len = slice_min_len
        self.slice_max_len = slice_max_len

        # load and slice segments
        if cache_dict is None:
            (
                self.interaction_segment_info_list,
                self.interaction_segment_len_list,
                self.interaction_segment_pose_list,
                self.interaction_segment_tsl_list,
                self.interaction_segment_shape_list,
                self.interaction_segment_hand_side_list,
                self.interaction_segment_text_list,
                self.interaction_segment_obj_traj_list,
                self.interaction_segment_frame_id_list,
                self.interaction_object_list,
            ) = self.load_dataset(rank)
        else:
            (
                self.interaction_segment_info_list,
                self.interaction_segment_len_list,
                self.interaction_segment_pose_list,
                self.interaction_segment_tsl_list,
                self.interaction_segment_shape_list,
                self.interaction_segment_hand_side_list,
                self.interaction_segment_text_list,
                self.interaction_segment_obj_traj_list,
                self.interaction_segment_frame_id_list,
                self.interaction_object_list,
            ) = self.load_cache(cache_dict)
        # handle reverse order segments
        self.append_reverse_segment = append_reverse_segment
        if self.append_reverse_segment:
            (
                self.interaction_segment_info_list,
                self.interaction_segment_len_list,
                self.interaction_segment_pose_list,
                self.interaction_segment_tsl_list,
                self.interaction_segment_shape_list,
                self.interaction_segment_hand_side_list,
                self.interaction_segment_text_list,
                self.interaction_segment_obj_traj_list,
                self.interaction_segment_frame_id_list,
                self.interaction_object_list,
            ) = self.load_reverse_segment()
            _logger.info("load reverse segment")

        self.len = len(self.interaction_segment_len_list)
        if not rank:
            _logger.info("collect %d segments", self.len)

        # load object if need to return verts
        self.enable_obj_model = enable_obj_model
        if self.enable_obj_model:
            obj_store = {}
            for obj_id in self.interaction_object_list:
                affordance_data = self.dataset.load_affordance(obj_id)
                obj_store[obj_id] = affordance_data.obj_mesh
            self.obj_store = obj_store
            assert all(el in self.obj_store for el in self.interaction_object_list)
        else:
            self.obj_store = None

        # load object embedding
        self.enable_obj_embedding = obj_embedding_prefix is not None
        self.obj_embedding_prefix = obj_embedding_prefix
        if self.enable_obj_embedding:
            self.obj_embedding_store = self.load_object_embedding()
        else:
            self.obj_embedding_store = None

        # load object pointcloud
        self.enable_obj_pointcloud = obj_pointcloud_prefix is not None
        self.obj_pointcloud_prefix = obj_pointcloud_prefix
        if self.enable_obj_pointcloud:
            self.obj_pointcloud_store = self.load_object_pointcloud()
        else:
            self.obj_pointcloud_store = None

    def __getitem__(self, index):
        segment_info = self.interaction_segment_info_list[index]
        segment_len = self.interaction_segment_len_list[index]
        segment_pose = self.interaction_segment_pose_list[index]
        segment_tsl = self.interaction_segment_tsl_list[index]
        segment_shape = self.interaction_segment_shape_list[index]
        segment_hand_side = self.interaction_segment_hand_side_list[index]
        segment_text = self.interaction_segment_text_list[index]
        segment_obj_traj = self.interaction_segment_obj_traj_list[index]
        segment_fid_list = self.interaction_segment_frame_id_list[index]

        # convert from transf to tslrot6d
        segment_pose_rot6d = rotmat_to_rot6d_np(segment_pose)
        segment_pose_rot6d_vec = segment_pose_rot6d.reshape((segment_pose_rot6d.shape[0], 16 * 6))
        # concat to get tslrot6d
        segment_pose_repr = np.concatenate((segment_tsl, segment_pose_rot6d_vec), axis=-1)

        # handle object trajectory
        obj_list = sorted(segment_obj_traj.keys())
        obj_traj_arr = [transf_to_tslrot6d_np(segment_obj_traj[_oid]) for _oid in obj_list]
        obj_traj_arr = np.stack(obj_traj_arr, axis=0)

        # get mask
        segment_mask = np.ones((self.slice_max_len,), dtype=np.float32)
        segment_mask[segment_len:] = 0.0

        res = {
            "info": segment_info,
            "len": segment_len,
            "mask": segment_mask,
            # "tsl": segment_tsl,
            # "pose": segment_pose,
            "pose_repr": segment_pose_repr,
            "shape": segment_shape,
            "hand_side": segment_hand_side,
            "text": segment_text,
            "obj_list": obj_list,
            "obj_num": len(obj_list),
            "obj_traj": obj_traj_arr,
            "frame_id": segment_fid_list,
        }

        # handle object verts, faces
        if self.enable_obj_model:
            obj_verts = [np.array(self.obj_store[_oid].vertices) for _oid in obj_list]
            obj_faces = [np.array(self.obj_store[_oid].faces) for _oid in obj_list]
            res["obj_verts"] = obj_verts
            res["obj_faces"] = obj_faces

        # handle object repr
        if self.enable_obj_embedding:
            obj_embedding = [self.obj_embedding_store[_oid] for _oid in obj_list]
            obj_embedding = np.stack(obj_embedding, axis=0)
            res["obj_embedding"] = obj_embedding

        # handle object pointcloud
        if self.enable_obj_pointcloud:
            obj_pointcloud = [self.obj_pointcloud_store[_oid] for _oid in obj_list]
            obj_pointcloud = np.stack(obj_pointcloud, axis=0)
            res["obj_pointcloud"] = obj_pointcloud
        return res

    def __len__(self):
        return self.len

    def get_cache(self):
        return {
            "interaction_segment_info_list": self.interaction_segment_info_list,
            "interaction_segment_len_list": self.interaction_segment_len_list,
            "interaction_segment_pose_list": self.interaction_segment_pose_list,
            "interaction_segment_tsl_list": self.interaction_segment_tsl_list,
            "interaction_segment_shape_list": self.interaction_segment_shape_list,
            "interaction_segment_hand_side_list": self.interaction_segment_hand_side_list,
            "interaction_segment_text_list": self.interaction_segment_text_list,
            "interaction_segment_obj_traj_list": self.interaction_segment_obj_traj_list,
            "interaction_segment_frame_id_list": self.interaction_segment_frame_id_list,
            "interaction_object_list": self.interaction_object_list,
        }

    def load_cache(self, cache_dict):
        return (
            cache_dict["interaction_segment_info_list"],
            cache_dict["interaction_segment_len_list"],
            cache_dict["interaction_segment_pose_list"],
            cache_dict["interaction_segment_tsl_list"],
            cache_dict["interaction_segment_shape_list"],
            cache_dict["interaction_segment_hand_side_list"],
            cache_dict["interaction_segment_text_list"],
            cache_dict["interaction_segment_obj_traj_list"],
            cache_dict["interaction_segment_frame_id_list"],
            cache_dict["interaction_object_list"],
        )
