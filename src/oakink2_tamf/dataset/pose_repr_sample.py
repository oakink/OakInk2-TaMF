from __future__ import annotations
import typing

if typing.TYPE_CHECKING:
    from typing import Optional

import os
import numpy as np
import pickle
import logging

import torch


_logger = logging.getLogger(__name__)


class GeneratedPoseReprSampleAdaptor(torch.utils.data.Dataset):
    def __init__(self, interaction_segment_dataset, dir_list: list[str]):
        super().__init__()
        self.interaction_segment_dataset = interaction_segment_dataset

        self.dir_list = dir_list
        pose_repr_info_list = []
        pose_repr_map = {}
        for dir_path in self.dir_list:
            dir_basename = os.path.basename(dir_path)
            for filename in sorted(el for el in os.listdir(dir_path) if os.path.splitext(el)[-1] == ".npy"):
                sample_id = int(os.path.splitext(filename)[0])
                npy_filepath = os.path.join(dir_path, filename)
                sample_pose_repr = np.load(npy_filepath)
                info_tuple = (dir_basename, sample_id)

                pose_repr_info_list.append(info_tuple)
                pose_repr_map[info_tuple] = sample_pose_repr

        assert len(pose_repr_info_list) == len(self.interaction_segment_dataset)

        self.pose_repr_info_list = pose_repr_info_list
        self.pose_repr_map = pose_repr_map
        self.len = len(self.pose_repr_info_list)

    def __getitem__(self, index):
        base_data = self.interaction_segment_dataset[index]
        pose_repr_info = self.pose_repr_info_list[index]
        pose_repr_sample = self.pose_repr_map[pose_repr_info]
        base_data["sample_info"] = pose_repr_info
        base_data["sample_pose_repr"] = pose_repr_sample
        return base_data

    def __len__(self):
        return self.len


class GuassianPerturbSampleAdaptor(torch.utils.data.Dataset):
    def __init__(self, interaction_segment_dataset, sigma_range):
        self.interaction_segment_dataset = interaction_segment_dataset
        self.sigma_min = float(sigma_range[0])
        self.sigma_max = float(sigma_range[1])

    def __getitem__(self, index):
        base_data = self.interaction_segment_dataset[index]
        pose_repr = base_data["pose_repr"]
        avail_len = base_data["len"]

        # select sigma
        sigma = np.random.uniform(self.sigma_min, self.sigma_max)
        # use sigma to create zero mean, sigma var noise same shape as pose_repr
        noise_tsl = np.random.normal(0, 0.1 * sigma, size=(avail_len, 3))
        noise_pose = np.random.normal(0, sigma, size=(avail_len, 96))
        sample_info = (index, sigma)
        sample_pose_repr = pose_repr.copy()
        sample_pose_repr[:avail_len, 0:3] += noise_tsl
        sample_pose_repr[:avail_len, 3:99] += noise_pose

        # normalize
        seq_len = pose_repr.shape[0]
        pose_rot6d = sample_pose_repr[:avail_len, 3:99]
        pose_rot6d = pose_rot6d.reshape((avail_len, 16, 6))

        pose_rot6d_a = pose_rot6d[..., 0:3]
        pose_rot6d_a = pose_rot6d_a / np.maximum(np.linalg.norm(pose_rot6d_a, axis=-1, keepdims=True), 1e-7)
        pose_rot6d_b = pose_rot6d[..., 3:6]
        pose_rot6d_b = pose_rot6d_b / np.maximum(np.linalg.norm(pose_rot6d_b, axis=-1, keepdims=True), 1e-7)
        pose_rot6d = np.concatenate((pose_rot6d_a, pose_rot6d_b), axis=-1)
        sample_pose_repr[:avail_len, 3:99] = pose_rot6d.reshape((avail_len, 96))

        base_data["sample_info"] = sample_info
        base_data["sample_pose_repr"] = sample_pose_repr
        return base_data

    def __len__(self):
        return len(self.interaction_segment_dataset)


class IdentitySampleAdaptor(torch.utils.data.Dataset):
    def __init__(self, interaction_segment_dataset):
        super().__init__()
        self.interaction_segment_dataset = interaction_segment_dataset
        self.len = len(self.interaction_segment_dataset)

    def __getitem__(self, index):
        base_data = self.interaction_segment_dataset[index]
        base_data["sample_info"] = None
        base_data["sample_pose_repr"] = base_data["pose_repr"]
        return base_data

    def __len__(self):
        return self.len