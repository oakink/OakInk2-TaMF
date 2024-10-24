from __future__ import annotations
import typing

if typing.TYPE_CHECKING:
    from typing import Optional, Union

import numpy as np


def segment_slice_from_gap(traj: np.ndarray, gap: int, max_len: int, min_len: int):
    res = []
    res_len = []
    # process
    # the first dimension is to slice!
    traj_len = int(traj.shape[0])
    # determine mode: using gap, using max_len or using min_len
    if traj_len < min_len * gap:
        gap = traj_len // min_len
    elif traj_len > max_len * gap:
        gap = (traj_len + max_len - 1) // max_len
    else:  # min_len * gap <= traj_len <= max_len * gap
        pass
    # slice
    for _offset in range(gap):
        sliced_traj = traj[_offset::gap]
        sliced_traj_len = sliced_traj.shape[0]
        assert min_len <= sliced_traj_len <= max_len
        # pad to max_len
        if sliced_traj_len < max_len:
            pad_len = max_len - sliced_traj_len
            pad = np.zeros((pad_len, *sliced_traj.shape[1:]), dtype=sliced_traj.dtype)
            sliced_traj = np.concatenate([sliced_traj, pad], axis=0)
        res.append(sliced_traj)
        res_len.append(sliced_traj_len)
    return res, res_len


class SegmentSlice:
    from_gap = segment_slice_from_gap
