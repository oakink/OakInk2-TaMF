import numpy as np
import torch

from torch.utils.data import default_collate

DEFAULT_COLLATE_KEY = [
    "pose_repr",
    "pose_repr_lh",
    "pose_repr_rh",
    "shape",
    "shape_lh",
    "shape_rh",
    "len",
    "mask",
    "obj_num",
    "sample_pose_repr",
]
NO_COLLATE_KEY = [
    "hand_side",
    "text",
    "obj_list",
    "info",
    "obj_verts",
    "obj_faces",
    "obj_pointcloud",
    "sample_info",
    "frame_id",
]
PAD_COLLATE_KEY = ["obj_traj", "obj_embedding"]
DROP_KEY = []


def interaction_segment_collate(batch):
    key_list = list(next(iter(batch)).keys())
    res = {}
    for key in key_list:
        if key in DEFAULT_COLLATE_KEY:
            res[key] = default_collate([_b[key] for _b in batch])
        elif key in NO_COLLATE_KEY:
            res[key] = [_b[key] for _b in batch]
        elif key in PAD_COLLATE_KEY:
            # find max dimension
            max_dim = max([_b[key].shape[0] for _b in batch])
            # zero pad at zero dim
            _blist = []
            for _b in batch:
                _bitem = _b[key]
                if _bitem.shape[0] < max_dim:
                    pad_len = max_dim - _bitem.shape[0]
                    pad = np.zeros((pad_len, *_bitem.shape[1:]), dtype=_bitem.dtype)
                    _bitem = np.concatenate((_bitem, pad), axis=0)
                _blist.append(_bitem)
            res[key] = default_collate(_blist)
        elif key in DROP_KEY:
            pass
        else:
            raise KeyError(f"unexpected key in batch! got {key}")
    return res


class SegmentCollate:
    def __init__(
        self, use_default=True, extra_default_key=None, extra_no_key=None, extra_pad_key=None, extra_drop_key=None
    ) -> None:
        if extra_default_key is None:
            extra_default_key = []
        if extra_no_key is None:
            extra_no_key = []
        if extra_pad_key is None:
            extra_pad_key = []
        if extra_drop_key is None:
            extra_drop_key = []

        self.default_collate_key = set(DEFAULT_COLLATE_KEY) if use_default else set()
        self.no_collate_key = set(NO_COLLATE_KEY) if use_default else set()
        self.pad_collate_key = set(PAD_COLLATE_KEY) if use_default else set()
        self.drop_key = set(DROP_KEY) if use_default else set()

        self.default_collate_key.update(extra_default_key)
        self.no_collate_key.update(extra_no_key)
        self.pad_collate_key.update(extra_pad_key)
        self.drop_key.update(extra_drop_key)

    def __call__(self, batch):
        key_list = list(next(iter(batch)).keys())
        res = {}
        for key in key_list:
            if key in self.default_collate_key:
                res[key] = default_collate([_b[key] for _b in batch])
            elif key in self.no_collate_key:
                res[key] = [_b[key] for _b in batch]
            elif key in self.pad_collate_key:
                # find max dimension
                max_dim = max([_b[key].shape[0] for _b in batch])
                # zero pad at zero dim
                _blist = []
                for _b in batch:
                    _bitem = _b[key]
                    if _bitem.shape[0] < max_dim:
                        pad_len = max_dim - _bitem.shape[0]
                        pad = np.zeros((pad_len, *_bitem.shape[1:]), dtype=_bitem.dtype)
                        _bitem = np.concatenate((_bitem, pad), axis=0)
                    _blist.append(_bitem)
                res[key] = default_collate(_blist)
            elif key in self.drop_key:
                pass
            else:
                raise KeyError(f"unexpected key in batch! got {key}")
        return res
