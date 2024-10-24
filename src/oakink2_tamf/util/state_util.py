import os
import torch
from typing import Sequence


def _need_filter_out(key, filter_out):
    if filter_out is None:
        return False

    if isinstance(filter_out, str):
        return key.startswith(filter_out)
    elif isinstance(filter_out, Sequence):
        for fltr in filter_out:
            if key.startswith(fltr):
                return True
        else:
            return False
    else:
        raise TypeError(f"unexpected filter out: {filter_out}")


def save_state(state_dict, filename, remove_prefix=None, filter_out=None):
    res = {}
    for k, v in state_dict.items():
        if remove_prefix is not None:
            _name_comp = k.split(".")
            _r_prefix_comp = remove_prefix.split(".")
            # check prefix match
            if _name_comp[: len(_r_prefix_comp)] == _r_prefix_comp:
                _name_comp = _name_comp[len(_r_prefix_comp) :]
                k = ".".join(_name_comp)
        if filter_out is not None:
            if _need_filter_out(k, filter_out):
                continue
        res[k] = v

    # save this to path
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(res, filename)
