import os
import numpy as np
import torch
import json
from copy import deepcopy
import pickle

import logging
import argparse
from config_reg import ConfigRegistry, ConfigEntrySource
from config_reg import (
    ConfigEntryCommandlineBoolPattern,
    ConfigEntryCommandlineSeqPattern,
)
from config_reg.callback import abspath_callback
from dev_fn.upkeep import log
from dev_fn.upkeep import ckpt
from dev_fn.upkeep.opt import argdict_to_string
from oakink2_tamf.util import log_suppress
from oakink2_tamf.launch.param import reg_base_param, reg_mano_param
from dev_fn.util.console_io import suppress_trimesh_logging
from dev_fn.transform.rotation_np import rot6d_to_rotmat_np, rotmat_to_quat_np
from dev_fn.transform.transform_np import tslrot6d_to_transf_np, transf_point_array_np

from oakink2_tamf.dataset.interaction_segment import InteractionSegmentData
import random
from dev_fn.util import random_util
from dev_fn.viz.control import VizControl
from manotorch.manolayer import ManoLayer, MANOOutput

_logger = logging.getLogger(__name__)

PROG = os.path.splitext(os.path.basename(__file__))[0]
MOCAP_WS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))

PARAM_PREFIX__DATA = "data"
PARAM_PREFIX__MANO = "mano"
PARAM_PREFIX__INFER = "infer"
PARAM_PREFIX__RUNTIME = "runtime"


def reg_entry(config_reg: ConfigRegistry):
    # override default
    config_reg.meta_info["exp_id"].default = "main"

    # base
    reg_base_param(config_reg, PARAM_PREFIX__DATA, MOCAP_WS_DIR)

    # mano
    reg_mano_param(config_reg, PARAM_PREFIX__MANO, MOCAP_WS_DIR)


def reg_extract(config_reg: ConfigRegistry):
    res = {}
    for _p in [PARAM_PREFIX__DATA, PARAM_PREFIX__MANO, PARAM_PREFIX__INFER, PARAM_PREFIX__RUNTIME]:
        try:
            res[_p] = config_reg.select(_p)
        except KeyError:
            pass
    return res


def _face_lh(mano_layer_lh):
    _close_faces = torch.Tensor([
        [92, 38, 122],
        [234, 92, 122],
        [239, 234, 122],
        [279, 239, 122],
        [215, 279, 122],
        [215, 122, 118],
        [215, 118, 117],
        [215, 117, 119],
        [215, 119, 120],
        [215, 120, 108],
        [215, 108, 79],
        [215, 79, 78],
        [215, 78, 121],
        [214, 215, 121],
    ])
    _th_closed_faces = torch.cat([mano_layer_lh.th_faces.clone().detach().cpu(), _close_faces[:, [2, 1, 0]].long()])
    hand_faces_lh = _th_closed_faces.cpu().numpy()
    return hand_faces_lh


def main():
    config_reg = ConfigRegistry(prog=PROG)
    ckpt.reg_entry(config_reg)
    reg_entry(config_reg)

    parser = argparse.ArgumentParser(prog=PROG)
    config_reg.hook(parser)
    config_reg.parse(parser)

    ckpt_cfg = ckpt.reg_extract(config_reg)
    run_cfg = reg_extract(config_reg)

    ckpt.ckpt_setup(ckpt_cfg)
    ckpt.ckpt_opt(ckpt_cfg, ckpt=ckpt_cfg, run=run_cfg)

    log.log_init()
    log.enable_console()
    _logger.info("ckpt_cfg: %s", argdict_to_string(ckpt_cfg))
    _logger.info("run_cfg: %s", argdict_to_string(run_cfg))
    log_suppress.suppress()
    suppress_trimesh_logging()

    # run
    commit = ckpt_cfg["commit"]
    ckpt_path = ckpt_cfg["ckpt_path"]

    # mano
    dtype = torch.float32
    device = torch.device("cpu")

    mano_layer_rh = ManoLayer(
        mano_assets_root=run_cfg["mano"]["mano_path"],
        rot_mode="quat",
        side="right",
        center_idx=0,
        use_pca=False,
        flat_hand_mean=True,
    ).to(device)
    hand_faces_rh = mano_layer_rh.get_mano_closed_faces().cpu().numpy()
    mano_layer_lh = ManoLayer(
        mano_assets_root=run_cfg["mano"]["mano_path"],
        rot_mode="quat",
        side="left",
        center_idx=0,
        use_pca=False,
        flat_hand_mean=True,
    ).to(device)
    hand_faces_lh = _face_lh(mano_layer_lh)

    # dataset
    process_range_list = run_cfg["data"]["process_range"]
    all_dataset = InteractionSegmentData(
        process_range_list=process_range_list[0:4],
        data_prefix=run_cfg["data"]["data_prefix"],
        enable_obj_model=True,
    )

    viz_control = VizControl()
    viz_control.attach_viz_ctx()

    info_set = set()
    for item in all_dataset:
        info = item['info']
        if info in info_set:
            continue
        info_set.add(info)

        print(item['text'])

        obj_verts = item["obj_verts"]
        obj_faces = item["obj_faces"]

        seg_len = item['len']
        hand_side = item['hand_side']

        hand_pose_repr = item['pose_repr']
        hand_tsl = hand_pose_repr[:, 0:3]
        hand_pose = rotmat_to_quat_np(rot6d_to_rotmat_np(hand_pose_repr[:, 3:99].reshape((-1, 16, 6))))
        hand_shape = item['shape']

        layer = mano_layer_rh if hand_side == "rh" else mano_layer_lh

        hand_verts = layer(torch.tensor(hand_pose, dtype=dtype, device=device),
                           torch.tensor(hand_shape, dtype=dtype, device=device)).verts.cpu().numpy()
        hand_verts = hand_verts + np.expand_dims(hand_tsl, axis=1)
        hand_faces = hand_faces_rh if hand_side == "rh" else hand_faces_lh
        obj_num = item['obj_num']
        obj_list = item['obj_list']
        obj_traj = item['obj_traj']
        obj_traj_transf = tslrot6d_to_transf_np(obj_traj)

        for offset in range(seg_len):
            for ooff in range(obj_num):
                viz_control.update_by_mesh(obj_list[ooff],
                                           verts=transf_point_array_np(obj_traj_transf[ooff, offset], obj_verts[ooff]),
                                           faces=obj_faces[ooff])
            viz_control.update_by_mesh("hand", verts=hand_verts[offset], faces=hand_faces)

            # viz_control.condition_reset()
            # while viz_control.condition():
            viz_control.step()

        for ooff in range(obj_num):
            viz_control.remove_geo(obj_list[ooff])
        viz_control.remove_geo("hand")

    viz_control.detach_viz_ctx()


if __name__ == "__main__":
    main()
