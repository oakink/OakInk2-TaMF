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
from oakink2_tamf.launch.param import reg_mano_param, reg_model_param
from dev_fn.util.console_io import suppress_trimesh_logging

from oakink2_tamf.dataset.interaction_segment import InteractionSegmentData
from dev_fn.transform.cast import map_copy_select_to
from dev_fn.transform.transform_np import tslrot6d_to_transf_np
from dev_fn.transform.rotation_np import rot6d_to_rotmat_np
from dev_fn.transform.rotation import rot6d_to_rotmat, rotmat_to_quat

from dev_fn.util.vis_o3d_util import VizContext, cvt_from_trimesh
from dev_fn.viz.control import VizControl

from oakink2_tamf.model.interaction_segment_mdm import InterationSegmentMDM
from oakink2_tamf.model.diffusion_util import create_gaussian_diffusion
from oakink2_tamf.dataset.collate import interaction_segment_collate

import manotorch
from manotorch.manolayer import ManoLayer

_logger = logging.getLogger(__name__)

PROG = os.path.splitext(os.path.basename(__file__))[0]
WS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))

PARAM_PREFIX__DATA = "data"
PARAM_PREFIX__DEBUG = "debug"
PARAM_PREFIX__RUNTIME = "runtime"


def reg_entry(config_reg: ConfigRegistry):
    # override default
    # config_reg.meta_info["exp_id"].default = "main"

    # base
    config_reg.register(
        "data_prefix",
        prefix=PARAM_PREFIX__DATA,
        category=str,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        callback=abspath_callback,
        default=f"{WS_DIR}/data",
        required=True,
    )
    config_reg.register(
        "process_range",
        prefix=PARAM_PREFIX__DATA,
        category=list[str],
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        cmdpattern=ConfigEntryCommandlineSeqPattern.COLON_SEP,
        default=["?(file:common/generate_split/main/split/test.txt)"],
    )

    # obj
    config_reg.register(
        "obj_embedding_prefix",
        prefix=PARAM_PREFIX__DATA,
        category=str,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        callback=abspath_callback,
        default="common/retrieve_obj_embedding/main/embedding",
    )

    # cache_dir
    config_reg.register(
        "cache_dict_filepath",
        prefix=PARAM_PREFIX__DEBUG,
        category=str,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        callback=abspath_callback,
        default="common/save_cache_dict/main/cache/test.pkl",
    )

    # mano
    reg_mano_param(config_reg, "mano", WS_DIR)

    # model related, weights
    reg_model_param(config_reg, "model")
    config_reg.register(
        "model_weight_filepath",
        prefix=PARAM_PREFIX__DEBUG,
        category=str,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        callback=abspath_callback,
        required=True
    )

    config_reg.register(
        "sample_id",
        prefix=PARAM_PREFIX__DEBUG,
        category=int,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        default=0,
    )


def reg_extract(config_reg: ConfigRegistry):
    res = {}
    for _p in [PARAM_PREFIX__DATA, PARAM_PREFIX__DEBUG, "mano", "model", PARAM_PREFIX__RUNTIME]:
        try:
            res[_p] = config_reg.select(_p)
        except KeyError:
            pass
    return res


def main():
    config_reg = ConfigRegistry(prog=PROG)
    reg_entry(config_reg)

    parser = argparse.ArgumentParser(prog=PROG)
    config_reg.hook(parser)
    config_reg.parse(parser)

    run_cfg = reg_extract(config_reg)

    log.log_init()
    log.enable_console()
    _logger.info("run_cfg: %s", argdict_to_string(run_cfg))
    log_suppress.suppress()
    suppress_trimesh_logging()

    with open(run_cfg["debug"]["cache_dict_filepath"], "rb") as ifstream:
        cache_dict = pickle.load(ifstream)

    process_range_list = run_cfg["data"]["process_range"]
    all_dataset = InteractionSegmentData(
        process_range_list=process_range_list,
        data_prefix=run_cfg["data"]["data_prefix"],
        obj_embedding_prefix=run_cfg["data"]["obj_embedding_prefix"],
        enable_obj_model=True,
        cache_dict=cache_dict,
    )
    all_object_store = all_dataset.obj_store

    device = torch.device(f"cuda:{3}")
    dtype = torch.float32
    mano_layer_rh = ManoLayer(
        mano_assets_root=run_cfg["mano"]["mano_path"],
        rot_mode="quat",
        side="right",
        center_idx=0,
        use_pca=False,
        flat_hand_mean=True,
    ).to(device)
    rh_faces = mano_layer_rh.th_faces.detach().clone().cpu().numpy()
    mano_layer_lh = ManoLayer(
        mano_assets_root=run_cfg["mano"]["mano_path"],
        rot_mode="quat",
        side="left",
        center_idx=0,
        use_pca=False,
        flat_hand_mean=True,
    ).to(device)
    lh_faces = mano_layer_lh.th_faces.detach().clone().cpu().numpy()

    # model
    model_cfg = run_cfg["model"]
    model = InterationSegmentMDM(
        input_dim=model_cfg["input_dim"],
        obj_input_dim=model_cfg["obj_input_dim"],
        hand_shape_dim=model_cfg["hand_shape_dim"],
        obj_embed_dim=model_cfg["obj_embed_dim"],
        latent_dim=model_cfg["latent_dim"],
        ff_size=model_cfg["ff_size"],
        num_layers=model_cfg["num_layers"],
        num_heads=model_cfg["num_heads"],
        dropout=model_cfg["dropout"],
        activation=model_cfg["activation"],
    ).to(device)
    diffusion = create_gaussian_diffusion(diffusion_steps=1000, noise_schedule="cosine")
    state_dict = torch.load(run_cfg["debug"]["model_weight_filepath"], map_location=device)
    # state_dict = {k[1:]: v for k, v in state_dict.items()}
    missing_key_list, unexpected_key_list = model.load_state_dict(state_dict, strict=False)
    missing_key_list = [k for k in missing_key_list if not k.startswith("clip_model")]
    print(missing_key_list)
    print(unexpected_key_list)

    sample_id = run_cfg["debug"]["sample_id"]
    gt_sample = all_dataset[sample_id]

    # debug
    gt_batch = interaction_segment_collate([gt_sample])
    batch_device = map_copy_select_to(
        gt_batch,
        device=device,
        dtype=dtype,
        select=("mask", "pose_repr", "shape", "obj_num", "obj_traj", "obj_embedding"),
    )
    with torch.no_grad():
        sample_fn = diffusion.p_sample_loop
        model.eval()
        input_shape = tuple(batch_device["pose_repr"].shape)
        input_shape = (input_shape[0], input_shape[2], 1, input_shape[1])
        sample = sample_fn(
            model,
            input_shape,  # adapt from mdm
            clip_denoised=False,
            model_kwargs={"batch": batch_device},
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=False,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )
    sample_ = sample.permute((0, 3, 1, 2))
    sample_np = sample_.detach().clone().cpu().numpy()

    # viz
    viz_control = VizControl()
    viz_control.attach_viz_ctx()

    info = gt_sample["info"]
    text = gt_sample["text"]
    avai_len = gt_sample["len"]
    hand_side = gt_sample["hand_side"]
    # pose_repr = gt_sample["pose_repr"]
    pose_repr = sample_np.squeeze(3).squeeze(0)
    # _tsl = gt_sample["tsl"]
    # _pose = gt_sample["pose"]
    shape = gt_sample["shape"]
    obj_list = gt_sample["obj_list"]
    obj_traj = gt_sample["obj_traj"]
    for obj_id in obj_list:
        viz_control.update_by_mesh(
            obj_id,
            geo=cvt_from_trimesh(all_object_store[obj_id]),
        )

    seqlen = shape.shape[0]
    shape_th = torch.from_numpy(shape).to(device=device, dtype=dtype)  # (seqlen, 10)
    pose_repr_th = torch.from_numpy(pose_repr).to(device=device, dtype=dtype)  # (seqlen, 99)

    with torch.no_grad():
        tsl_th = pose_repr_th[:, 0:3]  # (seqlen, 3)
        pose_rot6d_th = pose_repr_th[:, 3:99]  # (seqlen, 96)
        pose_rot6d_th = pose_rot6d_th.reshape((seqlen, 16, 6))  # (seqlen, 16, 6)
        pose_rotmat_th = rot6d_to_rotmat(pose_rot6d_th)  # (seqlen, 16, 4, 4)
        # pose_rotmat_th = torch.from_numpy(_pose).to(device=device, dtype=dtype)
        pose_quat_th = rotmat_to_quat(pose_rotmat_th)  # (seqlen, 16, 4)

        if hand_side == "rh":
            mano_out = mano_layer_rh(pose_coeffs=pose_quat_th, betas=shape_th)
        elif hand_side == "lh":
            mano_out = mano_layer_lh(pose_coeffs=pose_quat_th, betas=shape_th)
        else:
            raise ValueError(f"unexpected hand_side: {hand_side}")

    hand_verts_th = mano_out.verts + tsl_th.unsqueeze(1)
    hand_verts = hand_verts_th.detach().cpu().numpy()

    print(hand_verts.shape)
    print(text, hand_side, info)
    for frame_offset in range(avai_len):
        viz_control.update_by_mesh(
            "hand",
            verts=hand_verts[frame_offset],
            faces=rh_faces if hand_side == "rh" else lh_faces,
        )
        for _offset, obj_id in enumerate(obj_list):
            _tslrot6d = obj_traj[_offset, frame_offset]
            _transf = tslrot6d_to_transf_np(_tslrot6d)
            viz_control.update_by_mesh(
                obj_id,
                pose=_transf,
            )

        viz_control.condition_reset()
        while viz_control.condition():
            viz_control.step()


if __name__ == "__main__":
    main()
