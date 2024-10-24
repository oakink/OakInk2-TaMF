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
from oakink2_tamf.launch.param import reg_mano_param
from dev_fn.util.console_io import suppress_trimesh_logging

from oakink2_tamf.dataset.interaction_segment import InteractionSegmentData
from dev_fn.transform.cast import map_copy_select_to
from dev_fn.transform.transform_np import tslrot6d_to_transf_np, transf_point_array_np
from dev_fn.transform.rotation_np import rot6d_to_rotmat_np
from dev_fn.transform.rotation import rot6d_to_rotmat, rotmat_to_quat
from oakink2_tamf.dataset.pose_repr_sample import GeneratedPoseReprSampleAdaptor
from oakink2_tamf.dataset.collate import interaction_segment_collate
from oakink2_tamf.model.segment_refine_model import SegmentRefineModel

import scipy.signal

import manotorch
from manotorch.manolayer import ManoLayer

from tqdm import tqdm

_logger = logging.getLogger(__name__)

PROG = os.path.splitext(os.path.basename(__file__))[0]
WS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))

PARAM_PREFIX__DATA = "data"
PARAM_PREFIX__MANO = "mano"
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
        default=[f"?(file:{WS_DIR}/mocap_meta/process_range/test.txt)"],
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
    config_reg.register(
        "obj_pointcloud_prefix",
        prefix=PARAM_PREFIX__DATA,
        category=str,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        callback=abspath_callback,
        default="common/retrieve_obj_pointcloud/main/pointcloud",
    )

    # cache_dir
    config_reg.register(
        "cache_dict_filepath",
        prefix=PARAM_PREFIX__DATA,
        category=str,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        callback=abspath_callback,
        default="common/save_cache_dict/main/cache/test.pkl",
    )

    # mano
    reg_mano_param(config_reg, "mano", WS_DIR)

    # debug
    config_reg.register(
        "sample_refine_filepath",
        prefix=PARAM_PREFIX__DEBUG,
        category=str,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        callback=abspath_callback,
        default=f"{WS_DIR}/common/sample_refine/main/sample/test/arch_mdm_l__0399",
    )


def reg_extract(config_reg: ConfigRegistry):
    res = {}
    for _p in [PARAM_PREFIX__DATA, PARAM_PREFIX__DEBUG, "mano", "model"]:
        try:
            res[_p] = config_reg.select(_p)
        except KeyError:
            pass
    return res


def transf_merge_obj_pointcloud(obj_pointcloud, obj_traj):
    num_obj = obj_pointcloud.shape[0]
    seq_len = obj_traj.shape[1]
    num_verts = obj_pointcloud.shape[1]
    res = []
    _obj_traj_transf = tslrot6d_to_transf_np(obj_traj)  # (nobj, seqlen, 4, 4)
    _obj_pc = np.expand_dims(obj_pointcloud, 1)  # (nobj, 1, nv, 3)
    _obj_pc = np.broadcast_to(_obj_pc, (num_obj, seq_len, num_verts, 3))
    _pc_transf = transf_point_array_np(_obj_traj_transf, _obj_pc)
    # _pc_transf shape (nobj, seqlen, nv ,3)
    # swapaxes to (seqlen, nobj, nv, 3)
    _pc_transf = np.swapaxes(_pc_transf, 0, 1)
    # reshape to (seqlen, _, 3)
    _pc_transf = _pc_transf.reshape((_pc_transf.shape[0], -1, 3))
    return _pc_transf


def contact_min_cdist(hv, pc, device, dtype):
    # create torch tensor with dtype and device
    hv = torch.from_numpy(hv).to(device=device, dtype=dtype)
    pc = torch.from_numpy(pc).to(device=device, dtype=dtype)
    # torch cdist
    dist = torch.cdist(hv, pc, p=2)  # (seqlen, nv_h, nv_o)
    dist_flatten = dist.reshape((dist.shape[0], -1))
    min_cdist, _ = torch.min(dist_flatten, dim=1)
    min_cdist = min_cdist.detach().cpu().numpy().tolist()
    return min_cdist


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

    with open(run_cfg["data"]["cache_dict_filepath"], "rb") as ifstream:
        cache_dict = pickle.load(ifstream)

    process_range_list = run_cfg["data"]["process_range"]
    all_dataset = InteractionSegmentData(
        process_range_list=process_range_list,
        data_prefix=run_cfg["data"]["data_prefix"],
        obj_embedding_prefix=run_cfg["data"]["obj_embedding_prefix"],
        enable_obj_model=True,
        obj_pointcloud_prefix=run_cfg["data"]["obj_pointcloud_prefix"],
        append_reverse_segment=False,
        cache_dict=cache_dict,
    )
    all_object_store = all_dataset.obj_store
    # pose_repr_sample_dataset = GeneratedPoseReprSampleAdaptor(
    #     all_dataset,
    #     ["common/sample/main/sample/test/arch_mdm_l__0399"],
    # )

    device = torch.device(f"cuda:{4}")
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
    rh_faces_closed = mano_layer_rh.get_mano_closed_faces().detach().clone().cpu().numpy()
    mano_layer_lh = ManoLayer(
        mano_assets_root=run_cfg["mano"]["mano_path"],
        rot_mode="quat",
        side="left",
        center_idx=0,
        use_pca=False,
        flat_hand_mean=True,
    ).to(device)
    lh_faces = mano_layer_lh.th_faces.detach().clone().cpu().numpy()
    lh_faces_closed = mano_layer_lh.get_mano_closed_faces().detach().clone().cpu().numpy()

    dataset_res_list = []
    model_res_list = []

    duplicate_check = set()
    for sample_id in tqdm(range(len(all_dataset))):
        gt_sample = all_dataset[sample_id]
        info = gt_sample["info"]
        if info in duplicate_check:
            continue
        duplicate_check.add(info)

        load_filepath = os.path.join(
            run_cfg["debug"]["sample_refine_filepath"],
            str(info[0]).replace('/', '++'),
            str(info[1]),
            str(info[2]),
            "save_dict.pkl",
        )
        if not os.path.exists(load_filepath):
            continue
        with open(load_filepath, "rb") as ifstream:
            refined_sample = pickle.load(ifstream)
        refined_hand_joints = refined_sample["joints"]

        text = gt_sample["text"]
        avai_len = gt_sample["len"]
        hand_side = gt_sample["hand_side"]
        pose_repr = gt_sample["pose_repr"]
        # sample_pose_repr = pose_repr
        # sample_pose_repr = gt_sample["sample_pose_repr"]
        # _tsl = gt_sample["tsl"]
        # _pose = gt_sample["pose"]
        shape = gt_sample["shape"]
        obj_list = gt_sample["obj_list"]
        obj_traj = gt_sample["obj_traj"]
        obj_pointcloud = gt_sample["obj_pointcloud"]

        seqlen = shape.shape[0]
        shape_th = torch.from_numpy(shape).to(device=device, dtype=dtype)  # (seqlen, 10)
        gt_pose_repr_th = torch.from_numpy(pose_repr).to(device=device, dtype=dtype)  # (seqlen, 99)

        with torch.no_grad():
            tsl_th = gt_pose_repr_th[:, 0:3]  # (seqlen, 3)
            pose_rot6d_th = gt_pose_repr_th[:, 3:99]  # (seqlen, 96)
            pose_rot6d_th = pose_rot6d_th.reshape((seqlen, 16, 6))  # (seqlen, 16, 6)
            pose_rotmat_th = rot6d_to_rotmat(pose_rot6d_th)  # (seqlen, 16, 4, 4)
            pose_quat_th = rotmat_to_quat(pose_rotmat_th)  # (seqlen, 16, 4)

            if hand_side == "rh":
                mano_out = mano_layer_rh(pose_coeffs=pose_quat_th, betas=shape_th)
            elif hand_side == "lh":
                mano_out = mano_layer_lh(pose_coeffs=pose_quat_th, betas=shape_th)
            else:
                raise ValueError(f"unexpected hand_side: {hand_side}")

        gt_hand_joints_th = mano_out.joints + tsl_th.unsqueeze(1)
        gt_hand_joints = gt_hand_joints_th.detach().cpu().numpy()

        gt_hand_joints[avai_len:, :, :] = gt_hand_joints[avai_len - 1, :, :]
        refined_hand_joints[avai_len:, :, :] = refined_hand_joints[avai_len - 1, :, :]

        dataset_res_list.append(gt_hand_joints[:])
        model_res_list.append(refined_hand_joints[:])

    # adapt from https://github.com/magnux/MotionGAN/blob/master/test.py
    # see SAGA, which use the power spectrum KL divergence of Joints, which use accelarations of joints
    dataset_psd_list = []
    for offset in range(len(dataset_res_list)):
        _acc = np.diff(dataset_res_list[offset], n=2, axis=0)
        # ps (welch)
        # _, _psd = scipy.signal.welch(_acc, axis=0, nperseg=80) (n/2+1, 21, 3)
        # ps (fft)
        _fft = np.fft.fft(_acc, axis=0)
        _psd = np.abs(_fft) ** 2
        dataset_psd_list.append(_psd)
    dataset_psd = np.stack(dataset_psd_list, axis=0)
    print(dataset_psd.shape)

    model_psd_list = []
    for offset in range(len(model_res_list)):
        _acc = np.diff(model_res_list[offset], n=2, axis=0)
        # ps (welch)
        # _, _psd = scipy.signal.welch(_acc, axis=0, nperseg=80)
        # ps (fft)
        _fft = np.fft.fft(_acc, axis=0)
        _psd = np.abs(_fft) ** 2
        model_psd_list.append(_psd)
    model_psd = np.stack(model_psd_list, axis=0)
    print(model_psd.shape)

    # compute power specturm
    # axis 1 is the time axis
    # compute mean value along sample axis (axis 0) to get the distribution of PS
    dataset_mean_ps = np.sum(dataset_psd, axis=0) + 1e-8
    model_mean_ps = np.sum(model_psd, axis=0) + 1e-8
    print(dataset_mean_ps.shape, model_mean_ps.shape)
    # # normalize along frequency dimension (the new axis 0)
    dataset_mean_ps = dataset_mean_ps / np.sum(dataset_mean_ps, axis=0, keepdims=True)
    model_mean_ps = model_mean_ps / np.sum(model_mean_ps, axis=0, keepdims=True)
    print(dataset_mean_ps.shape, model_mean_ps.shape)

    # compute the pskl (first gt model, then model gt)
    num_feat = dataset_mean_ps.shape[1]
    pskl_1 = 1 / num_feat * np.sum(dataset_mean_ps * np.log(dataset_mean_ps / model_mean_ps))
    pskl_2 = 1 / num_feat * np.sum(model_mean_ps * np.log(model_mean_ps / dataset_mean_ps))
    print(pskl_1, pskl_2)


if __name__ == "__main__":
    main()
