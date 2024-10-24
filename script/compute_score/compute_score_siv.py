import os
import numpy as np
import torch
import json
from copy import deepcopy
import pickle
import trimesh

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
from dev_fn.external.libmesh import check_mesh_contains
from dev_fn.util.sdf_util import process_sdf

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

obj_sdf_map = {}


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


def solid_intersection_volume(hand_verts, hand_faces, obj_transf_map):
    siv = 0.0

    # viz_control = VizControl()
    # viz_control.attach_viz_ctx()

    hand_trimesh = trimesh.Trimesh(vertices=np.asarray(hand_verts), faces=np.asarray(hand_faces))
    # viz_control.update_by_mesh("hand", geo=cvt_from_trimesh(hand_trimesh))
    for obj_id in obj_transf_map:
        if obj_id not in obj_sdf_map:
            continue
        obj_transf = obj_transf_map[obj_id]
        _sdf = obj_sdf_map[obj_id]
        _obj_verts_in = _sdf.point[_sdf.sdf > 0]
        _obj_verts_in = _obj_verts_in + _sdf.mesh_center

        el_vol = np.prod(_sdf.tick_unit)
        query_verts = transf_point_array_np(obj_transf, _obj_verts_in)
        # viz_control.update_by_pc("obj", points=query_verts)
        inside = check_mesh_contains(hand_trimesh, query_verts)
        volume = inside.sum() * el_vol * (10**6)
        siv += volume

    # viz_control.condition_reset()
    # while viz_control.condition():
    #     viz_control.step()
    # assert False
    return siv


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

    gt_siv_list = []
    refined_siv_list = []

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
        refined_hand_verts = refined_sample["verts"]

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

        gt_hand_verts_th = mano_out.verts + tsl_th.unsqueeze(1)
        gt_hand_verts = gt_hand_verts_th.detach().cpu().numpy()

        # slice with seqlen
        gt_hand_verts = gt_hand_verts[:avai_len, ...]
        refined_hand_verts = refined_hand_verts[:avai_len, ...]
        obj_traj = obj_traj[:, :avai_len, ...]

        obj_transf = tslrot6d_to_transf_np(obj_traj)
        for obj_id in obj_list:
            if obj_id not in obj_sdf_map:
                obj_sdf_map[obj_id] = process_sdf(all_object_store[obj_id])

        _gt_siv_list = []
        _refined_siv_list = []
        for frame_offset in range(0, avai_len, 20):
            _gt_hv = gt_hand_verts[frame_offset]
            _hf = rh_faces_closed if hand_side == "rh" else lh_faces_closed
            _refined_hv = refined_hand_verts[frame_offset]
            _o_transf_map = {}
            for _offset, _oid in enumerate(obj_list):
                _o_transf_map[_oid] = obj_transf[_offset, frame_offset]
            _gt_siv = solid_intersection_volume(_gt_hv, _hf, _o_transf_map)
            _refined_siv = solid_intersection_volume(_refined_hv, _hf, _o_transf_map)
            _gt_siv_list.append(_gt_siv)
            _refined_siv_list.append(_refined_siv)

        gt_siv_list.extend(_gt_siv_list)
        refined_siv_list.extend(_refined_siv_list)

    print(np.mean(gt_siv_list))
    print(np.mean(refined_siv_list))

    os.makedirs("./tmp/compute_score/solid_intersection_volume/", exist_ok=True)
    np.save("./tmp/compute_score/solid_intersection_volume/gt.npy", gt_siv_list)
    np.save("./tmp/compute_score/solid_intersection_volume/refined.npy", refined_siv_list)


if __name__ == "__main__":
    main()
