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
from oakink2_tamf.dataset.collate import SegmentCollate, interaction_segment_collate
from oakink2_tamf.model.segment_encoder import SegmentEncoder
from oakink2_tamf.model.segment_encoder_param import reg_model_param
from oakink2_tamf.dataset.action_adapter import ActionRecognitionAdapter
from oakink2_tamf.model.segment_refine_model import SegmentRefineModel

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

    # model
    reg_model_param(config_reg, "model")

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

    # debug
    config_reg.register(
        "sample_refine_filepath",
        prefix=PARAM_PREFIX__DEBUG,
        category=str,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        callback=abspath_callback,
        default=f"{WS_DIR}/common/sample_refine/main/sample/test/arch_mdm_l__0399",
    )
    config_reg.register(
        "encoder_checkpoint_filepath",
        prefix=PARAM_PREFIX__DEBUG,
        category=str,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        callback=abspath_callback,
        required=True,
    )


def reg_extract(config_reg: ConfigRegistry):
    res = {}
    for _p in [PARAM_PREFIX__DATA, PARAM_PREFIX__DEBUG, "mano", "model"]:
        try:
            res[_p] = config_reg.select(_p)
        except KeyError:
            pass
    return res


import numpy as np
from scipy import linalg


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ("fid calculation produces singular product; "
               "adding %s to diagonal of cov estimates") % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    print("diff", diff.dot(diff))
    print("tr1", np.trace(sigma1))
    print("tr2", np.trace(sigma2))
    print("tr_covmean", tr_covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calculate_activation_statistics(activations):
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)

    return mu, sigma


def calculate_fid(statistics_1, statistics_2):
    return calculate_frechet_distance(statistics_1[0], statistics_1[1], statistics_2[0], statistics_2[1])


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
        obj_pointcloud_prefix=run_cfg["data"]["obj_pointcloud_prefix"],
        append_reverse_segment=False,
        cache_dict=cache_dict,
    )
    all_object_store = all_dataset.obj_store
    all_dataset_action = ActionRecognitionAdapter(all_dataset)
    # pose_repr_sample_dataset = GeneratedPoseReprSampleAdaptor(
    #     all_dataset,
    #     ["common/sample/main/sample/test/arch_mdm_l__0399"],
    # )

    device = torch.device(f"cuda:{4}")
    dtype = torch.float32

    model_cfg = run_cfg["model"]
    segment_encoder = SegmentEncoder(
        all_dataset_action.max_action,
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
    state_dict = torch.load(run_cfg["debug"]["encoder_checkpoint_filepath"], map_location=device)
    missing_key_list, unexpected_key_list = segment_encoder.load_state_dict(state_dict, strict=False)
    missing_key_list = [k for k in missing_key_list if not k.startswith("clip_model")]
    print(missing_key_list)
    print(unexpected_key_list)
    segment_collate_fn = SegmentCollate(extra_default_key=["action_label_id", "action_onehot"],
                                        extra_no_key=["action_label"])

    dataset_activation_list = []
    model_activation_list = []

    duplicate_check = set()
    for sample_id in tqdm(range(len(all_dataset_action))):
        gt_sample = all_dataset_action[sample_id]
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
            _refined_sample = pickle.load(ifstream)

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

        gt_batch = segment_collate_fn([gt_sample])
        gt_batch_device = map_copy_select_to(
            gt_batch,
            device=device,
            dtype=dtype,
            select=("mask", "pose_repr", "shape", "obj_embedding", "obj_traj"),
        )
        gt_batch_device = map_copy_select_to(
            gt_batch_device,
            device=device,
            dtype=torch.long,
            select=("obj_num", "action_label_id"),
        )
        with torch.no_grad():
            _output = segment_encoder(gt_batch_device)
            gt_encoding = _output["encoding"]
            gt_encoding = gt_encoding.detach().clone().cpu().numpy().squeeze(0)
        dataset_activation_list.append(gt_encoding)

        refined_sample = {}
        for k, v in gt_sample.items():
            if k == "pose_repr":
                pass
            refined_sample[k] = v
        refined_sample["pose_repr"] = _refined_sample["refine_pose_repr"].copy()
        refined_sample["pose_repr"][refined_sample['len']:] = 0.0
        refined_sample_batch = segment_collate_fn([refined_sample])
        refined_sample_batch_device = map_copy_select_to(
            refined_sample_batch,
            device=device,
            dtype=dtype,
            select=("mask", "pose_repr", "shape", "obj_embedding", "obj_traj"),
        )
        refined_sample_batch_device = map_copy_select_to(
            refined_sample_batch_device,
            device=device,
            dtype=torch.long,
            select=("obj_num", "action_label_id"),
        )
        with torch.no_grad():
            _output = segment_encoder(refined_sample_batch_device)
            refined_encoding = _output["encoding"]
            refined_encoding = refined_encoding.detach().clone().cpu().numpy().squeeze(0)
        model_activation_list.append(refined_encoding)

    dataset_activation = np.concatenate(dataset_activation_list, axis=0)
    model_activation = np.concatenate(model_activation_list, axis=0)

    print(dataset_activation.shape)
    print(model_activation.shape)

    dataset_statistics = calculate_activation_statistics(dataset_activation)
    model_statistics = calculate_activation_statistics(model_activation)
    print(dataset_statistics[0].shape, dataset_statistics[1].shape)
    fid_value = calculate_fid(dataset_statistics, model_statistics)
    print(fid_value)


if __name__ == "__main__":
    main()
