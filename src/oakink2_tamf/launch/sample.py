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

import torch.multiprocessing as mp

import manotorch
from manotorch.manolayer import ManoLayer


PROG = "debug"

_logger = logging.getLogger(__name__)

PROG = PROG = os.path.splitext(os.path.basename(__file__))[0]
WS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

PARAM_PREFIX__DATA = "data"
PARAM_PREFIX__DEBUG = "debug"
PARAM_PREFIX__RUNTIME = "runtime"


def reg_entry(config_reg: ConfigRegistry):
    # override default
    config_reg.meta_info["exp_id"].default = "main"

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
        "obj_embedding_prefix",
        prefix=PARAM_PREFIX__DATA,
        category=str,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        callback=abspath_callback,
        default=None,
    )

    # cache_dir
    config_reg.register(
        "process_range",
        prefix=PARAM_PREFIX__DATA,
        category=list[str],
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        cmdpattern=ConfigEntryCommandlineSeqPattern.COLON_SEP,
        default=[f"?(file:{WS_DIR}/asset/split/test.txt)"],
    )
    config_reg.register(
        "cache_dict_filepath",
        prefix=PARAM_PREFIX__DATA,
        category=str,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        callback=abspath_callback,
        default=f"{WS_DIR}/common/save_cache_dict/main/cache/test.pkl",
    )

    # mano
    # reg_mano_param(config_reg, "mano", MOCAP_WS_DIR)

    # model related, weights
    reg_model_param(config_reg, "model")
    config_reg.register(
        "model_weight_filepath",
        prefix=PARAM_PREFIX__DEBUG,
        category=str,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        callback=abspath_callback,
    )
    config_reg.register(
        "sample_save_offset",
        prefix=PARAM_PREFIX__DEBUG,
        category=str,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
    )
    config_reg.register(
        "num_worker",
        prefix=PARAM_PREFIX__RUNTIME,
        category=int,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        default=8,
    )
    config_reg.register(
        "device_id",
        prefix=PARAM_PREFIX__RUNTIME,
        category=list[int],
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        cmdpattern=ConfigEntryCommandlineSeqPattern.COMMA_SEP,
        default=[0, 1, 2, 3],
    )


def reg_extract(config_reg: ConfigRegistry):
    res = {}
    for _p in [
        PARAM_PREFIX__DATA,
        PARAM_PREFIX__DEBUG,
        "mano",
        "model",
        PARAM_PREFIX__RUNTIME,
    ]:
        try:
            res[_p] = config_reg.select(_p)
        except KeyError:
            pass
    return res


def sample_worker(
    log_queue,
    worker_id,
    num_worker,
    device_id,
    ckpt_cfg,
    run_cfg,
):
    log.configure_mp_worker(log_queue, worker_id)
    _logger.info("worker_id: %02d", worker_id)
    _logger.info("device_id: %d", device_id)

    commit = ckpt_cfg["commit"]
    ckpt_path = ckpt_cfg["ckpt_path"]

    process_range_list = run_cfg["data"]["process_range"]
    with open(run_cfg["data"]["cache_dict_filepath"], "rb") as ifstream:
        cache_dict = pickle.load(ifstream)
    all_dataset = InteractionSegmentData(
        process_range_list=process_range_list,
        data_prefix=run_cfg["data"]["data_prefix"],
        obj_embedding_prefix=run_cfg["data"]["obj_embedding_prefix"],
        enable_obj_model=True,
        cache_dict=cache_dict,
    )

    device = torch.device(f"cuda:{device_id}")
    dtype = torch.float32

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
    if worker_id == 0:
        _logger.info("missing_keys: %s", missing_key_list)
        _logger.info("unexpected_keys: %s", unexpected_key_list)

    worker_sample_id_start = int(len(all_dataset) * worker_id / num_worker)
    worker_sample_id_stop = int(len(all_dataset) * (worker_id + 1) / num_worker)
    _logger.info("%06d %06d", worker_sample_id_start, worker_sample_id_stop)

    for sample_id in range(worker_sample_id_start, worker_sample_id_stop):
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
        pose_repr_sample_np = sample_np.squeeze(3).squeeze(0)

        if commit:
            pose_repr_sampe_path = os.path.join(ckpt_path, "sample", run_cfg["debug"]["sample_save_offset"], f"{sample_id:06d}.npy")
            os.makedirs(os.path.dirname(pose_repr_sampe_path), exist_ok=True)
            np.save(pose_repr_sampe_path, pose_repr_sample_np)
        
        _logger.info("sample %06d", sample_id)


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

    num_worker = run_cfg["runtime"]["num_worker"]
    device_id_list = run_cfg["runtime"]["device_id"]

    # set multiprocessing context
    mp.set_start_method("spawn")
    log_queue = mp.Queue()
    log.configure_mp_main(log_queue)

    process_list = []
    for worker_id in range(num_worker):
        worker_device_id = device_id_list[worker_id % len(device_id_list)]
        process_list.append(
            mp.Process(
                target=sample_worker,
                kwargs=dict(
                    log_queue=log_queue,
                    worker_id=worker_id,
                    num_worker=num_worker,
                    device_id=worker_device_id,
                    ckpt_cfg=ckpt_cfg,
                    run_cfg=run_cfg,
                ),
            )
        )
    for worker_id in range(num_worker):
        process_list[worker_id].start()

    for worker_id in range(num_worker):
        process_list[worker_id].join()

    log.deconfigure_mp_main(log_queue)
    _logger.info("conclude parallel worker")


if __name__ == "__main__":
    main()
