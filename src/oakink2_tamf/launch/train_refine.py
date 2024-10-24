import os
import numpy as np
import torch
import cv2
import json

import logging
import argparse
import itertools
import shutil
import tqdm
import time
import pickle

import torch.multiprocessing as mp
import torch.distributed as dist
from ..util import ddp_util

from config_reg import ConfigRegistry, ConfigEntrySource
from config_reg import (
    ConfigEntryCommandlineBoolPattern,
    ConfigEntryCommandlineSeqPattern,
)
from config_reg.callback import abspath_callback
from dev_fn.upkeep import log
from ..util import log_suppress
from dev_fn.upkeep import ckpt
from dev_fn.upkeep.opt import argdict_to_string
from dev_fn.upkeep.config import cb__decode_file, cls__cb__link_bool_opt

from dev_fn.util.console_io import suppress_trimesh_logging
from dev_fn.util import pbar_util
from dev_fn.util import random_util

from dev_fn.transform.cast import map_copy_select_to
from ..dataset.interaction_segment import InteractionSegmentData
from ..dataset.collate import interaction_segment_collate
from ..dataset.pose_repr_sample import GeneratedPoseReprSampleAdaptor, GuassianPerturbSampleAdaptor
from ..model.segment_refine_model import SegmentRefineModel
from ..model.segment_refine_model_loss import SegmentRefineModelLoss
from ..util.net_util import clip_gradient
from ..util.state_util import save_state
from ..util.summary_writer import DDPSummaryWriter

from .param import reg_mano_param, reg_model_param
from .param.loss_refine import reg_loss_param

_logger = logging.getLogger(__name__)

PROG = "train"
WS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

PARAM_PREFIX__DATA = "data"
PARAM_PREFIX__TRAIN = "train"
PARAM_PREFIX__VAL = "val"
PARAM_PREFIX__TEST = "test"
PARAM_PREFIX__RUNTIME = "runtime"


def reg_entry(config_reg: ConfigRegistry):
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
        "enable_obj_model",
        prefix=PARAM_PREFIX__DATA,
        category=bool,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        cmdpattern=ConfigEntryCommandlineBoolPattern.SET_TRUE,
        default=False,
    )

    # obj
    config_reg.register(
        "obj_embedding_prefix",
        prefix=PARAM_PREFIX__DATA,
        category=str,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        callback=abspath_callback,
        default=None,
    )
    config_reg.register(
        "obj_pointcloud_prefix",
        prefix=PARAM_PREFIX__DATA,
        category=str,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        callback=abspath_callback,
        default=None,
    )
    # mano
    reg_mano_param(config_reg, "mano", WS_DIR)

    # train
    config_reg.register(
        "process_range",
        prefix=PARAM_PREFIX__TRAIN,
        category=list[str],
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        cmdpattern=ConfigEntryCommandlineSeqPattern.COLON_SEP,
        callback=cb__decode_file,
        default=[f"?(file:{WS_DIR}/asset/split/train.txt)"],
    )
    config_reg.register(
        "cache_dict_filepath",
        prefix=PARAM_PREFIX__TRAIN,
        category=str,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        callback=abspath_callback,
        default=None,
    )
    config_reg.register(
        "data.pose_repr_sample_dir_list",
        prefix=PARAM_PREFIX__TRAIN,
        category=list[str],
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        cmdpattern=ConfigEntryCommandlineSeqPattern.COLON_SEP,
    )
    config_reg.register(
        "data.gaussian_perturb_range",
        prefix=PARAM_PREFIX__TRAIN,
        category=list[float],
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        cmdpattern=ConfigEntryCommandlineSeqPattern.COLON_SEP,
    )
    config_reg.register(
        "batch_size",
        prefix=PARAM_PREFIX__TRAIN,
        category=int,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        default=16,
    )
    config_reg.register(
        "num_epoch",
        prefix=PARAM_PREFIX__TRAIN,
        category=int,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        default=400,
    )
    config_reg.register(
        "record_freq",
        prefix=PARAM_PREFIX__TRAIN,
        category=int,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        default=20,
    )
    config_reg.register(
        "scheduler_milestone",
        prefix=PARAM_PREFIX__TRAIN,
        category=list[int],
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        cmdpattern=ConfigEntryCommandlineSeqPattern.COMMA_SEP,
        default=[150, 250],
    )
    config_reg.register(
        "scheduler_gamma",
        prefix=PARAM_PREFIX__TRAIN,
        category=float,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        default=0.5,
    )
    reg_loss_param(config_reg, f"{PARAM_PREFIX__TRAIN}.loss", WS_DIR)
    config_reg.register(
        "reload_ckpt_model_filepath",
        prefix=PARAM_PREFIX__TRAIN,
        category=str,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        callback=abspath_callback,
        default=None,
    )
    config_reg.register(
        "reload_ckpt_optimizer_filepath",
        prefix=PARAM_PREFIX__TRAIN,
        category=str,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        callback=abspath_callback,
        default=None,
    )

    # val
    config_reg.register(
        "process_range",
        prefix=PARAM_PREFIX__VAL,
        category=list[str],
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        cmdpattern=ConfigEntryCommandlineSeqPattern.COLON_SEP,
        callback=cb__decode_file,
        default=[f"?(file:{WS_DIR}/asset/split/val.txt)"],
    )
    config_reg.register(
        "cache_dict_filepath",
        prefix=PARAM_PREFIX__VAL,
        category=str,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        callback=abspath_callback,
        default=None,
    )
    config_reg.register(
        "data.pose_repr_sample_dir_list",
        prefix=PARAM_PREFIX__VAL,
        category=list[str],
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        cmdpattern=ConfigEntryCommandlineSeqPattern.COLON_SEP,
    )
    config_reg.register(
        "batch_size",
        prefix=PARAM_PREFIX__VAL,
        category=int,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        default=8,
    )
    config_reg.register(
        "val_freq",
        prefix=PARAM_PREFIX__VAL,
        category=int,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        default=20,
    )

    # test
    config_reg.register(
        "process_range",
        prefix=PARAM_PREFIX__TEST,
        category=list[str],
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        cmdpattern=ConfigEntryCommandlineSeqPattern.COLON_SEP,
        callback=cb__decode_file,
        default=[f"?(file:{WS_DIR}/asset/split/test.txt)"],
    )
    config_reg.register(
        "cache_dict_filepath",
        prefix=PARAM_PREFIX__TEST,
        category=str,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        callback=abspath_callback,
        default=None,
    )
    config_reg.register(
        "data.pose_repr_sample_dir_list",
        prefix=PARAM_PREFIX__TEST,
        category=list[str],
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        cmdpattern=ConfigEntryCommandlineSeqPattern.COLON_SEP,
    )
    config_reg.register(
        "batch_size",
        prefix=PARAM_PREFIX__TEST,
        category=int,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        default=8,
    )
    config_reg.register(
        "test_freq",
        prefix=PARAM_PREFIX__TEST,
        category=int,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        default=100,
    )

    # model
    reg_model_param(config_reg, "model")

    ## runtime config
    config_reg.register(
        "num_worker",
        prefix=PARAM_PREFIX__RUNTIME,
        category=int,
        source=ConfigEntrySource.COMMANDLINE_ONLY,
        default=2,
    )
    config_reg.register(
        "device_id",
        prefix=PARAM_PREFIX__RUNTIME,
        category=list[int],
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        cmdpattern=ConfigEntryCommandlineSeqPattern.COMMA_SEP,
        default=[0],
    )
    config_reg.register(
        "seed",
        prefix=PARAM_PREFIX__RUNTIME,
        category=int,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        default=0,
    )


def reg_extract(config_reg: ConfigRegistry):
    res = {}
    for _p in [
        PARAM_PREFIX__DATA,
        PARAM_PREFIX__TRAIN,
        PARAM_PREFIX__VAL,
        PARAM_PREFIX__TEST,
        PARAM_PREFIX__RUNTIME,
        "model",
        "mano",
    ]:
        try:
            res[_p] = config_reg.select(_p)
        except KeyError:
            pass
    return res


def summary_extra_loss(summary_writer, loss_store, lr=None, step_id=None, prefix=None):
    time_curr = time.time()
    for loss_name, loss_node in loss_store.items():
        if loss_name == "loss":
            tag = f"{prefix}/loss" if prefix is not None else "loss"
            summary_writer.add_scalar(tag=tag, value=float(loss_node), global_step=step_id, walltime=time_curr)
        else:
            tag = f"{prefix}/loss_comp/{loss_name}" if prefix is not None else f"loss_comp/{loss_name}"
            summary_writer.add_scalar(tag=tag, value=float(loss_node), global_step=step_id, walltime=time_curr)
    if lr is not None:
        summary_writer.add_scalar(tag="lr", value=float(lr), global_step=step_id, walltime=time_curr)


def run(rank, world_size, ckpt_cfg, run_cfg):
    log.log_init()
    log.enable_console()
    ddp_util.log_status_update(rank)  # disable root_logger on non_zero rank

    ckpt.ckpt_setup(ckpt_cfg, rank=rank)
    ckpt.ckpt_opt(ckpt_cfg, rank=rank, world_size=world_size, ckpt=ckpt_cfg, run=run_cfg)

    _logger.info("world_size: %d", world_size)
    _logger.info("ckpt_cfg: %s", argdict_to_string(ckpt_cfg))
    _logger.info("run_cfg: %s", argdict_to_string(run_cfg))
    log_suppress.suppress()
    suppress_trimesh_logging()

    # device
    device_id_list = run_cfg["runtime"]["device_id"]
    device_id = device_id_list[rank] if rank is not None else device_id_list[0]
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device)
    dtype = torch.float32
    ddp_util.setup_ddp(rank, world_size)

    # ckpt
    commit = ckpt_cfg["commit"]
    ckpt_path = ckpt_cfg["ckpt_path"]

    # dataset
    train_process_range = run_cfg["train"]["process_range"]
    if run_cfg["train"]["cache_dict_filepath"] is not None:
        with open(run_cfg["train"]["cache_dict_filepath"], "rb") as ifstream:
            train_dataset_cache_dict = pickle.load(ifstream)
    else:
        train_dataset_cache_dict = None
    train_dataset_ = InteractionSegmentData(
        process_range_list=train_process_range,
        data_prefix=run_cfg["data"]["data_prefix"],
        obj_embedding_prefix=run_cfg["data"]["obj_embedding_prefix"],
        enable_obj_model=run_cfg["data"]["enable_obj_model"],
        obj_pointcloud_prefix=run_cfg["data"]["obj_pointcloud_prefix"],
        cache_dict=train_dataset_cache_dict,
        rank=rank,
    )
    train_pose_repr_sample_dataset = GeneratedPoseReprSampleAdaptor(
        train_dataset_, run_cfg["train"]["data"]["pose_repr_sample_dir_list"]
    )
    train_gaussian_perturb_dataset = GuassianPerturbSampleAdaptor(
        train_dataset_, run_cfg["train"]["data"]["gaussian_perturb_range"]
    )
    train_dataset = torch.utils.data.ConcatDataset([train_pose_repr_sample_dataset, train_gaussian_perturb_dataset])
    _logger.info("train_dataset: length %d", len(train_dataset))

    if not rank:
        val_process_range = run_cfg["val"]["process_range"]
        if run_cfg["val"]["cache_dict_filepath"] is not None:
            with open(run_cfg["val"]["cache_dict_filepath"], "rb") as ifstream:
                val_dataset_cache_dict = pickle.load(ifstream)
        else:
            val_dataset_cache_dict = None
        val_dataset_ = InteractionSegmentData(
            process_range_list=val_process_range,
            data_prefix=run_cfg["data"]["data_prefix"],
            obj_embedding_prefix=run_cfg["data"]["obj_embedding_prefix"],
            enable_obj_model=run_cfg["data"]["enable_obj_model"],
            obj_pointcloud_prefix=run_cfg["data"]["obj_pointcloud_prefix"],
            cache_dict=val_dataset_cache_dict,
            rank=rank,
        )
        val_dataset = GeneratedPoseReprSampleAdaptor(val_dataset_, run_cfg["val"]["data"]["pose_repr_sample_dir_list"])
        _logger.info("val_dataset: length %d", len(val_dataset))

        test_process_range = run_cfg["test"]["process_range"]
        if run_cfg["test"]["cache_dict_filepath"] is not None:
            with open(run_cfg["test"]["cache_dict_filepath"], "rb") as ifstream:
                test_dataset_cache_dict = pickle.load(ifstream)
        else:
            test_dataset_cache_dict = None
        test_dataset_ = InteractionSegmentData(
            process_range_list=test_process_range,
            data_prefix=run_cfg["data"]["data_prefix"],
            obj_embedding_prefix=run_cfg["data"]["obj_embedding_prefix"],
            enable_obj_model=run_cfg["data"]["enable_obj_model"],
            obj_pointcloud_prefix=run_cfg["data"]["obj_pointcloud_prefix"],
            cache_dict=test_dataset_cache_dict,
            rank=rank,
        )
        test_dataset = GeneratedPoseReprSampleAdaptor(
            test_dataset_, run_cfg["test"]["data"]["pose_repr_sample_dir_list"]
        )
        _logger.info("test_dataset: length %d", len(test_dataset))
    else:
        val_process_range, val_dataset = None, None
        test_process_range, test_dataset = None, None

    # handle batch_size
    world_batch_size = run_cfg["train"]["batch_size"]
    if world_batch_size % world_size != 0:
        _logger.warning("batch_size %d is not divisible by world_size %d", world_batch_size, world_size)
    batch_size = world_batch_size // world_size
    _logger.info("batch_size: %d | equiv world_batch_size %d", batch_size, batch_size * world_size)

    # handle data_loader
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        collate_fn=interaction_segment_collate,
        batch_size=batch_size,
        num_workers=run_cfg["runtime"]["num_worker"],
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        drop_last=True,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=None,
    )
    if not rank:
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            collate_fn=interaction_segment_collate,
            batch_size=run_cfg["val"]["batch_size"],
            num_workers=run_cfg["runtime"]["num_worker"],
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            persistent_workers=True,
            worker_init_fn=None,
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            collate_fn=interaction_segment_collate,
            batch_size=run_cfg["test"]["batch_size"],
            num_workers=run_cfg["runtime"]["num_worker"],
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            persistent_workers=True,
            worker_init_fn=None,
        )
    else:
        val_dataloader = None
        test_dataloader = None

    # model
    model_cfg = run_cfg["model"]
    model_ = SegmentRefineModel(
        run_cfg["mano"]["mano_path"],
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
        use_pc=run_cfg["data"]["obj_pointcloud_prefix"] is not None,
    ).to(device)
        # resume model
    if (reload_ckpt_model_filepath := run_cfg["train"]["reload_ckpt_model_filepath"]) is not None:
        _logger.info("model reload ckpt: %s", reload_ckpt_model_filepath)
        reload_ckpt_model_weight = torch.load(reload_ckpt_model_filepath, map_location=device)
        missing_key_list, unexpected_key_list = model_.load_state_dict(reload_ckpt_model_weight, strict=False)
        missing_key_list = [k for k in missing_key_list if not k.startswith("clip_model")]
        _logger.info("model reload ckpt missing_key_list: %s", missing_key_list)
        _logger.info("model reload ckpt unexpected_key_list: %s", unexpected_key_list)
    model = torch.nn.parallel.DistributedDataParallel(
        model_,
        device_ids=[device],
        output_device=device,
        # find_unused_parameters=True,
    )
    model_loss = SegmentRefineModelLoss(run_cfg["train"]["loss"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
    # resume optimizer
    if (reload_ckpt_optimizer_filepath := run_cfg["train"]["reload_ckpt_optimizer_filepath"]) is not None:
        _logger.info("optimzier reload ckpt: %s", reload_ckpt_optimizer_filepath)
        reload_ckpt_optimizer_param = torch.load(reload_ckpt_optimizer_filepath, map_location=device)
        optimizer.load_state_dict(reload_ckpt_optimizer_param)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=run_cfg["train"]["scheduler_milestone"],
        gamma=run_cfg["train"]["scheduler_gamma"],
    )

    # summary writer
    summary_writer_dir = ckpt.handle_save_path(f"?(ckpt_path)/summary", ckpt_path=ckpt_path) if commit else None
    summary_writer = DDPSummaryWriter(log_dir=summary_writer_dir, rank=rank)

    # seed
    seed = run_cfg["runtime"]["seed"]
    if rank is not None:
        seed = seed + rank
    random_util.setup_seed(seed)

    # mainloop
    dist.barrier()
    num_epoch = run_cfg["train"]["num_epoch"]
    for epoch_id in range(num_epoch):
        # epoch begin
        train_sampler.set_epoch(epoch_id)

        # epoch iterate
        pbar = (
            tqdm.tqdm(
                total=len(train_dataloader), position=0, bar_format=pbar_util.fmt, desc=f"train epoch {epoch_id:>04d}:"
            )
            if not rank
            else pbar_util.dummy_pbar()
        )
        for batch_id, batch in enumerate(train_dataloader):
            step_id = (epoch_id * len(train_dataloader) + batch_id) * world_size

            optimizer.zero_grad()
            model.train()

            batch_device = map_copy_select_to(
                batch,
                device=device,
                dtype=dtype,
                select=("mask", "pose_repr", "shape", "obj_num", "obj_traj", "obj_embedding", "sample_pose_repr"),
            )
            output = model(batch_device)
            loss, loss_store = model_loss(output, batch_device)
            # loss

            loss.backward()
            clip_gradient(optimizer, 0.1, 2.0)
            optimizer.step()
            optimizer.zero_grad()

            # summary writer
            summary_extra_loss(
                summary_writer,
                loss_store,
                lr=float(next(iter(optimizer.param_groups))["lr"]),
                step_id=step_id,
                prefix="train",
            )

            pbar.update()
        pbar.close()

        # epoch end
        scheduler.step()
        dist.barrier()
        _logger.info("train epoch %04d conclude | loss: %f", epoch_id, loss.item())
        _logger.info("train epoch %04d lr %s", epoch_id, [group["lr"] for group in optimizer.param_groups])
        _logger.info("train epoch %04d detail loss:", epoch_id)
        for loss_name, loss_node in loss_store.items():
            _logger.info("                     %s: %f", loss_name.ljust(20), float(loss_node))

        if not rank:
            # record_state
            record_freq = run_cfg["train"]["record_freq"]
            if (
                commit
                and (record_freq is not None and record_freq != -1)
                and (epoch_id == 0 or epoch_id % record_freq == record_freq - 1 or epoch_id == num_epoch - 1)
            ):
                model_weight_path = ckpt.handle_save_path(
                    f"?(ckpt_path)/save/model_{epoch_id:0>4}.pt", ckpt_path=ckpt_path
                )
                save_state(model.state_dict(), model_weight_path, remove_prefix="module", filter_out="clip_model")
                optimizer_weight_path = ckpt.handle_save_path(
                    f"?(ckpt_path)/save/optimizer_{epoch_id:0>4}.pt", ckpt_path=ckpt_path
                )
                save_state(optimizer.state_dict(), optimizer_weight_path)

        dist.barrier()

        if not rank:
            # val
            val_freq = run_cfg["val"]["val_freq"]
            if (val_freq is not None and val_freq != -1) and (
                epoch_id == 0 or epoch_id % val_freq == val_freq - 1 or epoch_id == num_epoch - 1
            ):
                with torch.no_grad():
                    val_pbar = tqdm.tqdm(
                        total=len(val_dataloader),
                        position=0,
                        bar_format=pbar_util.fmt,
                        desc=f"val   epoch {epoch_id:>04d}:",
                    )
                    model.eval()
                    for val_batch_id, val_batch in enumerate(val_dataloader):
                        step_id = (epoch_id * len(val_dataloader) + val_batch_id) * world_size
                        val_batch_device = map_copy_select_to(
                            val_batch,
                            device=device,
                            dtype=dtype,
                            select=("mask", "pose_repr", "shape", "obj_num", "obj_embedding"),
                        )
                        val_output = model(val_batch_device)
                        val_loss, val_loss_store = model_loss(val_output, val_batch_device)
                        summary_extra_loss(summary_writer, val_loss_store, step_id=step_id, prefix="val")

                        val_pbar.update()
                    val_pbar.close()
                    _logger.info("val epoch %04d conclude", epoch_id)
                    _logger.info("val epoch %04d conclude | loss: %f", epoch_id, loss.item())
                    _logger.info("val epoch %04d detail loss:", epoch_id)
                    for loss_name, loss_node in val_loss_store.items():
                        _logger.info("                     %s: %f", loss_name.ljust(20), float(loss_node))

            # test
            test_freq = run_cfg["test"]["test_freq"]
            if (test_freq is not None and test_freq != -1) and (
                epoch_id == 0 or epoch_id % test_freq == test_freq - 1 or epoch_id == num_epoch - 1
            ):
                with torch.no_grad():
                    test_pbar = tqdm.tqdm(
                        total=len(test_dataloader),
                        position=0,
                        bar_format=pbar_util.fmt,
                        desc=f"test  epoch {epoch_id:>04d}:",
                    )
                    model.eval()
                    for test_batch_id, test_batch in enumerate(test_dataloader):
                        step_id = (epoch_id * len(test_dataloader) + test_batch_id) * world_size
                        test_batch_device = map_copy_select_to(
                            test_batch,
                            device=device,
                            dtype=dtype,
                            select=("mask", "pose_repr", "shape", "obj_num", "obj_embedding"),
                        )
                        test_output = model(test_batch_device)
                        test_loss, test_loss_store = model_loss(test_output, test_batch_device)
                        summary_extra_loss(summary_writer, test_loss_store, step_id=step_id, prefix="test")

                        test_pbar.update()
                    test_pbar.close()
                    _logger.info("test epoch %04d conclude", epoch_id)
                    _logger.info("test epoch %04d conclude | loss: %f", epoch_id, loss.item())
                    _logger.info("test epoch %04d detail loss:", epoch_id)
                    for loss_name, loss_node in test_loss_store.items():
                        _logger.info("                     %s: %f", loss_name.ljust(20), float(loss_node))

    # conclude
    ddp_util.destroy_ddp()


def main():
    config_reg = ConfigRegistry(prog=PROG)
    ckpt.reg_entry(config_reg)
    reg_entry(config_reg)

    parser = argparse.ArgumentParser(prog=PROG)
    config_reg.hook(parser)
    config_reg.parse(parser)

    ckpt_cfg = ckpt.reg_extract(config_reg)
    run_cfg = reg_extract(config_reg)

    device_id_list = run_cfg["runtime"]["device_id"]
    ddp_util.validate_device_id_list(device_id_list)
    world_size = len(device_id_list)
    # replace underlying device
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(el) for el in device_id_list)
    run_cfg["runtime"]["device_id"] = list(range(world_size))

    mp.spawn(run, args=(world_size, ckpt_cfg, run_cfg), nprocs=world_size)


if __name__ == "__main__":
    main()
