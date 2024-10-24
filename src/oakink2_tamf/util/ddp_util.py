from __future__ import annotations
import typing

if typing.TYPE_CHECKING:
    from typing import Optional


import os
import torch
import torch.distributed as dist
import functools

import logging


def setup_ddp(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    assert rank == torch.distributed.get_rank(), "Something wrong with DDP setup"
    dist.barrier()


def destroy_ddp():
    # dist.barrier()
    dist.destroy_process_group()


def limit_thread_num():
    # tune multi-threading params
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    try:
        import cv2

        cv2.setNumThreads(0)
    except ImportError:
        pass


def validate_device_id_list(device_id_list):
    num_device_available = torch.cuda.device_count()
    for device_id in device_id_list:
        assert int(device_id) < num_device_available, "device_id: {} is not available".format(device_id)


def log_status_update(rank: Optional[int] = None):
    if rank is None or rank == 0:
        return

    root_logger = logging.getLogger()
    # clear all handlers
    for handler in root_logger.handlers:
        root_logger.removeHandler(handler)


def get_dist_info():
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def master_only(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank, _ = get_dist_info()
        if rank == 0:
            return func(*args, **kwargs)

    return wrapper
