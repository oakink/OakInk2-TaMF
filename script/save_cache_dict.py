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
from oakink2_tamf.launch.param import reg_base_param
from dev_fn.util.console_io import suppress_trimesh_logging

from oakink2_tamf.dataset.interaction_segment import InteractionSegmentData
import random
from dev_fn.util import random_util

_logger = logging.getLogger(__name__)

PROG = os.path.splitext(os.path.basename(__file__))[0]
MOCAP_WS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))

PARAM_PREFIX__DATA = "data"
PARAM_PREFIX__INFER = "infer"
PARAM_PREFIX__RUNTIME = "runtime"


def reg_entry(config_reg: ConfigRegistry):
    # override default
    config_reg.meta_info["exp_id"].default = "main"

    # base
    reg_base_param(config_reg, PARAM_PREFIX__DATA, MOCAP_WS_DIR)

    # split name
    config_reg.register(
        "split_name",
        prefix=PARAM_PREFIX__DATA,
        category=str,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        required=True,
    )


def reg_extract(config_reg: ConfigRegistry):
    res = {}
    for _p in [PARAM_PREFIX__DATA, PARAM_PREFIX__INFER, PARAM_PREFIX__RUNTIME]:
        try:
            res[_p] = config_reg.select(_p)
        except KeyError:
            pass
    return res


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

    # dataset
    process_range_list = run_cfg["data"]["process_range"]
    all_dataset = InteractionSegmentData(
        process_range_list=process_range_list,
        data_prefix=run_cfg["data"]["data_prefix"],
        enable_obj_model=True,
    )
    all_dataset_cache_dict = all_dataset.get_cache()

    if commit:
        cache_key = run_cfg["data"]["split_name"]
        cache_path = os.path.join(ckpt_path, "cache", f"{cache_key}.pkl")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as ofstream:
            pickle.dump(all_dataset_cache_dict, ofstream)


if __name__ == "__main__":
    main()
