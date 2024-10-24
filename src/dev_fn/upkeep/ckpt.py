from __future__ import annotations
import typing
import time
import os
import logging

import typing

if typing.TYPE_CHECKING:
    from typing import Optional

from config_reg import ConfigRegistry, ConfigEntrySource, ConfigEntryCallback, ConfigEntryValueUnspecified
from config_reg import ConfigEntryCommandlineBoolPattern
import yaml
from . import log, opt, rotate_file
from ..util.timestamp_util import global_timestamp
from ..util.subst_util import extract_special_part, replace_from_span

_logger = logging.getLogger()

THIS_FILE = os.path.normcase(os.path.normpath(__file__))
THIS_DIR = os.path.dirname(THIS_FILE)
CURR_WORKING_DIR = os.getcwd()


def reg_entry(config_reg: ConfigRegistry):
    # exp_id
    class cb__exp_id(ConfigEntryCallback):
        always = True

        @staticmethod
        def __call__(curr_key: str, curr_value: typing.Any, prog: str, dep: typing.Mapping) -> typing.Any:
            if curr_value == ConfigEntryValueUnspecified:
                timestr = time.strftime("%Y_%m%d_%H%M_%S", time.localtime(global_timestamp))
                return f"{prog}__{timestr}"
            else:
                cmd_list, span_list = extract_special_part(curr_value)
                # process each cmd
                replacement_list = []
                filtered_span_list = []
                for _offset, cmd in enumerate(cmd_list):
                    if cmd == "prog":
                        filtered_span_list.append(span_list[_offset])
                        replacement_list.append(prog)
                    elif cmd == "ts:date":
                        filtered_span_list.append(span_list[_offset])
                        replacement_list.append(time.strftime("%Y_%m%d", time.localtime(global_timestamp)))
                    elif cmd == "ts" or cmd == "ts:full":
                        filtered_span_list.append(span_list[_offset])
                        replacement_list.append(time.strftime("%Y_%m%d_%H%M_%S", time.localtime(global_timestamp)))
                    else:
                        filtered_span_list.append(span_list[_offset])
                        replacement_list.append("")
                        pass  # todo: check runtime cmd process storage

                # replace
                res = replace_from_span(curr_value, filtered_span_list, replacement_list)
                return res

    _cb__exp_id = cb__exp_id()
    # _cb__exp_id add custom handling...
    config_reg.register(
        "exp_id", prefix=None, category=str, source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG, callback=_cb__exp_id
    )

    # ckpt_path
    class cb__ckpt_path(ConfigEntryCallback):
        dependency: list[str] = ["exp_id"]

        @staticmethod
        def __call__(curr_key: str, curr_value: typing.Any, prog: str, dep: typing.Mapping) -> typing.Any:
            return os.path.normpath(os.path.join(CURR_WORKING_DIR, "common", prog, dep["exp_id"]))

    config_reg.register(
        "ckpt_path", prefix=None, category=str, source=ConfigEntrySource.COMMANDLINE_ONLY, callback=cb__ckpt_path()
    )

    # log_file
    class cb__log_file(ConfigEntryCallback):
        dependency: list[str] = ["ckpt_path"]

        @staticmethod
        def __call__(curr_key: str, curr_value: typing.Any, prog: str, dep: typing.Mapping) -> typing.Any:
            return os.path.join(dep["ckpt_path"], "log.txt")

    config_reg.register(
        "log_file", prefix=None, category=str, source=ConfigEntrySource.COMMANDLINE_ONLY, callback=cb__log_file()
    )

    # commit
    config_reg.register(
        "commit",
        prefix=None,
        category=bool,
        source=ConfigEntrySource.COMMANDLINE_ONLY,
        cmdpattern=ConfigEntryCommandlineBoolPattern.SET_TRUE,
        desc="run in commit mode",
        default=False,
    )


def reg_extract(config_reg: ConfigRegistry):
    key_list = ["exp_id", "ckpt_path", "log_file", "commit"]
    res = {}
    for key in key_list:
        res[key] = config_reg.select(key)
    return res


def ckpt_setup(ckpt_cfg, rank: Optional[int] = None):
    if rank is not None and rank != 0:
        # this is from other process
        return

    commit = ckpt_cfg["commit"]
    if commit:
        os.makedirs(ckpt_cfg["ckpt_path"], exist_ok=True)
        log.enable_file(ckpt_cfg["log_file"])
        _logger.info("commit mode: setup ckpt")

    else:
        _logger.info("dry run mode")

    _logger.info("cmd: %s", opt.get_command())


def handle_save_path(ori_path, *, ckpt_path=None):
    cmd_list, span_list = extract_special_part(ori_path)
    replacement_list = []
    filtered_span_list = []
    for _offset, cmd in enumerate(cmd_list):
        if cmd == "ckpt_path" and ckpt_path is not None:
            replacement_list.append(ckpt_path)
            filtered_span_list.append(span_list[_offset])
        else:
            replacement_list.append("")
            filtered_span_list.append(span_list[_offset])
    res = replace_from_span(ori_path, filtered_span_list, replacement_list)
    return res


def ckpt_opt(ckpt_cfg, rank: Optional[int] = None, **kwarg):
    if not rank:
        commit = ckpt_cfg["commit"]
        if commit:
            opt_file = os.path.join(ckpt_cfg["ckpt_path"], "opt.yml")
            rotate_file.rotate(opt_file)
            with open(opt_file, "w") as ostream:
                yaml.dump(kwarg, ostream, sort_keys=False)
