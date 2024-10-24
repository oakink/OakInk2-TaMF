from __future__ import annotations
import typing
import time
import os
import logging

import typing

if typing.TYPE_CHECKING:
    from typing import Optional

from config_reg import ConfigEntryValueUnspecified, ConfigEntryCallback
import yaml
import json
from ..util.timestamp_util import global_timestamp
import re
from ..util.subst_util import match_special, extract_special_part, replace_from_span

_logger = logging.getLogger()

THIS_FILE = os.path.normcase(os.path.normpath(__file__))
THIS_DIR = os.path.dirname(THIS_FILE)
CURR_WORKING_DIR = os.getcwd()

match_file = re.compile(r"^file:(.*)$")


def load_fileline(file_name):
    res = []
    file_path = os.path.normpath(os.path.abspath(file_name))
    if os.path.exists(file_path):
        with open(file_path, "r") as charstream:
            content = charstream.read()
        for line in content.splitlines():
            line = line.strip()
            res.append(line)
    return res


def dedup_fileine(res):
    return list(dict.fromkeys(res).keys())


class cls__cb__decode_file(ConfigEntryCallback):
    always = True

    @staticmethod
    def __call__(curr_key: str, curr_value: typing.Any, prog: str, dep: typing.Mapping) -> typing.Any:
        if curr_value == ConfigEntryValueUnspecified:
            return curr_value
        else:
            res = []
            for el in curr_value:
                # match for special case
                match_special_res = match_special.fullmatch(el)
                if not match_special_res:
                    res.append(el)
                    continue
                cmd = match_special_res.group(1)
                handled = False
                if not handled:
                    match_file_res = match_file.fullmatch(cmd)
                    if match_file_res:
                        file_name = match_file_res.group(1)
                        res.extend(load_fileline(file_name))
                        handled = True
            # deduplicate res
            res = list(dict.fromkeys(res).keys())
            return res


cb__decode_file = cls__cb__decode_file()


class cls__cb__link_bool_opt(ConfigEntryCallback):
    always = False

    def __init__(self, dependency: str):
        # overwrite
        self.dependency = [dependency]

    def __call__(self, curr_key: str, curr_value: typing.Any, prog: str, dep: typing.Mapping) -> typing.Any:
        assert curr_value == ConfigEntryValueUnspecified
        return dep[self.dependency[0]]


class cls__cb__timestamp(ConfigEntryCallback):
    always = True

    def __init__(self, prefix=None):
        self.prefix = prefix

    def __call__(self, curr_key: str, curr_value: typing.Any, prog: str, dep: typing.Mapping) -> typing.Any:
        if curr_value == ConfigEntryValueUnspecified:
            timestr = time.strftime("%Y_%m%d_%H%M_%S", time.localtime(global_timestamp))
            return f"{self.prefix}__{timestr}" if self.prefix is not None else timestr
        else:
            cmd_list, span_list = extract_special_part(curr_value)
            # process each cmd
            replacement_list = []
            filtered_span_list = []
            for _offset, cmd in enumerate(cmd_list):
                if cmd == "ts:date":
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
