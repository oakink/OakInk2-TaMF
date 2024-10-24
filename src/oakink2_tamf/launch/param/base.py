from __future__ import annotations

import os
import typing

if typing.TYPE_CHECKING:
    from typing import Optional

from config_reg import ConfigRegistry
from config_reg import ConfigEntrySource
from config_reg import ConfigEntryCommandlineSeqPattern, ConfigEntryCommandlineBoolPattern
from config_reg.callback import abspath_callback

from dev_fn.upkeep.config import cb__decode_file, cls__cb__link_bool_opt


def reg_base_param(config_reg: ConfigRegistry, param_prefix: Optional[str] = None, ws_dir: Optional[str] = None):
    if ws_dir is None:
        ws_dir = ""

    # data config
    config_reg.register(
        "data_prefix",
        prefix=param_prefix,
        category=str,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        callback=abspath_callback,
        default=f"{ws_dir}/data",
        required=True,
    )
    config_reg.register(
        "process_range",
        prefix=param_prefix,
        category=list[str],
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        cmdpattern=ConfigEntryCommandlineSeqPattern.COLON_SEP,
        callback=cb__decode_file,
        required=True,
    )
