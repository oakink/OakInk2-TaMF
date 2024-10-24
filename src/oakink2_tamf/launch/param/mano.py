from __future__ import annotations

import os
import typing

if typing.TYPE_CHECKING:
    from typing import Optional

from config_reg import ConfigRegistry
from config_reg import ConfigEntrySource
from config_reg import ConfigEntryCommandlineSeqPattern, ConfigEntryCommandlineBoolPattern
from config_reg.callback import abspath_callback


def reg_mano_param(config_reg: ConfigRegistry, param_prefix: Optional[str] = None, ws_dir: Optional[str] = None):
    if ws_dir is None:
        ws_dir = ""

    config_reg.register(
        "mano_path",
        prefix=param_prefix,
        category=str,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        callback=abspath_callback,
        default=f"{ws_dir}/asset/mano_v1_2",
    )
