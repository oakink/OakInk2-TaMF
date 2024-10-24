from __future__ import annotations

import os
import typing

if typing.TYPE_CHECKING:
    from typing import Optional

from config_reg import ConfigRegistry
from config_reg import ConfigEntrySource
from config_reg import ConfigEntryCommandlineSeqPattern, ConfigEntryCommandlineBoolPattern
from config_reg.callback import abspath_callback

def reg_model_param(config_reg: ConfigRegistry, param_prefix: Optional[str] = None, ws_dir: Optional[str] = None):
    if ws_dir is None:
        ws_dir = ""

    config_reg.register(
        "input_dim",
        prefix=param_prefix,
        category=int,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        default=99,
    )
    config_reg.register(
        "obj_input_dim",
        prefix=param_prefix,
        category=int,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        default=9,
    )
    config_reg.register(
        "hand_shape_dim",
        prefix=param_prefix,
        category=int,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        default=10,
    )
    config_reg.register(
        "obj_embed_dim",
        prefix=param_prefix,
        category=int,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        default=768,
    )
    config_reg.register(
        "latent_dim",
        prefix=param_prefix,
        category=int,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        default=256,
    )
    config_reg.register(
        "ff_size",
        prefix=param_prefix,
        category=int,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        default=1024,
    )
    config_reg.register(
        "num_layers",
        prefix=param_prefix,
        category=int,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        default=8,
    )
    config_reg.register(
        "num_heads",
        prefix=param_prefix,
        category=int,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        default=4,
    )
    config_reg.register(
        "dropout",
        prefix=param_prefix,
        category=float,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        default=0.1,
    )
    config_reg.register(
        "activation",
        prefix=param_prefix,
        category=str,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        default="gelu",
    )