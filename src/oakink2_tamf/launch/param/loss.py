from __future__ import annotations

import os
import typing

if typing.TYPE_CHECKING:
    from typing import Optional

from config_reg import ConfigRegistry
from config_reg import ConfigEntrySource
from config_reg import ConfigEntryCommandlineSeqPattern, ConfigEntryCommandlineBoolPattern
from config_reg.callback import abspath_callback


def reg_loss_param(config_reg: ConfigRegistry, param_prefix: Optional[str] = None, ws_dir: Optional[str] = None):
    if ws_dir is None:
        ws_dir = ""

    config_reg.register(
        "vpe_path",
        prefix=param_prefix,
        category=str,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        callback=abspath_callback,
        default=None,
    )

    config_reg.register(
        "c_weight_path",
        prefix=param_prefix,
        category=str,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        callback=abspath_callback,
        default=None,
    )

    config_reg.register(
        "coef_rec_joint_loss",
        prefix=param_prefix,
        category=float,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        default=0.0,
    )

    config_reg.register(
        "coef_rec_vert_loss",
        prefix=param_prefix,
        category=float,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        default=0.0,
    )

    config_reg.register(
        "coef_edge_len_loss",
        prefix=param_prefix,
        category=float,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        default=0.0,
    )

    # config_reg.register(
    #     "coef_interaction_dist_loss",
    #     prefix=param_prefix,
    #     category=float,
    #     source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
    #     default=0.0,
    # )

    config_reg.register(
        "coef_dist_h_loss",
        prefix=param_prefix,
        category=float,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        default=0.0,
    )

    config_reg.register(
        "coef_dist_o_loss",
        prefix=param_prefix,
        category=float,
        source=ConfigEntrySource.COMMANDLINE_OVER_CONFIG,
        default=0.0,
    )
