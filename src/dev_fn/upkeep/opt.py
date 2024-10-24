import sys
import shlex
from copy import deepcopy

import logging

_logger = logging.getLogger("__name__")


def arg_to_string(arg):
    res = "{\n"
    for k, v in vars(arg).items():
        res += f"  {k:<20}: {v}\n"
    res += "}"
    return res


def argdict_to_string(argdict, indent=0):
    def format_dict(data, indent):
        indent_str = "  " * indent
        result = "{\n"

        for k, v in data.items():
            result += f"{indent_str}{k:<20}: "
            if isinstance(v, dict):
                result += format_dict(v, indent + 1)
            elif isinstance(v, list):
                result += format_list(v, indent + 1)
            else:
                result += f"{v}\n"

        result += f"{indent_str}}}"
        if indent > 0:
            result += "\n"
        return result

    def format_list(data, indent):
        indent_str = "  " * indent
        result = "[\n"

        for el in data[:20]:
            result += indent_str + " - "
            if isinstance(el, dict):
                result += format_dict(el, indent + 1)
            elif isinstance(el, list):
                result += format_list(el, indent + 1)
            else:
                result += f"{el}\n"
        if len(data) > 20:
            result += f"{indent_str}  ...\n"

        result += f"{indent_str}]\n"
        return result

    if argdict is None:
        return "{}"
    else:
        return format_dict(argdict, indent)


def get_command():
    return shlex.join(deepcopy(sys.argv))


def ask_for_confirm_simple():
    while True:
        ret = input("confirm (y/n)?")
        if ret.upper() == "Y":
            _logger.info("-> confirm")
            return True
        elif ret.upper() == "N":
            _logger.info("-> exit")
            return False
