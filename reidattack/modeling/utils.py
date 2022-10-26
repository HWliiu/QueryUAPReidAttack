from collections import defaultdict
from typing import Dict, List

import torch.nn as nn


def get_missing_parameters_message(keys: List[str]) -> str:
    """
    Get a logging-friendly message to report parameter names (keys) that are in
    the model but not found in a checkpoint.

    Args:
        keys (list[str]): List of keys that were not found in the checkpoint.
    Returns:
        str: message.
    """
    groups = _group_checkpoint_keys(keys)
    msg = "Some model parameters or buffers are not found in the checkpoint:\n"
    msg += "\n".join("  " + k + _group_to_str(v) for k, v in groups.items())
    return msg


def get_unexpected_parameters_message(keys: List[str]) -> str:
    """
    Get a logging-friendly message to report parameter names (keys) that are in
    the checkpoint but not found in the model.

    Args:
        keys (list[str]): List of keys that were not found in the model.
    Returns:
        str: message.
    """
    groups = _group_checkpoint_keys(keys)
    msg = "The checkpoint state_dict contains keys that are not used by the model:\n"
    msg += "\n".join("  " + k + _group_to_str(v) for k, v in groups.items())
    return msg


def get_deleted_parameters_message(keys: List[str]) -> str:
    groups = _group_checkpoint_keys(keys)
    msg = "The checkpoint state_dict contains keys that are deleted:\n"
    msg += "\n".join("  " + k + _group_to_str(v) for k, v in groups.items())
    return msg


def _group_checkpoint_keys(keys: List[str]) -> Dict[str, List[str]]:
    """
    Group keys based on common prefixes. A prefix is the string up to the final
    "." in each key.

    Args:
        keys (list[str]): list of parameter names, i.e. keys in the model
            checkpoint dict.
    Returns:
        dict[list]: keys with common prefixes are grouped into lists.
    """
    groups = defaultdict(list)
    for key in keys:
        pos = key.rfind(".")
        if pos >= 0:
            head, tail = key[:pos], [key[pos + 1 :]]
        else:
            head, tail = key, []
        groups[head].extend(tail)
    return groups


def _group_to_str(group: List[str]) -> str:
    """
    Format a group of parameter name suffixes into a loggable string.

    Args:
        group (list[str]): list of parameter name suffixes.
    Returns:
        str: formated string.
    """
    if len(group) == 0:
        return ""

    if len(group) == 1:
        return "." + group[0]

    return ".{" + ", ".join(group) + "}"


class MutiInputSequential(nn.Sequential):
    """Only process the first input except the last module and the last module supports muti-input"""

    def forward(self, input, **kwargs):
        for i, module in enumerate(self):
            if i != len(self) - 1:
                input = module(input)
            else:
                input = module(input, **kwargs)
        return input
