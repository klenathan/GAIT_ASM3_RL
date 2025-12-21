"""
Stable-Baselines3 helpers.

This module is intentionally separated from `arena/utils.py` to avoid importing
PyTorch in environment-related code paths.
"""

from __future__ import annotations

from typing import Type, Union

import torch.nn as nn


_ACTIVATION_MAP: dict[str, Type[nn.Module]] = {
    # Common names (case-insensitive)
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
    "leakyrelu": nn.LeakyReLU,
    "silu": nn.SiLU,  # a.k.a. swish in some libs
    "swish": nn.SiLU,
    "gelu": nn.GELU,
    "selu": nn.SELU,
    "sigmoid": nn.Sigmoid,
    "softplus": nn.Softplus,
}


def resolve_activation_fn(activation: Union[str, Type[nn.Module]]) -> Type[nn.Module]:
    """
    Resolve an activation function for SB3 `policy_kwargs["activation_fn"]`.

    SB3 expects a torch.nn.Module *class* (e.g., nn.ReLU), not an instance.
    """
    if isinstance(activation, type) and issubclass(activation, nn.Module):
        return activation

    if not isinstance(activation, str):
        raise TypeError(
            f"activation must be a string or nn.Module class, got {type(activation)!r}"
        )

    key = activation.strip().lower()
    if key in _ACTIVATION_MAP:
        return _ACTIVATION_MAP[key]

    options = ", ".join(sorted({k for k in _ACTIVATION_MAP.keys() if k != "swish"}))
    raise ValueError(
        f"Unknown activation '{activation}'. Supported: {options}"
    )


