"""
Utilities for Stable Baselines3.
"""

from typing import Union, Type
import torch.nn as nn

def resolve_activation_fn(activation: Union[str, Type[nn.Module]]) -> Type[nn.Module]:
    """
    Resolve activation function from string or type.
    
    Args:
        activation: String name ('relu', 'tanh', etc.) or nn.Module type
        
    Returns:
        The nn.Module class (not instance)
    """
    if not isinstance(activation, str):
        return activation
        
    activation = activation.lower()
    if activation == "relu":
        return nn.ReLU
    elif activation == "tanh":
        return nn.Tanh
    elif activation == "sigmoid":
        return nn.Sigmoid
    elif activation == "elu":
        return nn.ELU
    elif activation == "selu":
        return nn.SELU
    
    raise ValueError(f"Unknown activation function: {activation}")
