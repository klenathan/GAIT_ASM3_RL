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
        
    act_name = activation.strip().lower()
    
    if act_name == "relu":
        return nn.ReLU
    elif act_name == "tanh":
        return nn.Tanh
    elif act_name == "sigmoid":
        return nn.Sigmoid
    elif act_name == "elu":
        return nn.ELU
    elif act_name == "selu":
        return nn.SELU
    elif act_name in ["silu", "swish"]:
        return nn.SiLU
    
    raise ValueError(f"Unknown activation function: '{activation}' (processed as '{act_name}')")
