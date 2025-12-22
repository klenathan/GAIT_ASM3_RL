"""
Learning rate schedule functions for Stable-Baselines3.

All schedule functions take a single argument `progress_remaining` which
goes from 1.0 (start of training) to 0.0 (end of training).
"""

from typing import Callable, Optional
import math


def linear_schedule(
    lr_start: float, 
    lr_end: float = 0.0,
    warmup_fraction: float = 0.0
) -> Callable[[float], float]:
    """
    Linear learning rate schedule with optional warmup.
    
    Args:
        lr_start: Initial learning rate (after warmup).
        lr_end: Final learning rate at end of training.
        warmup_fraction: Fraction of training for linear warmup (0-1).
    
    Returns:
        A function that takes progress_remaining and returns the current LR.
    """
    def schedule(progress_remaining: float) -> float:
        progress = 1.0 - progress_remaining
        
        # Warmup phase
        if warmup_fraction > 0 and progress < warmup_fraction:
            # Linear warmup from lr_end to lr_start
            warmup_progress = progress / warmup_fraction
            return lr_end + (lr_start - lr_end) * warmup_progress
        
        # Decay phase
        if warmup_fraction > 0:
            # Adjust progress to be relative to post-warmup
            decay_progress = (progress - warmup_fraction) / (1.0 - warmup_fraction)
        else:
            decay_progress = progress
        
        return lr_start + (lr_end - lr_start) * decay_progress
    
    return schedule


def exponential_schedule(
    lr_start: float,
    lr_end: float = 1e-6,
    decay_rate: float = 0.1,
) -> Callable[[float], float]:
    """
    Exponential decay learning rate schedule.
    
    Decays faster initially, then slows down. Good for quick initial exploration.
    
    Args:
        lr_start: Initial learning rate.
        lr_end: Minimum learning rate (floor).
        decay_rate: Controls decay speed (lower = faster decay).
    
    Returns:
        A function that takes progress_remaining and returns the current LR.
    """
    def schedule(progress_remaining: float) -> float:
        progress = 1.0 - progress_remaining
        # Exponential decay: lr = lr_start * (decay_rate)^progress
        # Clamped to lr_end minimum
        lr = lr_start * (decay_rate ** progress)
        return max(lr, lr_end)
    
    return schedule


def cosine_annealing_schedule(
    lr_start: float,
    lr_end: float = 0.0,
    warmup_fraction: float = 0.0,
) -> Callable[[float], float]:
    """
    Cosine annealing learning rate schedule with optional warmup.
    
    Provides smooth, gradual decay. Often leads to better generalization
    as it allows fine-tuning near the end of training.
    
    Args:
        lr_start: Initial learning rate (after warmup).
        lr_end: Final learning rate at end of training.
        warmup_fraction: Fraction of training for linear warmup (0-1).
    
    Returns:
        A function that takes progress_remaining and returns the current LR.
    """
    def schedule(progress_remaining: float) -> float:
        progress = 1.0 - progress_remaining
        
        # Warmup phase
        if warmup_fraction > 0 and progress < warmup_fraction:
            warmup_progress = progress / warmup_fraction
            return lr_end + (lr_start - lr_end) * warmup_progress
        
        # Cosine decay phase
        if warmup_fraction > 0:
            decay_progress = (progress - warmup_fraction) / (1.0 - warmup_fraction)
        else:
            decay_progress = progress
        
        # Cosine annealing: lr = lr_end + 0.5 * (lr_start - lr_end) * (1 + cos(pi * progress))
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
        return lr_end + (lr_start - lr_end) * cosine_factor
    
    return schedule


def constant_schedule(lr: float) -> Callable[[float], float]:
    """
    Constant learning rate (no decay).
    
    Args:
        lr: Learning rate to maintain throughout training.
    
    Returns:
        A function that always returns the same LR.
    """
    def schedule(progress_remaining: float) -> float:
        return lr
    
    return schedule


def get_lr_schedule(
    schedule_type: str,
    lr_start: float,
    lr_end: Optional[float] = None,
    warmup_fraction: float = 0.0,
    **kwargs
) -> Callable[[float], float]:
    """
    Factory function to get a learning rate schedule.
    
    Args:
        schedule_type: One of "constant", "linear", "exponential", "cosine".
        lr_start: Initial/base learning rate.
        lr_end: Final learning rate (defaults to lr_start * 0.1 for decaying schedules).
        warmup_fraction: Fraction of training for warmup.
        **kwargs: Additional arguments for specific schedules.
    
    Returns:
        A callable LR schedule function.
    
    Raises:
        ValueError: If schedule_type is unknown.
    """
    if lr_end is None:
        lr_end = lr_start * 0.1 if schedule_type != "constant" else lr_start
    
    schedules = {
        "constant": lambda: constant_schedule(lr_start),
        "linear": lambda: linear_schedule(lr_start, lr_end, warmup_fraction),
        "exponential": lambda: exponential_schedule(lr_start, lr_end, kwargs.get("decay_rate", 0.1)),
        "cosine": lambda: cosine_annealing_schedule(lr_start, lr_end, warmup_fraction),
    }
    
    if schedule_type not in schedules:
        raise ValueError(
            f"Unknown LR schedule: '{schedule_type}'. "
            f"Available: {list(schedules.keys())}"
        )
    
    return schedules[schedule_type]()
