"""
Model output extractors for visualization.
Auto-registers extractors on import.
"""

from .model_output import ModelOutput
from .registry import ExtractorRegistry
from .base import BaseExtractor
from .dqn import DQNExtractor
from .actor_critic import ActorCriticExtractor
from .multi_input import MultiInputExtractor
from .recurrent import RecurrentExtractor

# Auto-register extractors with priority
# Higher priority = checked first (for specialized extractors)
ExtractorRegistry.register(DQNExtractor(), priority=0)
ExtractorRegistry.register(ActorCriticExtractor(), priority=0)
ExtractorRegistry.register(MultiInputExtractor(), priority=5)  # Check before ActorCritic
ExtractorRegistry.register(RecurrentExtractor(), priority=10)  # Check first

__all__ = [
    'ModelOutput',
    'ExtractorRegistry',
    'BaseExtractor',
    'DQNExtractor',
    'ActorCriticExtractor',
    'MultiInputExtractor',
    'RecurrentExtractor',
]

