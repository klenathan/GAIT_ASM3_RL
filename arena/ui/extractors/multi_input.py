"""
Multi-Input extractor for dictionary observation space algorithms.
"""

from typing import Union, Dict
import numpy as np

from .model_output import ModelOutput
from .actor_critic import ActorCriticExtractor


class MultiInputExtractor(ActorCriticExtractor):
    """
    Extractor for actor-critic algorithms with MultiInputPolicy.
    
    Handles dictionary observations (e.g., PPO_DICT).
    Inherits distribution and value extraction from ActorCriticExtractor.
    """
    
    def supports(self, model) -> bool:
        """
        Check if model uses MultiInputPolicy.
        
        Supports: PPO with MultiInputPolicy (ppo_dict)
        """
        model_class = model.__class__.__name__.lower()
        policy_class = model.policy.__class__.__name__.lower()
        
        # Check if it's PPO or A2C
        is_actor_critic = any(algo in model_class for algo in ["ppo", "a2c"])
        
        # Check if using MultiInputPolicy
        is_multi_input = "multiinput" in policy_class
        
        # Exclude recurrent models (handled by RecurrentExtractor)
        is_recurrent = (
            "lstm" in model_class or
            "recurrent" in policy_class or
            hasattr(model.policy, "lstm_actor") or
            hasattr(model.policy, "lstm")
        )
        
        return is_actor_critic and is_multi_input and not is_recurrent
    
    # Note: extract_internal() is inherited from ActorCriticExtractor
    # The _process_observation() method in BaseExtractor already handles
    # dict observations correctly, so no override is needed.

