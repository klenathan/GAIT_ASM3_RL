"""
Actor-Critic extractor for standard MLP policy-based algorithms.
"""

from typing import Union, Dict
import numpy as np
import torch

from .model_output import ModelOutput
from .base import BaseExtractor


class ActorCriticExtractor(BaseExtractor):
    """Extractor for actor-critic algorithms (PPO, A2C) with MlpPolicy."""
    
    def supports(self, model) -> bool:
        """
        Check if model is a standard actor-critic algorithm.
        
        Supports: PPO, A2C (but not recurrent or dict variants)
        """
        model_class = model.__class__.__name__.lower()
        policy_class = model.policy.__class__.__name__.lower()
        
        # Check if it's PPO or A2C
        is_actor_critic = any(algo in model_class for algo in ["ppo", "a2c"])
        
        # Exclude recurrent models (handled by RecurrentExtractor)
        is_recurrent = (
            "lstm" in model_class or
            "recurrent" in policy_class or
            hasattr(model.policy, "lstm_actor") or
            hasattr(model.policy, "lstm")
        )
        
        # Exclude MultiInputPolicy (handled by MultiInputExtractor)
        is_multi_input = "multiinput" in policy_class
        
        return is_actor_critic and not is_recurrent and not is_multi_input
    
    def extract_internal(
        self, 
        model, 
        obs: Union[np.ndarray, Dict[str, np.ndarray]], 
        action: int,
        **kwargs
    ) -> ModelOutput:
        """
        Extract action probabilities, value, and entropy from actor-critic model.
        
        Args:
            model: PPO or A2C model instance
            obs: Observation (array or dict for MultiInputPolicy)
            action: Action taken
            **kwargs: Ignored for standard actor-critic
            
        Returns:
            ModelOutput with action_probs, value, and entropy
        """
        with torch.no_grad():
            # Process observation to tensor
            obs_tensor = self._process_observation(obs, model.device)
            
            # Get action distribution
            dist = self._extract_distribution(model, obs_tensor)
            
            # Get value estimate
            value = self._extract_value(model, obs_tensor)
            
            # Extract action probabilities
            action_probs = self._extract_action_probs(dist)
            
            # Extract entropy
            entropy = self._extract_entropy(dist)
            
            return ModelOutput(
                action_probs=action_probs,
                value=self._extract_scalar(value),
                entropy=entropy,
                action_taken=action,
                is_q_value=False
            )
    
    def _extract_distribution(self, model, obs_tensor):
        """
        Get action distribution from policy.
        
        Args:
            model: SB3 model
            obs_tensor: Processed observation tensor
            
        Returns:
            Action distribution
        """
        return model.policy.get_distribution(obs_tensor)
    
    def _extract_value(self, model, obs_tensor):
        """
        Get value estimate from policy.
        
        Args:
            model: SB3 model
            obs_tensor: Processed observation tensor
            
        Returns:
            Value tensor
        """
        return model.policy.predict_values(obs_tensor)
    
    def _extract_action_probs(self, dist) -> np.ndarray:
        """
        Extract action probabilities from distribution.
        
        Args:
            dist: Action distribution from policy
            
        Returns:
            1D numpy array of action probabilities
        """
        if hasattr(dist.distribution, "probs"):
            probs = dist.distribution.probs
        else:
            # Fallback: compute from logits
            logits = dist.distribution.logits
            probs = torch.exp(logits)
        
        return self._extract_vector(probs)
    
    def _extract_entropy(self, dist) -> float:
        """
        Extract entropy from distribution.
        
        Args:
            dist: Action distribution from policy
            
        Returns:
            Entropy value
        """
        entropy_tensor = dist.entropy()
        return self._extract_scalar(entropy_tensor)

