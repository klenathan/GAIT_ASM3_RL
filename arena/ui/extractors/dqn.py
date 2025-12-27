"""
DQN extractor for Q-value based algorithms.
"""

from typing import Union, Dict
import numpy as np
import torch

from .model_output import ModelOutput
from .base import BaseExtractor


class DQNExtractor(BaseExtractor):
    """Extractor for DQN (Deep Q-Network) models."""
    
    def supports(self, model) -> bool:
        """Check if model is a DQN variant."""
        model_class = model.__class__.__name__.lower()
        return "dqn" in model_class
    
    def extract_internal(
        self, 
        model, 
        obs: Union[np.ndarray, Dict[str, np.ndarray]], 
        action: int,
        **kwargs
    ) -> ModelOutput:
        """
        Extract Q-values from DQN model.
        
        Args:
            model: DQN model instance
            obs: Observation array (DQN doesn't support dict observations)
            action: Action taken
            **kwargs: Ignored for DQN
            
        Returns:
            ModelOutput with Q-values
            
        Raises:
            ValueError: If dict observation provided (DQN doesn't support)
        """
        # DQN doesn't support dict observations
        if isinstance(obs, dict):
            raise ValueError(
                f"DQN doesn't support dict observations. "
                f"Got dict with keys: {list(obs.keys())}"
            )
        
        with torch.no_grad():
            # Process observation to tensor
            obs_tensor = self._process_observation(obs, model.device)
            
            # Get Q-values from Q-network
            q_values = model.q_net(obs_tensor)
            q_values_np = self._extract_vector(q_values)
            
            return ModelOutput(
                q_values=q_values_np,
                action_taken=action,
                is_q_value=True
            )

