"""
Extraction utilities for model output visualization.
Supports action probabilities (PPO/A2C), Q-values (DQN), and value estimates.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Any
import numpy as np
import torch

@dataclass
class ModelOutput:
    """Container for model prediction metadata."""
    action_probs: Optional[np.ndarray] = None
    q_values: Optional[np.ndarray] = None
    value: Optional[float] = None
    entropy: Optional[float] = None
    action_taken: Optional[int] = None
    is_q_value: bool = False
    
    @property
    def labels(self) -> List[str]:
        """Names of actions; to be filled by the renderer based on style."""
        return []

class ModelOutputExtractor:
    """Handles introspection of SB3 models to retrieve distribution data."""
    
    def __init__(self):
        pass
        
    def extract(self, model, obs: np.ndarray, action: int, lstm_states=None, episode_start=None) -> ModelOutput:
        """
        Extract detailed model info for a single observation.
        
        Args:
            model: The loaded SB3 model (PPO, DQN, etc.)
            obs: The observation vector
            action: The action chosen by model.predict()
            lstm_states: Recurrent states if applicable
            episode_start: Episode start flag if applicable
            
        Returns:
            ModelOutput dataclass
        """
        # Convert observation to torch tensor
        obs_tensor = torch.as_tensor(obs).view(1, -1).to(model.device)
        
        # Determine algorithm type and extract accordingly
        algo_name = model.__class__.__name__.lower()
        
        output = ModelOutput(action_taken=action)
        
        try:
            with torch.no_grad():
                if "dqn" in algo_name:
                    # DQN: Get Q-values
                    q_values = model.q_net(obs_tensor)
                    output.q_values = q_values.cpu().numpy()[0]
                    output.is_q_value = True
                    
                elif any(algo in algo_name for algo in ["ppo", "a2c"]):
                    # Actor-Critic: Get distribution and value
                    if hasattr(model, "policy"):
                        if "lstm" in algo_name or (hasattr(model.policy, "recurrent_initial_state") and lstm_states is not None):
                            # Recurrent models
                            # For simplified extraction, we use the policy features
                            # Proper extraction requires passing hidden states
                            # SB3 RecurrentPPO policy.get_distribution takes obs, lstm_states, episode_start
                            dist = model.policy.get_distribution(obs_tensor, lstm_states, episode_start)
                            latent_pi, latent_vf, _ = model.policy.get_latent_features(obs_tensor, lstm_states, episode_start)
                            value = model.policy.value_net(latent_vf)
                        else:
                            # Standard MLP models
                            dist = model.policy.get_distribution(obs_tensor)
                            value = model.policy.predict_values(obs_tensor)
                            
                        # Extract probabilities if discrete
                        if hasattr(dist.distribution, "probs"):
                            output.action_probs = dist.distribution.probs.cpu().numpy()[0]
                        else:
                            # Fallback if probs aren't directly available (e.g. log_probs)
                            probs = torch.exp(dist.distribution.logits)
                            output.action_probs = probs.cpu().numpy()[0]
                            
                        output.value = float(value.cpu().numpy()[0][0])
                        output.entropy = float(dist.entropy().cpu().numpy()[0])
                        
        except Exception as e:
            print(f"ModelOutputExtractor error: {e}")
            
        return output
