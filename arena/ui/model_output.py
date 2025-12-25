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
        
    def extract(self, model, obs: Union[np.ndarray, Dict[str, np.ndarray]], action: int, lstm_states=None, episode_start=None) -> ModelOutput:
        """
        Extract detailed model info for a single observation.
        
        Args:
            model: The loaded SB3 model (PPO, DQN, etc.)
            obs: The observation vector (numpy array) or dict observation
            action: The action chosen by model.predict()
            lstm_states: Recurrent states if applicable
            episode_start: Episode start flag if applicable
            
        Returns:
            ModelOutput dataclass
        """
        # Determine algorithm type and extract accordingly
        algo_name = model.__class__.__name__.lower()
        
        output = ModelOutput(action_taken=action)
        
        try:
            with torch.no_grad():
                # Handle dictionary observations (for MultiInputPolicy)
                if isinstance(obs, dict):
                    # Convert dict observation to tensors on device
                    # Each value needs a batch dimension (unsqueeze at dim 0)
                    obs_tensor = {}
                    for key, value in obs.items():
                        # Convert to numpy array if not already
                        if not isinstance(value, np.ndarray):
                            value = np.array(value)
                        
                        tensor = torch.as_tensor(value).to(model.device)
                        # Ensure proper shape: if 1D, add batch dim; if already 2D with batch, keep it
                        if tensor.dim() == 0:
                            # Scalar - shouldn't happen but handle it
                            tensor = tensor.unsqueeze(0).unsqueeze(0)
                        elif tensor.dim() == 1:
                            # 1D array - add batch dimension: (N,) -> (1, N)
                            tensor = tensor.unsqueeze(0)
                        elif tensor.dim() > 2:
                            # If somehow 3D+, flatten or take first batch
                            # This shouldn't happen but handle it
                            while tensor.dim() > 2:
                                tensor = tensor[0]
                        # If already 2D with shape (1, N), keep it as is
                        obs_tensor[key] = tensor
                else:
                    # Convert observation to torch tensor
                    # Handle both 1D and 2D inputs
                    tensor = torch.as_tensor(obs).to(model.device)
                    if tensor.dim() == 1:
                        obs_tensor = tensor.unsqueeze(0)  # Add batch dimension
                    elif tensor.dim() == 0:
                        obs_tensor = tensor.unsqueeze(0).unsqueeze(0)
                    else:
                        obs_tensor = tensor
                
                if "dqn" in algo_name:
                    # DQN doesn't support dict observations
                    if isinstance(obs_tensor, dict):
                        print("ModelOutputExtractor: DQN doesn't support dict observations")
                        return output
                    # DQN: Get Q-values
                    q_values = model.q_net(obs_tensor)
                    output.q_values = q_values.cpu().numpy()[0]
                    output.is_q_value = True
                    
                elif any(algo in algo_name for algo in ["ppo", "a2c"]):
                    # Actor-Critic: Get distribution and value
                    if hasattr(model, "policy"):
                        try:
                            if "lstm" in algo_name or (hasattr(model.policy, "recurrent_initial_state") and lstm_states is not None):
                                # Recurrent models
                                # For simplified extraction, we use the policy features
                                # Proper extraction requires passing hidden states
                                # SB3 RecurrentPPO policy.get_distribution takes obs, lstm_states, episode_start
                                dist = model.policy.get_distribution(obs_tensor, lstm_states, episode_start)
                                latent_pi, latent_vf, _ = model.policy.get_latent_features(obs_tensor, lstm_states, episode_start)
                                value = model.policy.value_net(latent_vf)
                            else:
                                # Standard MLP models or MultiInputPolicy
                                dist = model.policy.get_distribution(obs_tensor)
                                value = model.policy.predict_values(obs_tensor)
                            
                            # Extract probabilities if discrete
                            try:
                                if hasattr(dist.distribution, "probs"):
                                    probs_np = dist.distribution.probs.cpu().numpy()
                                    # Handle both 1D and 2D outputs (with/without batch dim)
                                    output.action_probs = probs_np[0] if probs_np.ndim > 1 else probs_np
                                else:
                                    # Fallback if probs aren't directly available (e.g. log_probs)
                                    logits = dist.distribution.logits
                                    probs = torch.exp(logits)
                                    probs_np = probs.cpu().numpy()
                                    output.action_probs = probs_np[0] if probs_np.ndim > 1 else probs_np
                            except Exception as e:
                                # If probability extraction fails, skip it
                                pass
                            
                            # Extract value (handle various output shapes)
                            try:
                                value_np = value.cpu().numpy()
                                if value_np.ndim == 2:
                                    output.value = float(value_np[0, 0])
                                elif value_np.ndim == 1:
                                    output.value = float(value_np[0])
                                else:
                                    output.value = float(value_np.item())
                            except (IndexError, ValueError) as e:
                                # Fallback: try to get first element or item
                                try:
                                    value_np = value.cpu().numpy()
                                    output.value = float(value_np.flat[0])
                                except:
                                    try:
                                        output.value = float(value_np.item())
                                    except:
                                        pass
                            
                            # Extract entropy (handle various output shapes)
                            try:
                                entropy_tensor = dist.entropy()
                                entropy_np = entropy_tensor.cpu().numpy()
                                if entropy_np.ndim == 0:
                                    output.entropy = float(entropy_np.item())
                                elif entropy_np.ndim == 1:
                                    output.entropy = float(entropy_np[0])
                                else:
                                    # Multi-dimensional: take first element
                                    output.entropy = float(entropy_np.flat[0])
                            except (IndexError, ValueError, RuntimeError) as e:
                                # If entropy extraction fails, skip it
                                pass
                        except Exception as e:
                            # If policy operations fail, just skip model output extraction
                            # The error is already printed by the outer exception handler
                            pass
                        
        except Exception as e:
            import traceback
            print(f"ModelOutputExtractor error: {e}")
            # Only print full traceback in debug mode to avoid spam
            if hasattr(self, '_debug') and self._debug:
                traceback.print_exc()
            
        return output
