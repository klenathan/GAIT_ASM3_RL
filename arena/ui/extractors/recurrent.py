"""
Recurrent extractor for LSTM-based algorithms.
"""

from typing import Union, Dict, Optional, Tuple
import numpy as np
import torch

from .model_output import ModelOutput
from .base import BaseExtractor


class RecurrentExtractor(BaseExtractor):
    """Extractor for recurrent actor-critic algorithms (RecurrentPPO with LSTM)."""
    
    def supports(self, model) -> bool:
        """
        Check if model is a recurrent algorithm.
        
        Supports: RecurrentPPO (ppo_lstm)
        """
        model_class = model.__class__.__name__.lower()
        policy_class = model.policy.__class__.__name__.lower()
        
        # Check multiple ways to detect recurrent models
        is_recurrent = (
            "lstm" in model_class or
            "recurrent" in model_class or
            "recurrent" in policy_class or
            hasattr(model.policy, "lstm_actor") or
            hasattr(model.policy, "lstm")
        )
        
        return is_recurrent
    
    def extract_internal(
        self, 
        model, 
        obs: Union[np.ndarray, Dict[str, np.ndarray]], 
        action: int,
        lstm_states: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        episode_start: Optional[Union[bool, np.ndarray]] = None,
        **kwargs
    ) -> ModelOutput:
        """
        Extract action probabilities, value, and entropy from recurrent model.
        
        Args:
            model: RecurrentPPO model instance
            obs: Observation (array or dict)
            action: Action taken
            lstm_states: LSTM hidden states (h, c) - will initialize if None
            episode_start: Whether this is episode start - defaults to False
            **kwargs: Additional ignored parameters
            
        Returns:
            ModelOutput with action_probs, value, and entropy
        """
        with torch.no_grad():
            # Process observation to tensor
            obs_tensor = self._process_observation(obs, model.device)
            
            # Process episode_start flag
            if episode_start is None:
                ep_start = np.array([False])
            elif isinstance(episode_start, (bool, np.bool_)):
                ep_start = np.array([episode_start])
            elif isinstance(episode_start, np.ndarray):
                ep_start = episode_start
            else:
                ep_start = np.array([bool(episode_start)])
            
            # Extract features, distribution, and value through LSTM pipeline
            dist, value = self._forward_through_lstm(
                model, obs_tensor, lstm_states, ep_start
            )
            
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
    
    def _forward_through_lstm(
        self, 
        model, 
        obs_tensor: Union[torch.Tensor, Dict[str, torch.Tensor]], 
        lstm_states: Optional[Tuple], 
        episode_start: np.ndarray
    ) -> Tuple:
        """
        Forward pass through LSTM policy network.
        
        Pipeline: obs -> features_extractor -> LSTM -> mlp_extractor -> action_net/value_net
        
        Args:
            model: RecurrentPPO model
            obs_tensor: Processed observation tensor(s)
            lstm_states: LSTM hidden states or None
            episode_start: Episode start flags
            
        Returns:
            Tuple of (distribution, value_tensor)
        """
        # Step 1: Extract features using the feature extractor
        features = model.policy.features_extractor(obs_tensor)
        
        # Step 2: Determine which LSTM layer to use
        lstm_layer = self._get_lstm_layer(model.policy)
        
        # Step 3: Initialize or convert LSTM states
        lstm_states = self._prepare_lstm_states(
            lstm_states, lstm_layer, features.shape[0], features.device
        )
        
        # Step 4: Process through LSTM
        # Add sequence dimension: (batch, features) -> (batch, 1, features)
        features_seq = features.unsqueeze(1)
        lstm_out, _ = lstm_layer(features_seq, lstm_states)
        # Remove sequence dimension: (batch, 1, hidden) -> (batch, hidden)
        lstm_features = lstm_out.squeeze(1)
        
        # Step 5: Get actor and critic features from MLP extractor
        if hasattr(model.policy, 'mlp_extractor'):
            latent_pi = model.policy.mlp_extractor.forward_actor(lstm_features)
            latent_vf = model.policy.mlp_extractor.forward_critic(lstm_features)
        else:
            # No mlp_extractor, use LSTM output directly
            latent_pi = lstm_features
            latent_vf = lstm_features
        
        # Step 6: Get action distribution
        action_logits = model.policy.action_net(latent_pi)
        dist = model.policy.action_dist.proba_distribution(action_logits)
        
        # Step 7: Get value estimate
        value = model.policy.value_net(latent_vf)
        
        return dist, value
    
    def _get_lstm_layer(self, policy) -> torch.nn.LSTM:
        """
        Get the LSTM layer from policy.
        
        Args:
            policy: RecurrentPPO policy
            
        Returns:
            LSTM layer
            
        Raises:
            AttributeError: If no LSTM layer found
        """
        if hasattr(policy, 'lstm_actor'):
            return policy.lstm_actor
        elif hasattr(policy, 'lstm'):
            return policy.lstm
        else:
            raise AttributeError(
                "No LSTM layer found in policy. "
                "Expected 'lstm_actor' or 'lstm' attribute."
            )
    
    def _prepare_lstm_states(
        self, 
        lstm_states: Optional[Tuple],
        lstm_layer: torch.nn.LSTM,
        batch_size: int,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize or convert LSTM states to proper format.
        
        Args:
            lstm_states: Existing states (h, c) or None
            lstm_layer: LSTM layer
            batch_size: Batch size
            device: Target device
            
        Returns:
            Tuple of (h, c) tensors
        """
        if lstm_states is None:
            # Initialize LSTM states
            hidden_size = lstm_layer.hidden_size
            num_layers = lstm_layer.num_layers
            h_0 = torch.zeros(num_layers, batch_size, hidden_size, device=device)
            c_0 = torch.zeros(num_layers, batch_size, hidden_size, device=device)
            return (h_0, c_0)
        else:
            # Convert lstm_states to torch tensors if they're numpy arrays
            h, c = lstm_states
            if isinstance(h, np.ndarray):
                h = torch.as_tensor(h).to(device)
            if isinstance(c, np.ndarray):
                c = torch.as_tensor(c).to(device)
            return (h, c)
    
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

