"""
MLP Policy for PufferLib - Basic feedforward actor-critic.
"""

import torch
import torch.nn as nn
import numpy as np
from gymnasium import spaces


class MLPPolicy(nn.Module):
    """
    Multi-Layer Perceptron policy for standard Box observation spaces.
    
    Implements actor-critic architecture with shared feature extractor.
    """
    
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Discrete,
        hidden_size: int = 256,
        num_layers: int = 2,
        activation: str = "ReLU",
    ):
        super().__init__()
        
        self.observation_space = observation_space
        self.action_space = action_space
        
        # Get observation and action dimensions
        self.obs_dim = int(np.prod(observation_space.shape))
        self.num_actions = action_space.n
        
        # Activation function
        activation_fn = getattr(nn, activation)
        
        # Build shared feature extractor
        layers = []
        in_dim = self.obs_dim
        
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_size),
                activation_fn(),
            ])
            in_dim = hidden_size
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Actor head (policy)
        self.actor = nn.Linear(hidden_size, self.num_actions)
        
        # Critic head (value function)
        self.critic = nn.Linear(hidden_size, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
        
        # Smaller initialization for policy head
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.constant_(self.actor.bias, 0.0)
    
    def forward(self, observations):
        """
        Forward pass through the network.
        
        Args:
            observations: Tensor of shape (batch_size, obs_dim)
        
        Returns:
            logits: Tensor of shape (batch_size, num_actions)
            values: Tensor of shape (batch_size, 1)
        """
        # Extract features
        features = self.feature_extractor(observations)
        
        # Get policy logits and value estimate
        logits = self.actor(features)
        values = self.critic(features)
        
        return logits, values
    
    def get_value(self, observations):
        """Get value estimates only (for critic evaluation)."""
        features = self.feature_extractor(observations)
        return self.critic(features)
    
    def get_action_and_value(self, observations, actions=None, deterministic=False):
        """
        Get action distribution and value estimate.
        
        This is the main interface used during training and evaluation.
        
        Args:
            observations: Tensor of shape (batch_size, obs_dim)
            actions: Optional pre-selected actions for log prob calculation
            deterministic: If True, select argmax action instead of sampling
        
        Returns:
            actions: Selected actions
            log_probs: Log probabilities of selected actions
            entropy: Entropy of action distribution
            values: Value estimates
        """
        logits, values = self.forward(observations)
        
        # Create categorical distribution
        dist = torch.distributions.Categorical(logits=logits)
        
        # Sample or take deterministic action
        if actions is None:
            if deterministic:
                actions = logits.argmax(dim=-1)
            else:
                actions = dist.sample()
        
        # Get log probabilities and entropy
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return actions, log_probs, entropy, values.squeeze(-1)
