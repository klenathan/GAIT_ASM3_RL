"""
CNN Policy for PufferLib - Multi-channel image observation handling.
"""

import torch
import torch.nn as nn
import numpy as np
from gymnasium import spaces


class CNNPolicy(nn.Module):
    """
    CNN-based policy for Dict observation spaces with image + scalars.
    
    Designed for multi-channel heatmap observations from ArenaCNNEnv:
    - Image: (5, 64, 64) - player, enemies, spawners, projectiles, walls
    - Scalars: (7,) - health, cooldowns, progress, etc.
    
    Implements actor-critic architecture with CNN feature extractor.
    """
    
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Discrete,
        cnn_features: int = 512,
        mlp_hidden: int = 256,
        activation: str = "ReLU",
    ):
        super().__init__()
        
        self.observation_space = observation_space
        self.action_space = action_space
        
        # Get dimensions
        image_shape = observation_space['image'].shape  # (channels, height, width)
        scalar_dim = int(np.prod(observation_space['scalars'].shape))
        self.num_actions = action_space.n
        
        in_channels = image_shape[0]
        
        # Activation function
        activation_fn = getattr(nn, activation)
        
        # CNN Feature Extractor
        # Input: (batch, 5, 64, 64)
        self.cnn = nn.Sequential(
            # Conv block 1: 64x64 -> 32x32
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            activation_fn(),
            
            # Conv block 2: 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            activation_fn(),
            
            # Conv block 3: 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            activation_fn(),
            
            # Conv block 4: 8x8 -> 4x4
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),
            activation_fn(),
            
            # Flatten
            nn.Flatten(),
            
            # Dense layer
            nn.Linear(128 * 4 * 4, cnn_features),
            activation_fn(),
        )
        
        # Scalar feature processor
        self.scalar_net = nn.Sequential(
            nn.Linear(scalar_dim, 128),
            activation_fn(),
        )
        
        # Combined feature processing
        combined_dim = cnn_features + 128
        self.feature_net = nn.Sequential(
            nn.Linear(combined_dim, mlp_hidden),
            activation_fn(),
        )
        
        # Actor head (policy)
        self.actor = nn.Linear(mlp_hidden, self.num_actions)
        
        # Critic head (value function)
        self.critic = nn.Linear(mlp_hidden, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        
        # Smaller initialization for policy head
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.constant_(self.actor.bias, 0.0)
    
    def extract_features(self, observations):
        """
        Extract features from dict observations.
        
        Args:
            observations: Dict with keys 'image' and 'scalars'
                - image: Tensor of shape (batch_size, channels, height, width)
                - scalars: Tensor of shape (batch_size, scalar_dim)
        
        Returns:
            features: Combined feature tensor (batch_size, feature_dim)
        """
        # Process image through CNN
        image_features = self.cnn(observations['image'])
        
        # Process scalars
        scalar_features = self.scalar_net(observations['scalars'])
        
        # Combine features
        combined = torch.cat([image_features, scalar_features], dim=-1)
        
        # Final feature processing
        features = self.feature_net(combined)
        
        return features
    
    def forward(self, observations):
        """
        Forward pass through the network.
        
        Args:
            observations: Dict observation from environment
        
        Returns:
            logits: Tensor of shape (batch_size, num_actions)
            values: Tensor of shape (batch_size, 1)
        """
        # Extract features
        features = self.extract_features(observations)
        
        # Get policy logits and value estimate
        logits = self.actor(features)
        values = self.critic(features)
        
        return logits, values
    
    def get_value(self, observations):
        """Get value estimates only (for critic evaluation)."""
        features = self.extract_features(observations)
        return self.critic(features)
    
    def get_action_and_value(self, observations, actions=None, deterministic=False):
        """
        Get action distribution and value estimate.
        
        This is the main interface used during training and evaluation.
        
        Args:
            observations: Dict observation tensor
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
