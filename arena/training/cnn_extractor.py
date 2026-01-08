"""
Custom CNN Feature Extractor for heatmap + scalar observations.
"""

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CNNScalarExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for Dict observations with image + scalars.
    
    Processes:
    - "image": Multi-channel heatmap (5, 64, 64) through CNN
    - "scalars": Auxiliary features (7,) through MLP
    
    Outputs combined feature vector for PPO policy.
    """
    
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 160):
        # We compute the features_dim dynamically
        super().__init__(observation_space, features_dim)
        
        # Get observation shapes
        image_shape = observation_space["image"].shape  # (5, 64, 64)
        scalar_shape = observation_space["scalars"].shape  # (7,)
        
        n_channels = image_shape[0]  # 5
        scalar_dim = scalar_shape[0]  # 7
        
        # CNN branch for image processing
        # Input: (batch, 5, 64, 64)
        self.cnn = nn.Sequential(
            # Layer 1: 5 -> 32 channels, 64 -> 31 (with kernel 3, stride 1)
            nn.Conv2d(n_channels, 32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 31 -> 15
            
            # Layer 2: 32 -> 64 channels
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 13 -> 6
            
            # Layer 3: 64 -> 64 channels
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # Output: (batch, 64, 4, 4)
            
            nn.Flatten(),
        )
        
        # Compute CNN output size dynamically
        with torch.no_grad():
            sample_input = torch.zeros(1, *image_shape)
            cnn_output_dim = self.cnn(sample_input).shape[1]
        
        # CNN head to reduce dimensionality
        self.cnn_head = nn.Sequential(
            nn.Linear(cnn_output_dim, 128),
            nn.ReLU(),
        )
        
        # MLP branch for scalar features
        self.scalar_mlp = nn.Sequential(
            nn.Linear(scalar_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        
        # Combined output dimension
        self._features_dim = 128 + 32  # CNN (128) + Scalars (32) = 160
    
    def forward(self, observations: dict) -> torch.Tensor:
        """
        Process observations through CNN and MLP branches.
        
        Args:
            observations: Dict with "image" and "scalars" tensors
            
        Returns:
            Combined feature tensor of shape (batch, 160)
        """
        # Process image through CNN
        image = observations["image"]
        scalars = observations["scalars"]
        
        # Handle single sample (no batch dimension) - add batch dim if needed
        squeeze_output = False
        if image.dim() == 3:
            image = image.unsqueeze(0)
            squeeze_output = True
        if scalars.dim() == 1:
            scalars = scalars.unsqueeze(0)
        
        cnn_features = self.cnn(image)
        cnn_features = self.cnn_head(cnn_features)
        
        # Process scalars through MLP
        scalar_features = self.scalar_mlp(scalars)
        
        # Concatenate features
        combined = torch.cat([cnn_features, scalar_features], dim=1)
        
        # Remove batch dim if we added it
        if squeeze_output:
            combined = combined.squeeze(0)
        
        return combined
