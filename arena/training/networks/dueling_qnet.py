"""
Dueling Q-Network with deep architecture, residual connections, and layer normalization.

This implementation provides a 10-layer dueling architecture optimized for deep RL:
- 5 shared feature extraction layers with residual connections
- 3 value stream layers
- 3 advantage stream layers
- Layer normalization for training stability
- Configurable architecture and activation functions
"""

from typing import List, Type, Optional
import torch
import torch.nn as nn
import numpy as np


class ResidualBlock(nn.Module):
    """
    Residual block with layer normalization and activation.
    Implements: output = activation(LayerNorm(Linear(x))) + x
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation_fn: Type[nn.Module],
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.layer_norm = (
            nn.LayerNorm(out_features) if use_layer_norm else nn.Identity()
        )
        self.activation = activation_fn()

        # Projection layer if dimensions don't match
        self.projection = None
        if in_features != out_features:
            self.projection = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.projection is None else self.projection(x)
        out = self.linear(x)
        out = self.layer_norm(out)
        out = self.activation(out)
        return out + identity


class DuelingQNetwork(nn.Module):
    """
    Deep Dueling Q-Network with 10 layers.

    Architecture:
        Shared Network (5 layers):
            Input -> [512] -> [512 + Residual] -> [384] -> [384 + Residual] -> [256]

        Value Stream (3 layers):
            [256] -> [256] -> [128] -> [1]

        Advantage Stream (3 layers):
            [256] -> [256] -> [128] -> [num_actions]

        Output: Q(s,a) = V(s) + (A(s,a) - mean(A(s,·)))

    Args:
        observation_space: Gym observation space
        action_space: Gym action space (must be Discrete)
        shared_layers: List of hidden layer sizes for shared network
        value_layers: List of hidden layer sizes for value stream
        advantage_layers: List of hidden layer sizes for advantage stream
        activation_fn: Activation function class (e.g., nn.ReLU, nn.SiLU)
        use_layer_norm: Whether to use layer normalization
        use_residual: Whether to use residual connections (every 2 layers)
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        shared_layers: List[int] = None,
        value_layers: List[int] = None,
        advantage_layers: List[int] = None,
        activation_fn: Type[nn.Module] = nn.SiLU,
        use_layer_norm: bool = True,
        use_residual: bool = True,
    ):
        super().__init__()

        # Default architecture
        if shared_layers is None:
            shared_layers = [512, 512, 384, 384, 256]
        if value_layers is None:
            value_layers = [256, 128]
        if advantage_layers is None:
            advantage_layers = [256, 128]

        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.use_residual = use_residual

        # Build shared feature extractor
        self.shared_net = self._build_shared_network(
            observation_dim,
            shared_layers,
            activation_fn,
            use_layer_norm,
            use_residual,
        )

        # Build value stream
        value_input_dim = shared_layers[-1]
        self.value_net = self._build_stream(
            value_input_dim,
            value_layers,
            1,  # Output single value
            activation_fn,
            use_layer_norm,
        )

        # Build advantage stream
        advantage_input_dim = shared_layers[-1]
        self.advantage_net = self._build_stream(
            advantage_input_dim,
            advantage_layers,
            action_dim,
            activation_fn,
            use_layer_norm,
        )

        # Initialize weights
        self._initialize_weights()

    def _build_shared_network(
        self,
        input_dim: int,
        layer_sizes: List[int],
        activation_fn: Type[nn.Module],
        use_layer_norm: bool,
        use_residual: bool,
    ) -> nn.ModuleList:
        """Build shared feature extraction network with residual connections."""
        layers = nn.ModuleList()

        prev_dim = input_dim
        for i, layer_size in enumerate(layer_sizes):
            # Use residual block every 2 layers (after layers 1, 3, etc.)
            use_residual_here = (
                use_residual and i > 0 and i % 2 == 1 and prev_dim == layer_size
            )

            if use_residual_here:
                layers.append(
                    ResidualBlock(
                        prev_dim,
                        layer_size,
                        activation_fn,
                        use_layer_norm,
                    )
                )
            else:
                # Standard layer
                layers.append(nn.Linear(prev_dim, layer_size))
                if use_layer_norm:
                    layers.append(nn.LayerNorm(layer_size))
                layers.append(activation_fn())

            prev_dim = layer_size

        return layers

    def _build_stream(
        self,
        input_dim: int,
        layer_sizes: List[int],
        output_dim: int,
        activation_fn: Type[nn.Module],
        use_layer_norm: bool,
    ) -> nn.Sequential:
        """Build value or advantage stream."""
        layers = []

        prev_dim = input_dim
        for layer_size in layer_sizes:
            layers.append(nn.Linear(prev_dim, layer_size))
            if use_layer_norm:
                layers.append(nn.LayerNorm(layer_size))
            layers.append(activation_fn())
            prev_dim = layer_size

        # Output layer (no activation or normalization)
        layers.append(nn.Linear(prev_dim, output_dim))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize network weights using He initialization for ReLU-family activations."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # He initialization for ReLU-family activations
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_in", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the dueling network.

        Args:
            obs: Observation tensor of shape (batch_size, observation_dim)

        Returns:
            Q-values of shape (batch_size, action_dim)
        """
        # Shared feature extraction
        features = obs
        for layer in self.shared_net:
            features = layer(features)

        # Value stream: V(s)
        value = self.value_net(features)

        # Advantage stream: A(s,a)
        advantage = self.advantage_net(features)

        # Combine using dueling formula: Q(s,a) = V(s) + (A(s,a) - mean(A(s,·)))
        # This forces the value stream to learn state value independently of actions
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values

    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def set_training_mode(self, mode: bool) -> None:
        """
        Set training mode for the network.
        
        Args:
            mode: If True, set to training mode. If False, set to eval mode.
        """
        if mode:
            self.train()
        else:
            self.eval()

    def _predict(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """
        Predict actions from observations.
        
        Args:
            obs: Observation tensor of shape (batch_size, observation_dim)
            deterministic: If True, return argmax action. If False, sample from epsilon-greedy
                          (note: epsilon-greedy is handled by the policy, so this just returns argmax)
        
        Returns:
            Action tensor of shape (batch_size,)
        """
        q_values = self(obs)
        # For DQN, deterministic=True means greedy (argmax), False is handled by epsilon-greedy
        # The actual epsilon-greedy exploration is handled by the DQN algorithm
        return q_values.argmax(dim=1)


def test_dueling_qnetwork():
    """Test function to verify network architecture."""
    print("Testing DuelingQNetwork...")

    # Test with Arena environment dimensions
    obs_dim = 44
    action_dim = 5  # Style 1
    batch_size = 32

    # Create network
    network = DuelingQNetwork(
        observation_dim=obs_dim,
        action_dim=action_dim,
        shared_layers=[512, 512, 384, 384, 256],
        value_layers=[256, 128],
        advantage_layers=[256, 128],
        activation_fn=nn.SiLU,
        use_layer_norm=True,
        use_residual=True,
    )

    print(f"Network created successfully!")
    print(f"Total parameters: {network.get_num_parameters():,}")

    # Test forward pass
    obs = torch.randn(batch_size, obs_dim)
    q_values = network(obs)

    assert q_values.shape == (batch_size, action_dim), (
        f"Expected shape ({batch_size}, {action_dim}), got {q_values.shape}"
    )

    print(f"Forward pass successful! Output shape: {q_values.shape}")
    print(f"Sample Q-values: {q_values[0].detach().numpy()}")

    # Test gradient flow
    loss = q_values.sum()
    loss.backward()

    has_gradients = all(
        p.grad is not None for p in network.parameters() if p.requires_grad
    )
    assert has_gradients, "Some parameters don't have gradients!"

    print("Gradient flow verified!")
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_dueling_qnetwork()
