"""
Custom DQN Policy using the deep Dueling Q-Network architecture.

This policy integrates our custom DuelingQNetwork with Stable-Baselines3's DQN infrastructure,
enabling the use of industry-standard deep architectures while maintaining full compatibility
with SB3's training, saving, and loading mechanisms.
"""

from typing import Any, Dict, List, Optional, Type
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
)
from stable_baselines3.common.type_aliases import Schedule

from arena.training.networks.dueling_qnet import DuelingQNetwork


class ScaledDQNPolicy(DQNPolicy):
    """
    Custom DQN Policy using deep Dueling Q-Network architecture.

    This policy replaces the standard SB3 Q-network with our custom 10-layer
    dueling architecture featuring residual connections and layer normalization.

    Additional policy_kwargs:
        shared_layers: List of layer sizes for shared network (default: [512, 512, 384, 384, 256])
        value_layers: List of layer sizes for value stream (default: [256, 128])
        advantage_layers: List of layer sizes for advantage stream (default: [256, 128])
        use_layer_norm: Whether to use layer normalization (default: True)
        use_residual: Whether to use residual connections (default: True)

    Example:
        >>> policy_kwargs = {
        ...     "shared_layers": [512, 512, 384, 384, 256],
        ...     "value_layers": [256, 128],
        ...     "advantage_layers": [256, 128],
        ...     "activation_fn": nn.SiLU,
        ...     "use_layer_norm": True,
        ...     "use_residual": True,
        ... }
        >>> model = DQN("ScaledDQNPolicy", env, policy_kwargs=policy_kwargs)
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        net_arch: Optional[List[int]] = None,  # Ignored for scaled DQN
        activation_fn: Type[nn.Module] = nn.SiLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        # Custom architecture parameters
        shared_layers: Optional[List[int]] = None,
        value_layers: Optional[List[int]] = None,
        advantage_layers: Optional[List[int]] = None,
        use_layer_norm: bool = True,
        use_residual: bool = True,
    ):
        # Store custom architecture parameters before calling super().__init__
        self.shared_layers = shared_layers or [512, 512, 384, 384, 256]
        self.value_layers = value_layers or [256, 128]
        self.advantage_layers = advantage_layers or [256, 128]
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual

        # Call parent constructor
        # Note: net_arch is ignored for ScaledDQNPolicy as we use custom architecture
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=None,  # We don't use the standard net_arch
            activation_fn=activation_fn,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )

    def make_q_net(self) -> DuelingQNetwork:
        """
        Create the Q-network using our custom DuelingQNetwork architecture.

        This method overrides the parent class to use our deep dueling architecture
        instead of the standard SB3 MLP.

        Returns:
            DuelingQNetwork instance
        """
        # Get observation and action dimensions
        # The features_extractor already flattens the observation
        observation_dim = self.features_extractor.features_dim
        action_dim = int(self.action_space.n)

        # Create our custom dueling Q-network
        q_net = DuelingQNetwork(
            observation_dim=observation_dim,
            action_dim=action_dim,
            shared_layers=self.shared_layers,
            value_layers=self.value_layers,
            advantage_layers=self.advantage_layers,
            activation_fn=self.activation_fn,
            use_layer_norm=self.use_layer_norm,
            use_residual=self.use_residual,
        )

        return q_net

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Build the Q-network and target network.

        This is called by the parent class during initialization.
        We override make_q_net(), so this will use our custom network.
        """
        # Build the main Q-network
        self.q_net = self.make_q_net()

        # Build the target Q-network (copy of main network)
        self.q_net_target = self.make_q_net()
        self.q_net_target.load_state_dict(self.q_net.state_dict())

        # Target network is not trained
        self.q_net_target.set_training_mode(False)

        # Setup optimizer
        if self.optimizer_class is not None:
            self.optimizer = self.optimizer_class(
                self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
            )

        # Print network info
        num_params = self.q_net.get_num_parameters()
        print(f"\n{'=' * 60}")
        print(f"Scaled DQN Policy Initialized")
        print(f"{'=' * 60}")
        print(f"Architecture:")
        print(f"  Shared layers:    {self.shared_layers}")
        print(f"  Value layers:     {self.value_layers}")
        print(f"  Advantage layers: {self.advantage_layers}")
        print(f"  Activation:       {self.activation_fn.__name__}")
        print(f"  Layer norm:       {self.use_layer_norm}")
        print(f"  Residual conn:    {self.use_residual}")
        print(f"\nNetwork Statistics:")
        print(f"  Total parameters: {num_params:,}")
        print(f"  Observation dim:  {self.features_extractor.features_dim}")
        print(f"  Action dim:       {int(self.action_space.n)}")
        print(f"{'=' * 60}\n")


# Register the policy with Stable-Baselines3
# This allows using "ScaledDQNPolicy" as a string in DQN constructor
from stable_baselines3.dqn import policies as dqn_policies

dqn_policies.ScaledDQNPolicy = ScaledDQNPolicy
