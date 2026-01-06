"""
Scaled DQN Trainer implementation with deep 10-layer dueling architecture.

This trainer uses an industry-standard deep Q-network with:
- 10 layers (5 shared + 3 value + 3 advantage streams)
- Dueling architecture for better value/advantage separation
- Residual connections for gradient flow
- Layer normalization for training stability
- ~1.2M parameters (6x more than standard DQN)
- Optimized hyperparameters for deep networks
"""

from typing import Dict, Any
from stable_baselines3 import DQN

from arena.training.base import BaseTrainer
from arena.training.registry import AlgorithmRegistry
from arena.core.config import SCALED_DQN_DEFAULT, SCALED_DQN_GPU_DEFAULT
from arena.training.utils import resolve_activation_fn
from arena.training.schedules import get_lr_schedule


@AlgorithmRegistry.register("scaled_dqn")
class ScaledDQNTrainer(BaseTrainer):
    """
    Scaled Deep Q-Network trainer with 10-layer dueling architecture.

    Architecture:
        - Shared network: 5 layers [512, 512, 384, 384, 256] with residual connections
        - Value stream: 3 layers [256, 128, 1]
        - Advantage stream: 3 layers [256, 128, num_actions]
        - Layer normalization throughout
        - SiLU activation functions
        - ~1.2M total parameters

    Key Improvements over Standard DQN:
        - 6x more parameters for complex function approximation
        - Dueling architecture separates state value from action advantages
        - Residual connections enable training of deeper networks
        - Layer normalization stabilizes deep network training
        - Optimized hyperparameters (lower LR, larger batches, more warmup)

    Usage:
        python arena/train.py --algo scaled_dqn --style 1 --steps 2000000
    """

    algorithm_name = "scaled_dqn"
    algorithm_class = DQN
    policy_type = "ScaledDQNPolicy"  # Our custom policy

    def get_hyperparameters(self) -> Dict[str, Any]:
        """
        Get hyperparameters optimized for deep networks.

        Returns:
            Dictionary of hyperparameters for SB3 DQN
        """
        # Use GPU-optimized hyperparameters if available
        hparams = SCALED_DQN_GPU_DEFAULT if self._is_gpu() else SCALED_DQN_DEFAULT

        # Build learning rate schedule
        lr_schedule = get_lr_schedule(
            schedule_type=self.config.lr_schedule,
            lr_start=hparams.learning_rate,
            lr_end=self.config.lr_end,
            warmup_fraction=self.config.lr_warmup_fraction,
        )

        # Convert dataclass to dict for SB3
        return {
            "learning_rate": lr_schedule,
            "buffer_size": hparams.buffer_size,
            "batch_size": hparams.batch_size,
            "gamma": hparams.gamma,
            "exploration_fraction": hparams.exploration_fraction,
            "exploration_initial_eps": hparams.exploration_initial_eps,
            "exploration_final_eps": hparams.exploration_final_eps,
            "target_update_interval": hparams.target_update_interval,
            "train_freq": hparams.train_freq,
            "gradient_steps": hparams.gradient_steps,
            "learning_starts": hparams.learning_starts,
            "max_grad_norm": hparams.max_grad_norm,
            "verbose": hparams.verbose,
        }

    def get_policy_kwargs(self) -> Dict[str, Any]:
        """
        Get policy configuration for custom deep dueling architecture.

        Returns:
            Dictionary of policy_kwargs for ScaledDQNPolicy
        """
        return {
            "shared_layers": self.config.scaled_dqn_shared_layers,
            "value_layers": self.config.scaled_dqn_value_layers,
            "advantage_layers": self.config.scaled_dqn_advantage_layers,
            "activation_fn": resolve_activation_fn(self.config.scaled_dqn_activation),
            "use_layer_norm": self.config.scaled_dqn_use_layer_norm,
            "use_residual": self.config.scaled_dqn_use_residual,
        }
