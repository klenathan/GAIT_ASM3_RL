"""
DQN Trainer implementation.
"""

from typing import Dict, Any
from stable_baselines3 import DQN

from arena.training.base import BaseTrainer
from arena.training.registry import AlgorithmRegistry
from arena.core.config import DQN_DEFAULT, DQN_GPU_DEFAULT
from arena.training.utils import resolve_activation_fn
from arena.training.schedules import get_lr_schedule

@AlgorithmRegistry.register("dqn")
class DQNTrainer(BaseTrainer):
    """Deep Q-Network trainer."""
    
    algorithm_name = "dqn"
    algorithm_class = DQN
    policy_type = "MlpPolicy"
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get hyperparameters based on device."""
        hparams = DQN_GPU_DEFAULT if self._is_gpu() else DQN_DEFAULT
        
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
            "verbose": hparams.verbose,
        }
    
    def get_policy_kwargs(self) -> Dict[str, Any]:
        """Get net architecture and activation."""
        return {
            "net_arch": self.config.dqn_hidden_layers,
            "activation_fn": resolve_activation_fn(self.config.dqn_activation),
        }
