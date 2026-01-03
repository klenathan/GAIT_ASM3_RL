"""
PPO CNN Trainer implementation.
Uses CNN feature extraction for heatmap observations.
"""

from typing import Dict, Any
from stable_baselines3 import PPO

from arena.training.base import BaseTrainer
from arena.training.registry import AlgorithmRegistry
from arena.core.config import PPO_DEFAULT, PPO_GPU_DEFAULT
from arena.training.utils import resolve_activation_fn
from arena.training.schedules import get_lr_schedule
from arena.training.cnn_extractor import CNNScalarExtractor
from arena.core.environment_cnn import ArenaCNNEnv


@AlgorithmRegistry.register("ppo_cnn")
class PPOCNNTrainer(BaseTrainer):
    """
    PPO trainer with CNN feature extraction for heatmap observations.
    
    Uses ArenaCNNEnv which provides:
    - Multi-channel heatmap image (5 channels, 64x64)
    - Auxiliary scalar features (7 dims)
    
    The CNN processes spatial information (entity positions, threats, walls)
    while scalar features provide precise numerical state info.
    """
    
    algorithm_name = "ppo_cnn"
    algorithm_class = PPO
    policy_type = "MultiInputPolicy"
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get hyperparameters optimized for CNN training."""
        hparams = PPO_GPU_DEFAULT if self._is_gpu() else PPO_DEFAULT
        
        # Use smaller batch sizes for CNN (memory considerations)
        batch_size = min(hparams.batch_size, 128)
        batch_size = self._pick_valid_batch_size(batch_size, hparams.n_steps)
        
        # Learning rate schedule
        lr_schedule = get_lr_schedule(
            schedule_type=self.config.lr_schedule,
            lr_start=hparams.learning_rate,
            lr_end=self.config.lr_end,
            warmup_fraction=self.config.lr_warmup_fraction,
        )
        
        return {
            "learning_rate": lr_schedule,
            "n_steps": hparams.n_steps,
            "batch_size": batch_size,
            "n_epochs": hparams.n_epochs,
            "gamma": hparams.gamma,
            "gae_lambda": hparams.gae_lambda,
            "clip_range": hparams.clip_range,
            "ent_coef": hparams.ent_coef,
            "vf_coef": hparams.vf_coef,
            "max_grad_norm": hparams.max_grad_norm,
            "target_kl": hparams.target_kl,
            "verbose": hparams.verbose,
        }
    
    def get_policy_kwargs(self) -> Dict[str, Any]:
        """Get policy configuration with custom CNN extractor."""
        return {
            "features_extractor_class": CNNScalarExtractor,
            "features_extractor_kwargs": {"features_dim": 160},
            "net_arch": dict(pi=[128, 64], vf=[128, 64]),  # Smaller MLP after extraction
            "activation_fn": resolve_activation_fn(self.config.ppo_activation),
        }
    
    def _make_env_fn(self):
        """Environment factory using ArenaCNNEnv."""
        def _init():
            env = ArenaCNNEnv(
                control_style=self.config.style,
                render_mode="human" if self.config.render else None,
                curriculum_manager=self.curriculum_manager,
            )
            return env
        return _init
