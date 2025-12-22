"""
PPO Trainer implementation.
"""

from typing import Dict, Any
from stable_baselines3 import PPO

from arena.training.base import BaseTrainer
from arena.training.registry import AlgorithmRegistry
from arena.core.config import PPO_DEFAULT, PPO_GPU_DEFAULT
from arena.training.utils import resolve_activation_fn

@AlgorithmRegistry.register("ppo")
class PPOTrainer(BaseTrainer):
    """Proximal Policy Optimization trainer."""
    
    algorithm_name = "ppo"
    algorithm_class = PPO
    policy_type = "MlpPolicy"
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get hyperparameters based on device and env rollout size."""
        hparams = PPO_GPU_DEFAULT if self._is_gpu() else PPO_DEFAULT
        
        # Ensure batch size is valid for the rollout size
        batch_size = self._pick_valid_batch_size(
            hparams.batch_size, 
            hparams.n_steps
        )
        
        if batch_size != hparams.batch_size:
            print(f"Adjusted PPO batch_size {hparams.batch_size} -> {batch_size} to divide rollout size.")
            
        return {
            "learning_rate": hparams.learning_rate,
            "n_steps": hparams.n_steps,
            "batch_size": batch_size,
            "n_epochs": hparams.n_epochs,
            "gamma": hparams.gamma,
            "gae_lambda": hparams.gae_lambda,
            "clip_range": hparams.clip_range,
            "ent_coef": hparams.ent_coef,
            "vf_coef": hparams.vf_coef,
            "max_grad_norm": hparams.max_grad_norm,
            "verbose": hparams.verbose,
        }
    
    def get_policy_kwargs(self) -> Dict[str, Any]:
        """Get net architecture and activation."""
        return {
            "net_arch": self.config.ppo_net_arch,
            "ortho_init": True,
            "activation_fn": resolve_activation_fn(self.config.ppo_activation),
        }
