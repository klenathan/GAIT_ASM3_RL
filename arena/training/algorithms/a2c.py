"""
A2C (Advantage Actor-Critic) Trainer implementation.
"""

from typing import Dict, Any
from stable_baselines3 import A2C

from arena.training.base import BaseTrainer
from arena.training.registry import AlgorithmRegistry
from arena.core.config import A2C_DEFAULT, A2C_GPU_DEFAULT
from arena.training.utils import resolve_activation_fn

@AlgorithmRegistry.register("a2c")
class A2CTrainer(BaseTrainer):
    """Advantage Actor-Critic trainer."""
    
    algorithm_name = "a2c"
    algorithm_class = A2C
    policy_type = "MlpPolicy"
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get hyperparameters based on device."""
        hparams = A2C_GPU_DEFAULT if self._is_gpu() else A2C_DEFAULT
            
        return {
            "learning_rate": hparams.learning_rate,
            "n_steps": hparams.n_steps,
            "gamma": hparams.gamma,
            "gae_lambda": hparams.gae_lambda,
            "ent_coef": hparams.ent_coef,
            "vf_coef": hparams.vf_coef,
            "max_grad_norm": hparams.max_grad_norm,
            "rms_prop_eps": hparams.rms_prop_eps,
            "use_rms_prop": hparams.use_rms_prop,
            "normalize_advantage": hparams.normalize_advantage,
            "verbose": hparams.verbose,
        }
    
    def get_policy_kwargs(self) -> Dict[str, Any]:
        """Get net architecture and activation."""
        return {
            "net_arch": self.config.a2c_net_arch,
            "ortho_init": True,
            "activation_fn": resolve_activation_fn(self.config.a2c_activation),
        }
