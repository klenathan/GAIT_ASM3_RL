"""
PPO LSTM (Recurrent) Trainer implementation.
"""

from typing import Dict, Any
from sb3_contrib import RecurrentPPO

from arena.training.base import BaseTrainer
from arena.training.registry import AlgorithmRegistry
from arena.core.config import PPO_LSTM_DEFAULT, PPO_LSTM_GPU_DEFAULT
from arena.training.utils import resolve_activation_fn

@AlgorithmRegistry.register("ppo_lstm")
class PPOLSTMTrainer(BaseTrainer):
    """Recurrent PPO trainer using LSTM."""
    
    algorithm_name = "ppo_lstm"
    algorithm_class = RecurrentPPO
    policy_type = "MlpLstmPolicy"
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get hyperparameters based on device."""
        hparams = PPO_LSTM_GPU_DEFAULT if self._is_gpu() else PPO_LSTM_DEFAULT
        
        # Ensure batch size is valid for the rollout size
        batch_size = self._pick_valid_batch_size(
            hparams.batch_size, 
            hparams.n_steps
        )
        
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
        """Get net architecture, LSTM params and activation."""
        return {
            "net_arch": self.config.ppo_lstm_net_arch,
            "lstm_hidden_size": self.config.ppo_lstm_hidden_size,
            "n_lstm_layers": self.config.ppo_lstm_n_layers,
            "ortho_init": True,
            "activation_fn": resolve_activation_fn(self.config.ppo_activation),
        }
