"""
PPO LSTM (Recurrent) Trainer implementation.
"""

from typing import Dict, Any
from sb3_contrib import RecurrentPPO

from arena.training.base import BaseTrainer
from arena.training.registry import AlgorithmRegistry
from arena.core.config import PPO_LSTM_DEFAULT, PPO_LSTM_GPU_DEFAULT
from arena.training.utils import resolve_activation_fn
from arena.training.schedules import get_lr_schedule

@AlgorithmRegistry.register("ppo_lstm")
class PPOLSTMTrainer(BaseTrainer):
    """
    Recurrent PPO trainer using LSTM.
    
    Uses RecurrentPPO from sb3_contrib which maintains LSTM hidden states
    across timesteps within an episode. This is beneficial for partially
    observable environments where the agent needs memory.
    
    Key differences from standard PPO:
    - Uses MlpLstmPolicy instead of MlpPolicy
    - Shorter rollouts (n_steps=512 vs 2048) due to memory requirements
    - Smaller batch sizes (32/64 vs 64/256) for efficiency
    - LSTM states are automatically managed during training
    """
    
    algorithm_name = "ppo_lstm"
    algorithm_class = RecurrentPPO
    policy_type = "MlpLstmPolicy"
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """
        Get hyperparameters based on device.
        
        RecurrentPPO uses the same hyperparameters as PPO, but with
        adjusted defaults (shorter rollouts, smaller batches) to account
        for the additional memory requirements of LSTM.
        """
        hparams = PPO_LSTM_GPU_DEFAULT if self._is_gpu() else PPO_LSTM_DEFAULT
        
        # Ensure batch size is valid for the rollout size
        # RecurrentPPO requires batch_size to divide (n_steps * num_envs)
        batch_size = self._pick_valid_batch_size(
            hparams.batch_size, 
            hparams.n_steps
        )
        
        # Build learning rate schedule
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
        """
        Get net architecture, LSTM params and activation.
        
        Returns policy configuration including:
        - net_arch: MLP layers before LSTM
        - lstm_hidden_size: Size of LSTM hidden state
        - n_lstm_layers: Number of stacked LSTM layers
        - ortho_init: Orthogonal initialization for stability
        - activation_fn: Activation function for MLP layers
        """
        return {
            "net_arch": self.config.ppo_lstm_net_arch,
            "lstm_hidden_size": self.config.ppo_lstm_hidden_size,
            "n_lstm_layers": self.config.ppo_lstm_n_layers,
            "ortho_init": True,
            "activation_fn": resolve_activation_fn(self.config.ppo_activation),
        }
