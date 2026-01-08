"""
PPO Dict Trainer implementation with dictionary observation space.
"""

from typing import Dict, Any
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import CombinedExtractor

from arena.training.base import BaseTrainer
from arena.training.registry import AlgorithmRegistry
from arena.core.config import PPO_DEFAULT, PPO_GPU_DEFAULT
from arena.training.utils import resolve_activation_fn
from arena.training.schedules import get_lr_schedule
from arena.core.environment_dict import ArenaDictEnv
from arena.core.curriculum import CurriculumManager
from stable_baselines3.common.monitor import Monitor


@AlgorithmRegistry.register("ppo_dict")
class PPODictTrainer(BaseTrainer):
    """
    Proximal Policy Optimization trainer with dictionary observation space.
    
    Uses MultiInputPolicy for structured observations with semantic grouping:
    - player_state: Agent's internal state 
    - combat_targets: Enemy and spawner locations
    - mission_progress: Phase and objective tracking
    - spatial_awareness: Environmental boundaries
    - enemy_count: Tactical threat level
    """
    
    algorithm_name = "ppo_dict"
    algorithm_class = PPO
    policy_type = "MultiInputPolicy"
    
    def _make_env_fn(self):
        """Environment factory using dict observation env."""
        render_mode = "human" if self.config.render else None
        curriculum_manager = self.curriculum_manager
        control_style = self.config.style
        
        def _init():
            env = ArenaDictEnv(
                control_style=control_style, 
                render_mode=render_mode,
                curriculum_manager=curriculum_manager
            )
            return Monitor(env)
        return _init
    
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
        Get policy network configuration for dict observations.
        
        Uses CombinedExtractor to process each observation group separately
        before combining features for the policy and value networks.
        """
        # Get activation function
        activation_fn = resolve_activation_fn(self.config.ppo_activation)
        
        # Configure feature extractor for dict observations
        # CombinedExtractor creates a separate MLP for each observation group
        # Default architecture: 64 units per observation component
        features_extractor_kwargs = {
            "cnn_output_dim": 256,  # Not used for Box spaces, but good to set
        }
        
        return {
            "net_arch": self.config.ppo_net_arch,
            "activation_fn": activation_fn,
            "ortho_init": True,
            # SB3's CombinedExtractor will automatically create feature extractors
            # for each dict key. Each Box observation gets a simple linear layer.
            "features_extractor_class": CombinedExtractor,
            "features_extractor_kwargs": features_extractor_kwargs,
        }
