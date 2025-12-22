"""
Base Trainer class for Deep RL Arena.
"""

import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, Optional, Type, List

import torch
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

from arena.core.config import TrainerConfig
from arena.core.device import DeviceManager
from arena.core.environment import ArenaEnv
from arena.training.callbacks import ArenaCallback, PerformanceCallback, HParamCallback

class BaseTrainer(ABC):
    """
    Abstract base class for all RL trainers.
    Handles environment creation, device setup, and the training loop.
    """
    
    algorithm_name: str = ""
    algorithm_class: Type[BaseAlgorithm] = None
    policy_type: str = "MlpPolicy"
    
    def __init__(self, config: TrainerConfig):
        self.config = config
        self.device = DeviceManager.get_device(config.device)
        self.run_name = f"{config.algo}_style{config.style}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Runtime components
        self.env: Optional[Union[DummyVecEnv, SubprocVecEnv]] = None
        self.model: Optional[BaseAlgorithm] = None
        self.callbacks: List[Any] = []
        
        # Initialize directories
        os.makedirs(self.config.model_save_dir, exist_ok=True)
        os.makedirs(self.config.tensorboard_log_dir, exist_ok=True)

    @abstractmethod
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Return algorithm-specific hyperparameters."""
        pass

    @abstractmethod
    def get_policy_kwargs(self) -> Dict[str, Any]:
        """Return policy network configuration."""
        pass

    def _make_env_fn(self):
        """Environment factory."""
        render_mode = "human" if self.config.render else None
        def _init():
            env = ArenaEnv(control_style=self.config.style, render_mode=render_mode)
            return Monitor(env)
        return _init

    def create_environment(self) -> None:
        """Create vectorized training environment."""
        num_envs = self.config.num_envs
        if num_envs is None:
            num_envs = DeviceManager.get_recommended_num_envs(self.device)
            
        # Workaround for MPS multiprocessing
        if self.device == "mps" and num_envs > 1:
            print(f"Applying MPS multiprocessing workaround: using CPU for env workers.")
            
        if num_envs > 1 and not self.config.render:
            try:
                print(f"Creating {num_envs} parallel environments...")
                self.env = SubprocVecEnv([self._make_env_fn() for _ in range(num_envs)])
                DeviceManager.limit_threads_for_vecenv(num_envs, self.device)
            except Exception as e:
                print(f"Failed to create SubprocVecEnv: {e}. Falling back to DummyVecEnv.")
                self.env = DummyVecEnv([self._make_env_fn() for _ in range(num_envs)])
        else:
            self.env = DummyVecEnv([self._make_env_fn()])

    def setup_callbacks(self, hparams: Dict[str, Any]) -> None:
        """Setup training callbacks."""
        checkpoint_dir = os.path.join(
            self.config.model_save_dir, 
            self.config.algo, 
            f"style{self.config.style}"
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = CheckpointCallback(
            save_freq=self.config.checkpoint_freq,
            save_path=checkpoint_dir,
            name_prefix=self.run_name,
            save_replay_buffer=self.config.save_replay_buffer,
            save_vecnormalize=self.config.save_vecnormalize,
        )
        
        # Prepare flat hparams for TensorBoard
        tb_hparams = {
            "algo": self.config.algo,
            "style": self.config.style,
            **hparams
        }
        
        self.callbacks = CallbackList([
            # checkpoint,  # Disabled checkpoint logging
            ArenaCallback(verbose=0),
            PerformanceCallback(verbose=0),
            HParamCallback(tb_hparams)
        ])

    def train(self) -> BaseAlgorithm:
        """Main training loop."""
        DeviceManager.setup_optimizations(self.device)
        
        self.create_environment()
        
        hyperparams = self.get_hyperparameters()
        policy_kwargs = self.get_policy_kwargs()
        
        print(f"\nInitializing {self.algorithm_name.upper()} model on {self.device}...")
        
        self.model = self.algorithm_class(
            self.policy_type,
            self.env,
            policy_kwargs=policy_kwargs,
            tensorboard_log=self.config.tensorboard_log_dir,
            device=self.device,
            **hyperparams
        )
        
        self.setup_callbacks(hyperparams)
        
        # Calculate logging interval to be roughly consistent across algos
        # We want to log roughly every 2000-5000 steps.
        # SB3 logs every (n_steps * num_envs * log_interval) steps.
        rollout_size = self.env.num_envs * hyperparams.get("n_steps", 1)
        # Default target of ~1000 steps per log point
        target_log_steps = 1000
        log_interval = max(1, target_log_steps // rollout_size)
        
        print(f"Starting training: {self.run_name} (log_interval={log_interval})")
        self.model.learn(
            total_timesteps=self.config.total_timesteps,
            callback=self.callbacks,
            tb_log_name=self.run_name,
            progress_bar=self.config.progress_bar,
            log_interval=log_interval
        )
        
        self.save_final_model()
        self.env.close()
        return self.model

    def save_final_model(self) -> str:
        """Save the final trained model."""
        save_path = os.path.join(
            self.config.model_save_dir,
            self.config.algo,
            f"style{self.config.style}",
            f"{self.run_name}_final"
        )
        self.model.save(save_path)
        print(f"Final model saved to: {save_path}.zip")
        return save_path

    def _is_gpu(self) -> bool:
        return self.device in ["cuda", "mps"]

    def _pick_valid_batch_size(self, preferred: int, n_steps: int) -> int:
        """Ensure batch_size divides (n_steps * num_envs)."""
        num_envs = self.env.num_envs
        rollout_size = n_steps * num_envs
        
        if rollout_size % preferred == 0:
            return preferred
            
        # Try some common GPU-friendly powers of 2
        for bs in (512, 256, 128, 64, 32):
            if bs <= rollout_size and rollout_size % bs == 0:
                return bs
                
        # Fallback to finding any divisor
        for bs in range(min(preferred, rollout_size), 0, -1):
            if rollout_size % bs == 0:
                return bs
        return 1
