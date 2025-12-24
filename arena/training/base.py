"""
Base Trainer class for Deep RL Arena.
"""

import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, Optional, Type, List, Union

import torch
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

from arena.core.config import TrainerConfig
from arena.core import config as arena_config
from arena.core.device import DeviceManager
from arena.core.environment import ArenaEnv
from arena.core.curriculum import CurriculumManager, CurriculumConfig
from arena.training.callbacks import (
    ArenaCallback, PerformanceCallback, HParamCallback, 
    CurriculumCallback, LearningRateCallback, CheckpointWithStateCallback
)
from arena.training.training_state import find_training_state, save_training_state, get_training_state_path, TrainingState

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
        
        # Set up unified directory structure
        self.run_dir = os.path.join(
            config.runs_dir,
            config.algo,
            f"style{config.style}",
            self.run_name
        )
        self.checkpoint_dir = os.path.join(self.run_dir, "checkpoints")
        self.final_dir = os.path.join(self.run_dir, "final")
        self.log_dir = os.path.join(self.run_dir, "logs")
        
        # Runtime components
        self.env: Optional[Union[DummyVecEnv, SubprocVecEnv, VecNormalize]] = None
        self.model: Optional[BaseAlgorithm] = None
        self.callbacks: List[Any] = []
        self.curriculum_manager: Optional[CurriculumManager] = None
        
        # Initialize curriculum if enabled
        if arena_config.CURRICULUM_ENABLED:
            self.curriculum_manager = CurriculumManager(
                CurriculumConfig(enabled=True)
            )
        
        # Initialize unified directory structure
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.final_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

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
        curriculum_manager = self.curriculum_manager
        control_style = self.config.style
        def _init():
            env = ArenaEnv(
                control_style=control_style, 
                render_mode=render_mode,
                curriculum_manager=curriculum_manager
            )
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
        
        # Wrap with VecNormalize for observation and reward normalization
        # This is critical for stable PPO training with variable reward scales
        self.env = VecNormalize(
            self.env,
            norm_obs=True,           # Normalize observations (running mean/std)
            norm_reward=True,        # Normalize rewards (critical for value function!)
            clip_obs=10.0,           # Clip extreme observations
            clip_reward=10.0,        # Clip extreme rewards
            gamma=0.99,              # Discount factor for reward normalization
        )
        print("VecNormalize wrapper enabled (obs + reward normalization)")

    def setup_callbacks(self, hparams: Dict[str, Any]) -> None:
        """Setup training callbacks."""
        # Use custom checkpoint callback that also saves training state
        checkpoint = CheckpointWithStateCallback(
            curriculum_manager=self.curriculum_manager,
            save_freq=self.config.checkpoint_freq,
            save_path=self.checkpoint_dir,
            name_prefix=self.run_name,
            save_replay_buffer=self.config.save_replay_buffer,
            save_vecnormalize=self.config.save_vecnormalize,
            verbose=1,
        )
        
        # Prepare flat hparams for TensorBoard
        tb_hparams = {
            "algo": self.config.algo,
            "style": self.config.style,
            **hparams
        }
        
        self.callbacks = CallbackList([
            checkpoint, 
            ArenaCallback(verbose=0),
            PerformanceCallback(verbose=0),
            HParamCallback(tb_hparams),
            CurriculumCallback(self.curriculum_manager, verbose=1),
            LearningRateCallback(verbose=0),
        ])

    def _find_vecnormalize_stats(self, model_path: str) -> Optional[str]:
        """Find VecNormalize stats file matching the model."""
        import glob
        
        model_dir = os.path.dirname(model_path)
        model_name = os.path.basename(model_path).replace('.zip', '')
        
        # Extract run prefix (e.g., ppo_style2_20251222_154509)
        parts = model_name.split('_')
        if len(parts) >= 4:
            # Try to find matching vecnormalize file in same directory (new structure)
            run_prefix = '_'.join(parts[:4])  # algo_styleX_YYYYMMDD_HHMMSS
            pattern = os.path.join(model_dir, f"{run_prefix}_*_vecnormalize*.pkl")
            matches = sorted(glob.glob(pattern), reverse=True)  # Latest first
            if matches:
                return matches[0]
            
            # Try in parent directory (for new unified structure where checkpoints/ and final/ are separate)
            parent_dir = os.path.dirname(model_dir)
            for subdir in ['checkpoints', 'final', '.']:
                search_dir = os.path.join(parent_dir, subdir) if subdir != '.' else parent_dir
                pattern = os.path.join(search_dir, f"{run_prefix}_*_vecnormalize*.pkl")
                matches = sorted(glob.glob(pattern), reverse=True)
                if matches:
                    return matches[0]
        
        # Fallback: find any vecnormalize file in the same directory
        pattern = os.path.join(model_dir, "*vecnormalize*.pkl")
        matches = sorted(glob.glob(pattern), reverse=True)
        return matches[0] if matches else None
    
    def _load_pretrained_model(
        self,
        hyperparams: Dict[str, Any],
        policy_kwargs: Dict[str, Any],
    ) -> BaseAlgorithm:
        """Load a pretrained SB3 model and validate spaces for safe resuming."""
        model_path = self.config.pretrained_model_path
        if not model_path:
            raise ValueError("pretrained_model_path is not set.")

        print(f"\nLoading pretrained model from: {model_path}")
        
        # Load VecNormalize stats BEFORE loading model (to wrap env correctly)
        if self.config.load_vecnormalize:
            vecnorm_path = self._find_vecnormalize_stats(model_path)
            if vecnorm_path and os.path.exists(vecnorm_path):
                print(f"Loading VecNormalize stats: {vecnorm_path}")
                # VecNormalize.load expects the unwrapped vec_env
                # Get the underlying vec_env from current VecNormalize wrapper
                unwrapped_env = self.env.venv if isinstance(self.env, VecNormalize) else self.env
                self.env = VecNormalize.load(vecnorm_path, unwrapped_env)
                print("VecNormalize stats loaded successfully")
            else:
                print("No VecNormalize stats found, using fresh normalization")

        # Note: algorithm_class.load() restores weights and algorithm state; we attach the new env.
        # We also set tensorboard_log so continuing training logs to the current run directory.
        model = self.algorithm_class.load(
            model_path,
            env=self.env,
            device=self.device,
            tensorboard_log=self.log_dir,
        )

        # Validate action/observation spaces match the new env (required for resume).
        if model.observation_space != self.env.observation_space:
            raise ValueError(
                "Loaded model observation_space does not match current environment. "
                f"model={model.observation_space}, env={self.env.observation_space}. "
                "Make sure you resume with the same control style and observation setup."
            )
        if model.action_space != self.env.action_space:
            raise ValueError(
                "Loaded model action_space does not match current environment. "
                f"model={model.action_space}, env={self.env.action_space}. "
                "Make sure you resume with the same algo and control style."
            )

        # If user changed hyperparams via CLI/config, apply the most important ones for continuing.
        # (SB3 load() keeps saved hyperparams; we only override what the current config controls.)
        if "learning_rate" in hyperparams:
            model.learning_rate = hyperparams["learning_rate"]
        if "policy_kwargs" in getattr(model, "__dict__", {}):
            model.policy_kwargs = policy_kwargs
        
        # Load replay buffer for off-policy algorithms (DQN)
        if self.config.load_replay_buffer and hasattr(model, 'replay_buffer'):
            replay_buffer_path = model_path.replace(".zip", "_replay_buffer.pkl")
            if os.path.exists(replay_buffer_path):
                print(f"Loading replay buffer: {replay_buffer_path}")
                model.load_replay_buffer(replay_buffer_path)
                print("Replay buffer loaded successfully")
        
        # Load curriculum state
        if self.config.load_curriculum and self.curriculum_manager:
            training_state = find_training_state(model_path)
            if training_state:
                curriculum_data = {
                    "current_stage_index": training_state.curriculum_stage_index,
                    "metrics": training_state.curriculum_metrics
                }
                self.curriculum_manager.load_from_dict(curriculum_data)
                print(f"Restored curriculum state: stage {self.curriculum_manager.current_stage_index} "
                      f"({self.curriculum_manager.current_stage.name})")
            else:
                print("No training state found, curriculum starts from beginning")

        return model

    def train(self) -> BaseAlgorithm:
        """Main training loop."""
        DeviceManager.setup_optimizations(self.device)
        
        self.create_environment()
        
        hyperparams = self.get_hyperparameters()
        policy_kwargs = self.get_policy_kwargs()
        
        if self.config.pretrained_model_path:
            # Resume/transfer learning: load existing model and attach the new env.
            self.model = self._load_pretrained_model(hyperparams, policy_kwargs)
            print(
                f"Resuming training on {self.device} "
                f"(reset_num_timesteps={self.config.reset_num_timesteps})..."
            )
        else:
            print(f"\nInitializing {self.algorithm_name.upper()} model on {self.device}...")
            self.model = self.algorithm_class(
                self.policy_type,
                self.env,
                policy_kwargs=policy_kwargs,
                tensorboard_log=self.log_dir,
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
            log_interval=log_interval,
            reset_num_timesteps=self.config.reset_num_timesteps,
        )
        
        self.save_final_model()
        self.env.close()
        return self.model

    def save_final_model(self) -> str:
        """Save the final trained model with complete training state."""
        save_path = os.path.join(self.final_dir, f"{self.run_name}_final")
        self.model.save(save_path)
        print(f"Final model saved to: {save_path}.zip")
        print(f"All run files located in: {self.run_dir}")
        
        # Save training state for transfer learning
        self._save_training_state(save_path)
        
        return save_path
    
    def _save_training_state(self, model_path: str):
        """Save training state for resuming/transfer learning."""
        if not self.curriculum_manager:
            return
        
        # Get curriculum state
        curriculum_dict = self.curriculum_manager.to_dict()
        
        # Create training state
        state = TrainingState(
            model_path=model_path + ".zip",
            algo=self.config.algo,
            style=self.config.style,
            total_timesteps_completed=self.model.num_timesteps,
            total_episodes=0,  # SB3 doesn't track this directly
            curriculum_stage_index=curriculum_dict["current_stage_index"],
            curriculum_metrics=curriculum_dict["metrics"],
        )
        
        # Save to JSON
        state_path = get_training_state_path(model_path + ".zip")
        save_training_state(state_path, state)

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
