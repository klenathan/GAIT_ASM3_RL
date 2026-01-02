"""
Base Trainer class for PyTorch-based Deep RL Arena.
Replaces SB3 dependencies with native PyTorch training loop.
"""

import os
import json
import torch
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, Optional, List
from torch.utils.tensorboard import SummaryWriter

try:
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn, MofNCompleteColumn
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    Progress = None
    Console = None

from arena.core.config import TrainerConfig
from arena.core import config as arena_config
from arena.core.device import DeviceManager
from arena.core.environment import ArenaEnv
from arena.core.vec_env import TorchVecEnv
from arena.core.curriculum import CurriculumManager, CurriculumConfig
from arena.training.training_state import TrainingState, save_training_state, get_training_state_path


class BaseTorchTrainer(ABC):
    """
    Abstract base class for PyTorch-based RL trainers.
    Handles environment creation, device setup, and the training loop.
    """
    
    algorithm_name: str = ""
    
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
        self.env: Optional[TorchVecEnv] = None
        self.model = None
        self.writer: Optional[SummaryWriter] = None
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
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=os.path.join(self.log_dir, f"{self.run_name}_1"))

    @abstractmethod
    def create_model(self) -> Any:
        """Create and return the RL model."""
        pass

    @abstractmethod
    def train_step(self) -> Dict[str, float]:
        """Perform one training step and return metrics."""
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
            return env
        
        return _init

    def create_environment(self) -> None:
        """Create GPU-vectorized training environment."""
        num_envs = self.config.num_envs
        if num_envs is None:
            num_envs = DeviceManager.get_recommended_num_envs(self.device)
        
        print(f"Creating {num_envs} vectorized environments on {self.device}...")
        
        # Create GPU-vectorized environment
        self.env = TorchVecEnv(
            env_fns=[self._make_env_fn() for _ in range(num_envs)],
            device=self.device,
            normalize_obs=True,
            normalize_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=0.99,
        )
        
        print(f"GPU-vectorized environment created with {self.env.num_envs} environments")

    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics to TensorBoard."""
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)

    def log_hparams(self, hparams: Dict[str, Any]):
        """Log hyperparameters to TensorBoard."""
        # Convert to string representation for TensorBoard
        hparam_dict = {}
        for key, value in hparams.items():
            if isinstance(value, (int, float, str, bool)):
                hparam_dict[key] = value
            else:
                hparam_dict[key] = str(value)
        
        # Write as text
        hparam_text = "| Parameter | Value |\n|-----------|-------|\n"
        for key, value in hparam_dict.items():
            hparam_text += f"| {key} | {value} |\n"
        
        self.writer.add_text("hyperparameters", hparam_text, 0)

    def save_checkpoint(self, step: int):
        """Save training checkpoint."""
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"{self.run_name}_{step}_steps.pt"
        )
        
        checkpoint = {
            'model': self.model,
            'step': step,
            'config': self.config.__dict__,
        }
        
        # Save normalization stats
        if self.env is not None:
            checkpoint['normalization_stats'] = self.env.get_normalization_stats()
        
        # Save curriculum state
        if self.curriculum_manager:
            checkpoint['curriculum'] = self.curriculum_manager.to_dict()
        
        self.model.save(checkpoint_path)
        
        # Save additional metadata
        metadata_path = checkpoint_path.replace('.pt', '_metadata.json')
        metadata = {
            'step': step,
            'algorithm': self.algorithm_name,
            'style': self.config.style,
            'timestamp': datetime.now().isoformat(),
        }
        
        if self.env is not None:
            norm_stats_path = checkpoint_path.replace('.pt', '_normalization.json')
            with open(norm_stats_path, 'w') as f:
                json.dump(self.env.get_normalization_stats(), f, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"Loading checkpoint: {checkpoint_path}")
        
        # Load normalization stats if available
        norm_stats_path = checkpoint_path.replace('.pt', '_normalization.json')
        if os.path.exists(norm_stats_path):
            with open(norm_stats_path, 'r') as f:
                norm_stats = json.load(f)
                # Convert lists back to numpy arrays
                import numpy as np
                for key in norm_stats:
                    if isinstance(norm_stats[key], list):
                        norm_stats[key] = np.array(norm_stats[key])
                
                if self.env is not None:
                    self.env.set_normalization_stats(norm_stats)
                    print("✓ Normalization stats loaded")
        
        # Load model
        self.model.load(checkpoint_path)
        print("✓ Model loaded successfully")

    def train(self) -> Any:
        """Main training loop."""
        DeviceManager.setup_optimizations(self.device)
        
        # Create environment and model
        self.create_environment()
        self.model = self.create_model()
        
        # Log hyperparameters
        hparams = {
            'algorithm': self.algorithm_name,
            'style': self.config.style,
            'device': self.device,
            'num_envs': self.env.num_envs,
            'total_timesteps': self.config.total_timesteps,
        }
        self.log_hparams(hparams)
        
        print(f"\nStarting training: {self.run_name}")
        print(f"Device: {self.device}")
        print(f"Environments: {self.env.num_envs}")
        print(f"Total timesteps: {self.config.total_timesteps}")
        
        # Training loop
        total_steps = 0
        episode_rewards = []
        episode_lengths = []
        
        # Create rich progress bar if enabled
        progress = None
        task_id = None
        console = None
        
        if self.config.progress_bar and HAS_RICH:
            console = Console()
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(complete_style="green", finished_style="bold green"),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("•"),
                MofNCompleteColumn(),
                TextColumn("•"),
                TextColumn("[cyan]{task.speed:,.0f} steps/s[/cyan]"),
                TextColumn("•"),
                TimeElapsedColumn(),
                TextColumn("<"),
                TimeRemainingColumn(),
                console=console,
            )
            progress.start()
            task_id = progress.add_task("🚀 Training", total=self.config.total_timesteps)
        
        try:
            while total_steps < self.config.total_timesteps:
                # Perform training step
                metrics = self.train_step()
                
                # Update total steps
                prev_steps = total_steps
                total_steps = self.model.num_timesteps
                steps_delta = total_steps - prev_steps
                
                # Update progress bar
                if progress is not None and task_id is not None:
                    # Build description with key metrics
                    desc_parts = ["🚀 Training"]
                    
                    if 'rollout/ep_rew_mean' in metrics:
                        desc_parts.append(f"📊 [yellow]{metrics['rollout/ep_rew_mean']:.1f}[/yellow]")
                    
                    if 'arena/win_rate' in metrics:
                        win_rate = metrics['arena/win_rate']
                        if win_rate > 0.5:
                            emoji, color = '🏆', 'green'
                        elif win_rate > 0.2:
                            emoji, color = '🎯', 'yellow'
                        else:
                            emoji, color = '📈', 'red'
                        desc_parts.append(f"{emoji} [{color}]{win_rate:.1%}[/{color}]")
                    
                    if 'train/policy_loss' in metrics:
                        desc_parts.append(f"💥 [magenta]{metrics['train/policy_loss']:.3f}[/magenta]")
                    
                    if 'arena/enemies_destroyed' in metrics:
                        desc_parts.append(f"⚔️  [red]{metrics['arena/enemies_destroyed']:.1f}[/red]")
                    
                    progress.update(task_id, advance=steps_delta, description=" ".join(desc_parts))
                
                # Checkpoint saving
                if self.config.checkpoint_freq > 0 and total_steps % self.config.checkpoint_freq == 0:
                    if console is not None:
                        console.print(f"💾 [bold green]Checkpoint saved at {total_steps:,} steps[/bold green]")
                    self.save_checkpoint(total_steps)
                
                # Curriculum progression
                # Note: Episodes are recorded via the callback system during rollout collection
                # Here we just check if we should advance
                if self.curriculum_manager and self.curriculum_manager.check_advancement():
                    msg = f"🎓 Curriculum advanced to stage {self.curriculum_manager.current_stage_index}: {self.curriculum_manager.current_stage.name}"
                    if console is not None:
                        console.print(f"[bold yellow]{msg}[/bold yellow]")
                    else:
                        print(f"\n{msg}")
                    
                    self.writer.add_scalar(
                        'curriculum/stage',
                        self.curriculum_manager.current_stage_index,
                        total_steps
                    )
        finally:
            # Stop progress bar
            if progress is not None:
                progress.stop()
                self.save_checkpoint(total_steps)
                
                # Curriculum progression
                # Note: Episodes are recorded via the callback system during rollout collection
                # Here we just check if we should advance
                if self.curriculum_manager and self.curriculum_manager.check_advancement():
                    msg = f"🎓 \033[93mCurriculum advanced to stage {self.curriculum_manager.current_stage_index}: {self.curriculum_manager.current_stage.name}\033[0m"
                    if pbar is not None:
                        pbar.write(msg)
                    else:
                        print(f"\n{msg}")
                    
                    self.writer.add_scalar(
                        'curriculum/stage',
                        self.curriculum_manager.current_stage_index,
                        total_steps
                    )
        # finally:
        #     # Close progress bar
        #     if pbar is not None:
        #         pbar.close()
        
        # Save final model
        self.save_final_model()
        
        # Cleanup
        if self.env:
            self.env.close()
        if self.writer:
            self.writer.close()
        
        return self.model

    def save_final_model(self) -> str:
        """Save the final trained model with complete training state."""
        save_path = os.path.join(self.final_dir, f"{self.run_name}_final.pt")
        self.model.save(save_path)
        print(f"Final model saved to: {save_path}")
        
        # Save normalization stats
        if self.env is not None:
            norm_stats_path = os.path.join(self.final_dir, f"{self.run_name}_normalization_final.json")
            with open(norm_stats_path, 'w') as f:
                json.dump(
                    self.env.get_normalization_stats(),
                    f,
                    indent=2,
                    default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x)
                )
            print(f"Normalization stats saved to: {norm_stats_path}")
        
        # Save training state
        if self.curriculum_manager:
            curriculum_dict = self.curriculum_manager.to_dict()
            state = TrainingState(
                model_path=save_path,
                algo=self.config.algo,
                style=self.config.style,
                total_timesteps_completed=self.model.num_timesteps,
                total_episodes=0,
                curriculum_stage_index=curriculum_dict["current_stage_index"],
                curriculum_metrics=curriculum_dict["metrics"],
            )
            
            state_path = get_training_state_path(save_path)
            save_training_state(state_path, state)
            print(f"Training state saved to: {state_path}")
        
        print(f"All run files located in: {self.run_dir}")
        
        return save_path
