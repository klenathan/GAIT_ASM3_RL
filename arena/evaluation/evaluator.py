"""
Model evaluation logic for Deep RL Arena - PufferLib/PyTorch version.
"""

import torch
import numpy as np
import pygame
import json
import os
from pathlib import Path
import warnings
from typing import Optional, Dict, Any

from arena.training.policies import MLPPolicy, LSTMPolicy, CNNPolicy
from arena.ui.renderer import ArenaRenderer
from arena.game.human_controller import HumanController
from arena.core.environment_dict import ArenaDictEnv
from arena.core.environment_cnn import ArenaCNNEnv
from arena.core.environment import ArenaEnv
from arena.core import config
from arena.core.device import DeviceManager

# Suppress pkg_resources deprecation warning from pygame
warnings.filterwarnings(
    "ignore", message=".*pkg_resources is deprecated.*", category=UserWarning
)


class Evaluator:
    """Handles PyTorch model evaluation with visualization."""

    def __init__(self, requested_device: str = "auto"):
        self.device = DeviceManager.get_device(requested_device)
        self.renderer = None
        self.env = None
        self.model = None
        self.policy_type = None
        self.env_type = None
        self.control_style = 1
        self.obs_normalizer = None
        self.config_data = None
        self.lstm_state = None

    def setup_ui(self):
        """Initialize renderer."""
        if self.renderer is None:
            self.renderer = ArenaRenderer()

    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """
        Load a trained PyTorch model from checkpoint.
        
        Args:
            checkpoint_path: Path to .pt checkpoint file
        
        Returns:
            True if successful, False otherwise
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            print(f"Checkpoint not found: {checkpoint_path}")
            return False
        
        # Load checkpoint
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            print(f"Loading checkpoint: {checkpoint_path}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            return False
        
        # Extract configuration
        if 'config' in checkpoint:
            self.config_data = checkpoint['config']
            self.policy_type = self.config_data.get('policy_type', 'mlp')
            self.env_type = self.config_data.get('env_type', 'standard')
            self.control_style = self.config_data.get('style', 1)
            print(f"Config: policy={self.policy_type}, env={self.env_type}, style={self.control_style}")
        else:
            # Try to infer from checkpoint path
            print("No config in checkpoint, inferring from path...")
            if 'lstm' in str(checkpoint_path).lower():
                self.policy_type = 'lstm'
            elif 'cnn' in str(checkpoint_path).lower():
                self.policy_type = 'cnn'
            else:
                self.policy_type = 'mlp'
            
            if 'dict' in str(checkpoint_path).lower():
                self.env_type = 'dict'
            elif 'cnn' in str(checkpoint_path).lower():
                self.env_type = 'cnn'
            else:
                self.env_type = 'standard'
            
            # Extract style from path
            import re
            match = re.search(r'style(\d+)', str(checkpoint_path))
            if match:
                self.control_style = int(match.group(1))
        
        # Create environment
        self.env = self._create_env()
        
        # Create policy
        obs_space = self.env.observation_space
        action_space = self.env.action_space
        
        try:
            if self.policy_type == 'mlp':
                self.model = MLPPolicy(obs_space, action_space)
            elif self.policy_type == 'lstm':
                self.model = LSTMPolicy(obs_space, action_space)
            elif self.policy_type == 'cnn':
                self.model = CNNPolicy(obs_space, action_space)
            else:
                print(f"Unknown policy type: {self.policy_type}")
                return False
            
            # Load weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            print(f"Model loaded successfully: {self.policy_type}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False
        
        # Load normalization stats if available
        if 'obs_normalizer' in checkpoint:
            from arena.training.pufferl_trainer import ObservationNormalizer
            obs_shape = obs_space.shape if hasattr(obs_space, 'shape') else obs_space['image'].shape
            self.obs_normalizer = ObservationNormalizer(obs_shape)
            self.obs_normalizer.load_state_dict(checkpoint['obs_normalizer'])
            print("Observation normalizer loaded")
        
        return True
    
    def _create_env(self):
        """Create evaluation environment based on config."""
        render_mode = "human" if self.renderer else "rgb_array"
        
        if self.env_type == 'standard':
            return ArenaEnv(
                control_style=self.control_style,
                render_mode=render_mode,
            )
        elif self.env_type == 'dict':
            return ArenaDictEnv(
                control_style=self.control_style,
                render_mode=render_mode,
            )
        elif self.env_type == 'cnn':
            return ArenaCNNEnv(
                control_style=self.control_style,
                render_mode=render_mode,
            )
        else:
            raise ValueError(f"Unknown env_type: {self.env_type}")
    
    def evaluate_episode(
        self,
        deterministic: bool = True,
        render: bool = True,
        max_steps: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run a single evaluation episode.
        
        Args:
            deterministic: Use deterministic policy (argmax) if True
            render: Render to screen if True
            max_steps: Maximum steps per episode (None = no limit)
        
        Returns:
            Dictionary with episode statistics
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_checkpoint() first.")
        
        if render and self.renderer is None:
            self.setup_ui()
        
        # Reset environment
        obs, info = self.env.reset()
        
        # Initialize LSTM state if needed
        if self.policy_type == 'lstm':
            self.lstm_state = self.model.init_state(1, self.device)
        
        episode_reward = 0
        episode_length = 0
        done = False
        clock = pygame.time.Clock() if render else None
        
        while not done:
            if max_steps and episode_length >= max_steps:
                break
            
            # Process observation
            if self.obs_normalizer:
                obs_norm = self.obs_normalizer.normalize(obs[np.newaxis, ...])
            else:
                obs_norm = obs[np.newaxis, ...]
            
            # Convert to tensor
            if isinstance(obs, dict):
                obs_tensor = {
                    k: torch.from_numpy(v).float().to(self.device)
                    for k, v in obs_norm.items()
                }
            else:
                obs_tensor = torch.from_numpy(obs_norm).float().to(self.device)
            
            # Get action from policy
            with torch.no_grad():
                if self.policy_type == 'lstm':
                    action, _, _, _, self.lstm_state = self.model.get_action_and_value(
                        obs_tensor,
                        self.lstm_state,
                        deterministic=deterministic
                    )
                else:
                    action, _, _, _ = self.model.get_action_and_value(
                        obs_tensor,
                        deterministic=deterministic
                    )
            
            # Step environment
            action_np = action.cpu().numpy()[0]
            obs, reward, terminated, truncated, info = self.env.step(action_np)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            
            # Render
            if render:
                self.env.render()
                clock.tick(config.FPS)
                
                # Handle pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        done = True
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            done = True
        
        return {
            'reward': episode_reward,
            'length': episode_length,
            'win': info.get('win', False),
        }
    
    def evaluate_multiple_episodes(
        self,
        num_episodes: int,
        deterministic: bool = True,
        render: bool = False,
        stochastic: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluate model over multiple episodes.
        
        Args:
            num_episodes: Number of episodes to run
            deterministic: Use deterministic policy if True
            render: Render to screen if True
            stochastic: Override deterministic=False (for backward compat)
        
        Returns:
            Dictionary with aggregated statistics
        """
        if stochastic:
            deterministic = False
        
        rewards = []
        lengths = []
        wins = 0
        
        print(f"Evaluating {num_episodes} episodes...")
        
        for i in range(num_episodes):
            result = self.evaluate_episode(
                deterministic=deterministic,
                render=render and (i == 0),  # Only render first episode
            )
            
            rewards.append(result['reward'])
            lengths.append(result['length'])
            if result['win']:
                wins += 1
            
            if (i + 1) % 10 == 0:
                print(f"Episode {i + 1}/{num_episodes}: "
                      f"Reward={result['reward']:.2f}, "
                      f"Length={result['length']}, "
                      f"Win={result['win']}")
        
        stats = {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'mean_length': np.mean(lengths),
            'win_rate': wins / num_episodes,
            'num_episodes': num_episodes,
        }
        
        print("\n=== Evaluation Results ===")
        print(f"Mean Reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
        print(f"Min/Max Reward: {stats['min_reward']:.2f} / {stats['max_reward']:.2f}")
        print(f"Mean Length: {stats['mean_length']:.1f}")
        print(f"Win Rate: {stats['win_rate']:.2%}")
        
        return stats
    
    def close(self):
        """Clean up resources."""
        if self.env:
            self.env.close()
        if self.renderer:
            pygame.quit()


def find_latest_checkpoint(runs_dir: str = "./runs", style: int = 1) -> Optional[str]:
    """
    Find the latest checkpoint for a given style.
    
    Args:
        runs_dir: Base runs directory
        style: Control style (1 or 2)
    
    Returns:
        Path to latest checkpoint or None
    """
    runs_path = Path(runs_dir)
    pattern = f"ppo/style{style}/*/checkpoints/*.pt"
    
    checkpoints = list(runs_path.glob(pattern))
    if not checkpoints:
        pattern = f"ppo/style{style}/*/final/model.pt"
        checkpoints = list(runs_path.glob(pattern))
    
    if not checkpoints:
        return None
    
    # Sort by modification time
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
    return str(latest)
