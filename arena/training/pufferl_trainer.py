"""
PufferLib-based Trainer for Arena Environment.

This replaces the old SB3-based training infrastructure with PufferLib's
explicit PPO training loop for high-performance RL training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
from datetime import datetime

import pufferlib
import pufferlib.vector
from pufferlib.pufferl import PuffeRL

from arena.core import config
from arena.core.puffer_envs import make_env
from arena.core.curriculum import CurriculumManager
from arena.training.policies import MLPPolicy, LSTMPolicy, CNNPolicy


@dataclass
class PufferTrainConfig:
    """Configuration for PufferLib training."""
    
    # Basic settings
    algo: str = "ppo"  # Currently only PPO supported
    env_type: str = "standard"  # 'standard', 'dict', or 'cnn'
    style: int = 1  # Control style: 1 or 2
    total_timesteps: int = 1_000_000
    device: str = "cuda"  # 'cuda', 'cpu', or 'mps'
    
    # Environment settings
    num_envs: int = 16
    vec_backend: str = "Multiprocessing"  # 'Serial', 'Multiprocessing'
    
    # Training hyperparameters (PPO)
    learning_rate: float = 3e-4
    batch_size: int = 512  # Steps per batch
    minibatch_size: int = 128  # Minibatch size for updates
    num_epochs: int = 4  # Epochs per batch
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE parameter
    clip_eps: float = 0.2  # PPO clip epsilon
    value_coef: float = 0.5  # Value loss coefficient
    entropy_coef: float = 0.01  # Entropy bonus coefficient
    max_grad_norm: float = 0.5  # Gradient clipping
    
    # Policy architecture
    policy_type: str = "mlp"  # 'mlp', 'lstm', or 'cnn'
    hidden_size: int = 256
    num_layers: int = 2
    activation: str = "ReLU"
    
    # Normalization
    normalize_obs: bool = True
    normalize_rewards: bool = True
    clip_obs: float = 10.0
    clip_rewards: float = 10.0
    
    # Curriculum learning
    curriculum_enabled: bool = True
    
    # Logging and checkpoints
    log_interval: int = 10  # Log every N batches
    checkpoint_freq: int = 100_000  # Save checkpoint every N steps
    runs_dir: str = "./runs"
    experiment_name: Optional[str] = None
    
    # Learning rate schedule
    lr_schedule: str = "constant"  # 'constant', 'linear', 'cosine'
    lr_warmup_steps: int = 0
    lr_min: float = 1e-6
    
    # Wandb/Neptune logging
    use_wandb: bool = False
    use_neptune: bool = False
    wandb_project: Optional[str] = None
    neptune_project: Optional[str] = None
    tags: list = None
    
    # Resume training
    load_checkpoint: Optional[str] = None


class ObservationNormalizer:
    """Running mean/std normalization for observations (replaces SB3's VecNormalize)."""
    
    def __init__(self, obs_shape, clip=10.0, epsilon=1e-8):
        self.mean = np.zeros(obs_shape, dtype=np.float32)
        self.var = np.ones(obs_shape, dtype=np.float32)
        self.count = epsilon
        self.clip = clip
        self.epsilon = epsilon
    
    def update(self, observations):
        """Update running statistics."""
        batch_mean = np.mean(observations, axis=0)
        batch_var = np.var(observations, axis=0)
        batch_count = observations.shape[0]
        
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        new_var = M2 / total_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = total_count
    
    def normalize(self, observations):
        """Normalize observations using running statistics."""
        normalized = (observations - self.mean) / np.sqrt(self.var + self.epsilon)
        return np.clip(normalized, -self.clip, self.clip)
    
    def state_dict(self):
        """Get normalizer state for checkpointing."""
        return {
            'mean': self.mean,
            'var': self.var,
            'count': self.count,
        }
    
    def load_state_dict(self, state_dict):
        """Load normalizer state from checkpoint."""
        self.mean = state_dict['mean']
        self.var = state_dict['var']
        self.count = state_dict['count']


class RewardNormalizer:
    """Running normalization for rewards."""
    
    def __init__(self, gamma=0.99, clip=10.0, epsilon=1e-8):
        self.var = 1.0
        self.mean = 0.0
        self.gamma = gamma
        self.returns = None
        self.clip = clip
        self.epsilon = epsilon
    
    def update(self, rewards, dones):
        """Update running statistics from episode returns."""
        if self.returns is None:
            self.returns = np.zeros(len(rewards))
        
        self.returns = self.returns * self.gamma + rewards
        self.returns[dones] = 0
        
        # Update variance
        batch_mean = np.mean(self.returns)
        batch_var = np.var(self.returns)
        
        # Simple exponential moving average
        self.mean = 0.99 * self.mean + 0.01 * batch_mean
        self.var = 0.99 * self.var + 0.01 * batch_var
    
    def normalize(self, rewards):
        """Normalize rewards using running statistics."""
        normalized = rewards / np.sqrt(self.var + self.epsilon)
        return np.clip(normalized, -self.clip, self.clip)
    
    def state_dict(self):
        """Get normalizer state for checkpointing."""
        return {
            'mean': self.mean,
            'var': self.var,
        }
    
    def load_state_dict(self, state_dict):
        """Load normalizer state from checkpoint."""
        self.mean = state_dict['mean']
        self.var = state_dict['var']


class PufferTrainer:
    """Main trainer class for PufferLib-based PPO training."""
    
    def __init__(self, config: PufferTrainConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Create experiment directory
        self.setup_experiment_dir()
        
        # Save config
        self.save_config()
        
        # Create curriculum manager if enabled
        self.curriculum_manager = None
        if config.curriculum_enabled:
            self.curriculum_manager = CurriculumManager(
                window_size=config.CURRICULUM_WINDOW if hasattr(config, 'CURRICULUM_WINDOW') else 100
            )
        
        # Create vectorized environment
        self.vecenv = self.create_vecenv()
        
        # Create policy
        self.policy = self.create_policy()
        self.policy.to(self.device)
        
        # Create optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.learning_rate)
        
        # Create LR scheduler if needed
        self.scheduler = self.create_scheduler()
        
        # Create normalizers
        self.obs_normalizer = None
        self.reward_normalizer = None
        if config.normalize_obs:
            obs_shape = self.vecenv.single_observation_space.shape
            self.obs_normalizer = ObservationNormalizer(obs_shape, clip=config.clip_obs)
        if config.normalize_rewards:
            self.reward_normalizer = RewardNormalizer(
                gamma=config.gamma,
                clip=config.clip_rewards
            )
        
        # Training state
        self.global_step = 0
        self.num_updates = 0
        self.start_time = time.time()
        
        # Metrics storage
        self.episode_rewards = []
        self.episode_lengths = []
        
        # Load checkpoint if resuming
        if config.load_checkpoint:
            self.load_checkpoint(config.load_checkpoint)
    
    def setup_experiment_dir(self):
        """Create experiment directory structure."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.config.experiment_name:
            exp_name = self.config.experiment_name
        else:
            exp_name = f"{self.config.algo}_{self.config.policy_type}_style{self.config.style}_{timestamp}"
        
        self.experiment_dir = Path(self.config.runs_dir) / self.config.algo / f"style{self.config.style}" / exp_name
        self.checkpoint_dir = self.experiment_dir / "checkpoints"
        self.log_dir = self.experiment_dir / "logs"
        
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        
        print(f"Experiment directory: {self.experiment_dir}")
    
    def save_config(self):
        """Save configuration to JSON."""
        config_path = self.experiment_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2)
    
    def create_vecenv(self):
        """Create vectorized PufferLib environment."""
        def env_creator():
            return make_env(
                env_type=self.config.env_type,
                control_style=self.config.style,
                render_mode=None,
                curriculum_manager=self.curriculum_manager,
            )
        
        vecenv = pufferlib.vector.make(
            env_creator=env_creator,
            backend=self.config.vec_backend,
            num_envs=self.config.num_envs,
        )
        
        return vecenv
    
    def create_policy(self):
        """Create policy network based on config."""
        obs_space = self.vecenv.single_observation_space
        action_space = self.vecenv.single_action_space
        
        if self.config.policy_type == "mlp":
            policy = MLPPolicy(
                observation_space=obs_space,
                action_space=action_space,
                hidden_size=self.config.hidden_size,
                num_layers=self.config.num_layers,
                activation=self.config.activation,
            )
        elif self.config.policy_type == "lstm":
            policy = LSTMPolicy(
                observation_space=obs_space,
                action_space=action_space,
                hidden_size=self.config.hidden_size,
                num_layers=self.config.num_layers,
                activation=self.config.activation,
            )
        elif self.config.policy_type == "cnn":
            policy = CNNPolicy(
                observation_space=obs_space,
                action_space=action_space,
                activation=self.config.activation,
            )
        else:
            raise ValueError(f"Unknown policy type: {self.config.policy_type}")
        
        return policy
    
    def create_scheduler(self):
        """Create learning rate scheduler."""
        if self.config.lr_schedule == "constant":
            return None
        elif self.config.lr_schedule == "linear":
            total_steps = self.config.total_timesteps
            return optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda step: max(0.0, 1.0 - step / total_steps)
            )
        elif self.config.lr_schedule == "cosine":
            total_steps = self.config.total_timesteps
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=self.config.lr_min
            )
        else:
            return None
    
    def train(self):
        """Main training loop."""
        print(f"Starting training for {self.config.total_timesteps} timesteps")
        print(f"Device: {self.device}")
        print(f"Num envs: {self.config.num_envs}")
        print(f"Policy: {self.config.policy_type}")
        
        # Initialize LSTM state if needed
        lstm_state = None
        if self.config.policy_type == "lstm":
            lstm_state = self.policy.init_state(self.config.num_envs, self.device)
        
        # Reset environments
        observations = self.vecenv.reset()
        
        # Main training loop
        while self.global_step < self.config.total_timesteps:
            # Collect batch of experiences
            batch = self.collect_batch(observations, lstm_state)
            observations = batch['next_observations']
            
            if self.config.policy_type == "lstm":
                lstm_state = batch['next_lstm_state']
            
            # Train on batch
            metrics = self.train_batch(batch)
            
            # Update curriculum if enabled
            if self.curriculum_manager and len(self.episode_rewards) > 0:
                recent_reward = self.episode_rewards[-1]
                self.curriculum_manager.add_episode_reward(recent_reward)
                self.curriculum_manager.check_advancement()
            
            # Logging
            if self.num_updates % self.config.log_interval == 0:
                self.log_metrics(metrics)
            
            # Checkpointing
            if self.global_step % self.config.checkpoint_freq < self.config.batch_size:
                self.save_checkpoint()
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            self.num_updates += 1
        
        # Final checkpoint
        self.save_checkpoint(final=True)
        self.vecenv.close()
        
        print(f"Training complete! Total time: {time.time() - self.start_time:.2f}s")
    
    def collect_batch(self, initial_observations, initial_lstm_state=None):
        """Collect a batch of experiences from the environment."""
        batch_obs = []
        batch_actions = []
        batch_log_probs = []
        batch_values = []
        batch_rewards = []
        batch_dones = []
        
        observations = initial_observations
        lstm_state = initial_lstm_state
        
        with torch.no_grad():
            for _ in range(self.config.batch_size // self.config.num_envs):
                # Normalize observations
                if self.obs_normalizer:
                    self.obs_normalizer.update(observations)
                    observations_norm = self.obs_normalizer.normalize(observations)
                else:
                    observations_norm = observations
                
                # Convert to tensor
                obs_tensor = torch.from_numpy(observations_norm).float().to(self.device)
                
                # Get action from policy
                if self.config.policy_type == "lstm":
                    actions, log_probs, _, values, lstm_state = self.policy.get_action_and_value(
                        obs_tensor, lstm_state
                    )
                else:
                    actions, log_probs, _, values = self.policy.get_action_and_value(obs_tensor)
                
                # Step environment
                next_obs, rewards, dones, infos = self.vecenv.step(actions.cpu().numpy())
                
                # Normalize rewards
                if self.reward_normalizer:
                    self.reward_normalizer.update(rewards, dones)
                    rewards_norm = self.reward_normalizer.normalize(rewards)
                else:
                    rewards_norm = rewards
                
                # Store transitions
                batch_obs.append(observations_norm)
                batch_actions.append(actions.cpu().numpy())
                batch_log_probs.append(log_probs.cpu().numpy())
                batch_values.append(values.cpu().numpy())
                batch_rewards.append(rewards_norm)
                batch_dones.append(dones)
                
                # Track episode statistics
                for info in infos:
                    if 'episode' in info:
                        self.episode_rewards.append(info['episode']['r'])
                        self.episode_lengths.append(info['episode']['l'])
                
                observations = next_obs
                self.global_step += self.config.num_envs
        
        # Stack batch
        batch = {
            'observations': np.array(batch_obs),
            'actions': np.array(batch_actions),
            'log_probs': np.array(batch_log_probs),
            'values': np.array(batch_values),
            'rewards': np.array(batch_rewards),
            'dones': np.array(batch_dones),
            'next_observations': observations,
        }
        
        if self.config.policy_type == "lstm":
            batch['next_lstm_state'] = lstm_state
        
        return batch
    
    def compute_gae(self, rewards, values, dones, next_values):
        """Compute Generalized Advantage Estimation."""
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.config.gamma * next_value * (1 - dones[t]) - values[t]
            last_gae = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae
        
        returns = advantages + values
        return advantages, returns
    
    def train_batch(self, batch):
        """Train on a collected batch using PPO."""
        # Compute advantages
        with torch.no_grad():
            if self.obs_normalizer:
                next_obs_norm = self.obs_normalizer.normalize(batch['next_observations'])
            else:
                next_obs_norm = batch['next_observations']
            
            next_obs_tensor = torch.from_numpy(next_obs_norm).float().to(self.device)
            next_values = self.policy.get_value(next_obs_tensor).cpu().numpy()
        
        advantages, returns = self.compute_gae(
            batch['rewards'],
            batch['values'],
            batch['dones'],
            next_values
        )
        
        # Flatten batch
        b_obs = torch.from_numpy(batch['observations']).float().to(self.device)
        b_actions = torch.from_numpy(batch['actions']).long().to(self.device)
        b_log_probs = torch.from_numpy(batch['log_probs']).float().to(self.device)
        b_advantages = torch.from_numpy(advantages).float().to(self.device)
        b_returns = torch.from_numpy(returns).float().to(self.device)
        
        # Flatten time dimension
        b_obs = b_obs.reshape(-1, *b_obs.shape[2:])
        b_actions = b_actions.flatten()
        b_log_probs = b_log_probs.flatten()
        b_advantages = b_advantages.flatten()
        b_returns = b_returns.flatten()
        
        # Normalize advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
        
        # PPO update
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        for epoch in range(self.config.num_epochs):
            # Minibatch updates
            indices = np.arange(len(b_obs))
            np.random.shuffle(indices)
            
            for start in range(0, len(b_obs), self.config.minibatch_size):
                end = start + self.config.minibatch_size
                mb_indices = indices[start:end]
                
                # Get new predictions
                if self.config.policy_type == "lstm":
                    # For LSTM, we need to handle sequences properly
                    # This is simplified - proper implementation would maintain hidden states
                    _, new_log_probs, entropy, new_values = self.policy.get_action_and_value(
                        b_obs[mb_indices],
                        actions=b_actions[mb_indices]
                    )
                else:
                    _, new_log_probs, entropy, new_values = self.policy.get_action_and_value(
                        b_obs[mb_indices],
                        actions=b_actions[mb_indices]
                    )
                
                # PPO policy loss
                log_ratio = new_log_probs - b_log_probs[mb_indices]
                ratio = torch.exp(log_ratio)
                
                mb_advantages = b_advantages[mb_indices]
                policy_loss1 = -mb_advantages * ratio
                policy_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - self.config.clip_eps, 1 + self.config.clip_eps
                )
                policy_loss = torch.max(policy_loss1, policy_loss2).mean()
                
                # Value loss
                value_loss = ((new_values - b_returns[mb_indices]) ** 2).mean()
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (
                    policy_loss +
                    self.config.value_coef * value_loss +
                    self.config.entropy_coef * entropy_loss
                )
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                
                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
        
        num_updates = self.config.num_epochs * (len(b_obs) // self.config.minibatch_size)
        
        return {
            'loss': total_loss / num_updates,
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates,
        }
    
    def log_metrics(self, metrics):
        """Log training metrics."""
        elapsed = time.time() - self.start_time
        fps = self.global_step / elapsed if elapsed > 0 else 0
        
        log_str = f"Step: {self.global_step} | FPS: {fps:.0f} | "
        log_str += f"Loss: {metrics['loss']:.4f} | "
        
        if len(self.episode_rewards) > 0:
            mean_reward = np.mean(self.episode_rewards[-100:])
            mean_length = np.mean(self.episode_lengths[-100:])
            log_str += f"Reward: {mean_reward:.2f} | Length: {mean_length:.0f}"
        
        print(log_str)
    
    def save_checkpoint(self, final=False):
        """Save model checkpoint."""
        if final:
            checkpoint_path = self.experiment_dir / "final" / "model.pt"
            checkpoint_path.parent.mkdir(exist_ok=True)
        else:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_{self.global_step}.pt"
        
        checkpoint = {
            'global_step': self.global_step,
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': asdict(self.config),
        }
        
        if self.obs_normalizer:
            checkpoint['obs_normalizer'] = self.obs_normalizer.state_dict()
        if self.reward_normalizer:
            checkpoint['reward_normalizer'] = self.reward_normalizer.state_dict()
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.policy.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
        
        if 'obs_normalizer' in checkpoint and self.obs_normalizer:
            self.obs_normalizer.load_state_dict(checkpoint['obs_normalizer'])
        if 'reward_normalizer' in checkpoint and self.reward_normalizer:
            self.reward_normalizer.load_state_dict(checkpoint['reward_normalizer'])
        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Checkpoint loaded from: {checkpoint_path}")
        print(f"Resuming from step: {self.global_step}")
