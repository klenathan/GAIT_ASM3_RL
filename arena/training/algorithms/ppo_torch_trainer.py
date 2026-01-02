"""
PPO Trainer implementation using PyTorch-native PPO.
Replaces SB3 PPO with GPU-optimized implementation.
"""

from typing import Dict, Any
from arena.training.base_torch import BaseTorchTrainer
from arena.training.registry import AlgorithmRegistry
from arena.core.config import PPO_DEFAULT, PPO_GPU_DEFAULT
from arena.training.utils import resolve_activation_fn
from arena.training.schedules import get_lr_schedule
from arena.training.algorithms.ppo_torch import TorchPPO


@AlgorithmRegistry.register("ppo_torch")
class PPOTorchTrainer(BaseTorchTrainer):
    """Proximal Policy Optimization trainer using PyTorch."""
    
    algorithm_name = "ppo_torch"
    
    def __init__(self, config):
        super().__init__(config)
        # Initialize episode tracking buffer
        self._episode_buffer = {
            'rewards': [],
            'lengths': [],
            'wins': [],
            'enemies': [],
            'spawners': [],
        }
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get hyperparameters based on device."""
        hparams = PPO_GPU_DEFAULT if self._is_gpu() else PPO_DEFAULT
        
        # Ensure batch size is valid for the rollout size
        num_envs = self.env.num_envs if self.env else self.config.num_envs or 1
        rollout_size = hparams.n_steps * num_envs
        
        batch_size = hparams.batch_size
        if rollout_size % batch_size != 0:
            # Find a valid batch size
            for bs in (512, 256, 128, 64, 32):
                if bs <= rollout_size and rollout_size % bs == 0:
                    batch_size = bs
                    break
        
        if batch_size != hparams.batch_size:
            print(f"Adjusted PPO batch_size {hparams.batch_size} -> {batch_size} to divide rollout size.")
        
        return {
            'learning_rate': hparams.learning_rate,
            'n_steps': hparams.n_steps,
            'batch_size': batch_size,
            'n_epochs': hparams.n_epochs,
            'gamma': hparams.gamma,
            'gae_lambda': hparams.gae_lambda,
            'clip_range': hparams.clip_range,
            'ent_coef': hparams.ent_coef,
            'vf_coef': hparams.vf_coef,
            'max_grad_norm': hparams.max_grad_norm,
            'target_kl': hparams.target_kl,
        }
    
    def create_model(self) -> TorchPPO:
        """Create PPO model."""
        hparams = self.get_hyperparameters()
        
        # Get observation and action dimensions
        obs_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
        
        # Build learning rate schedule
        lr_schedule_fn = get_lr_schedule(
            schedule_type=self.config.lr_schedule,
            lr_start=hparams['learning_rate'],
            lr_end=self.config.lr_end,
            warmup_fraction=self.config.lr_warmup_fraction,
        )
        
        # Get activation function
        activation = resolve_activation_fn(self.config.ppo_activation)
        activation_name = {
            'ReLU': 'relu',
            'Tanh': 'tanh',
            'SiLU': 'silu',
        }.get(activation.__name__, 'silu')
        
        # Extract network architecture (handle dict or list format)
        net_arch = self.config.ppo_net_arch
        if isinstance(net_arch, dict):
            # Use policy network architecture
            hidden_sizes = tuple(net_arch.get('pi', [256, 256]))
        else:
            # Assume it's already a list/tuple
            hidden_sizes = tuple(net_arch)
        
        # Create model
        model = TorchPPO(
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=str(self.device),
            hidden_sizes=hidden_sizes,
            activation=activation_name,
            **hparams
        )
        
        # Store schedule function for later use
        self.lr_schedule_fn = lr_schedule_fn
        self.total_timesteps = self.config.total_timesteps
        
        print(f"✓ PPO model created with {sum(p.numel() for p in model.policy.parameters())} parameters")
        
        return model
    
    def train_step(self) -> Dict[str, float]:
        """Perform one PPO training iteration."""
        # Update learning rate based on schedule
        if hasattr(self, 'lr_schedule_fn'):
            progress = self.model.num_timesteps / self.total_timesteps
            new_lr = self.lr_schedule_fn(progress)
            self.model.set_learning_rate(new_lr)
        
        # Collect rollouts
        rollout, episode_infos = self.model.collect_rollouts(self.env, self.model.n_steps)
        
        # Process episode infos
        for info in episode_infos:
            if 'episode' in info:
                self._episode_buffer['rewards'].append(info['episode']['r'])
                self._episode_buffer['lengths'].append(info['episode']['l'])
                
                # Keep only last 100 episodes
                if len(self._episode_buffer['rewards']) > 100:
                    self._episode_buffer['rewards'].pop(0)
                    self._episode_buffer['lengths'].pop(0)
                
                # Track arena-specific metrics (from the episode-end info)
                win = info.get('win', False)
                spawners_killed = info.get('total_spawners_destroyed', 0)
                enemies_killed = info.get('total_enemies_destroyed', 0)
                episode_length = info.get('episode_steps', 0)
                episode_reward = info['episode']['r']
                
                self._episode_buffer['wins'].append(1 if win else 0)
                self._episode_buffer['enemies'].append(enemies_killed)
                self._episode_buffer['spawners'].append(spawners_killed)
                
                # Keep only last 100 episodes
                if len(self._episode_buffer['wins']) > 100:
                    self._episode_buffer['wins'].pop(0)
                if len(self._episode_buffer['enemies']) > 100:
                    self._episode_buffer['enemies'].pop(0)
                if len(self._episode_buffer['spawners']) > 100:
                    self._episode_buffer['spawners'].pop(0)
                
                # Record episode for curriculum
                if self.curriculum_manager:
                    self.curriculum_manager.record_episode(
                        spawners_killed=spawners_killed,
                        won=win,
                        length=episode_length,
                        reward=episode_reward,
                        enemy_kills=enemies_killed,
                    )
        
        # Train on collected data
        train_metrics = self.model.train_on_rollout(rollout)
        
        # Calculate metrics
        metrics = {
            **train_metrics,
            'time/total_timesteps': self.model.num_timesteps,
        }
        
        # Add episode metrics if available
        if len(self._episode_buffer['rewards']) > 0:
            metrics['rollout/ep_rew_mean'] = sum(self._episode_buffer['rewards']) / len(self._episode_buffer['rewards'])
            metrics['rollout/ep_len_mean'] = sum(self._episode_buffer['lengths']) / len(self._episode_buffer['lengths'])
            
            if len(self._episode_buffer['wins']) > 0:
                metrics['arena/win_rate'] = sum(self._episode_buffer['wins']) / len(self._episode_buffer['wins'])
                metrics['curriculum/win_rate'] = metrics['arena/win_rate']
            
            if len(self._episode_buffer['enemies']) > 0:
                metrics['arena/enemies_destroyed'] = sum(self._episode_buffer['enemies']) / len(self._episode_buffer['enemies'])
                metrics['curriculum/enemies_destroyed'] = metrics['arena/enemies_destroyed']
            
            if len(self._episode_buffer['spawners']) > 0:
                metrics['arena/spawners_destroyed'] = sum(self._episode_buffer['spawners']) / len(self._episode_buffer['spawners'])
                metrics['curriculum/spawners_destroyed'] = metrics['arena/spawners_destroyed']
        
        return metrics
    
    def _is_gpu(self) -> bool:
        """Check if using GPU device."""
        return self.device in ["cuda", "mps"]
