"""
GPU-accelerated vectorized environment wrapper for PyTorch-based RL.
Eliminates CPU bottleneck from SB3's SubprocVecEnv by batching operations on GPU.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional, Any, Dict
import gymnasium as gym
from collections import deque


class TorchVecEnv:
    """
    GPU-accelerated vectorized environment wrapper.
    
    Runs multiple environments in parallel and batches observations/actions on GPU.
    Unlike SB3's SubprocVecEnv which uses CPU multiprocessing, this keeps all
    operations on GPU for maximum throughput.
    """
    
    def __init__(
        self,
        env_fns: List[callable],
        device: str = "cuda",
        normalize_obs: bool = True,
        normalize_reward: bool = True,
        clip_obs: float = 10.0,
        clip_reward: float = 10.0,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
    ):
        """
        Args:
            env_fns: List of functions that create environments
            device: Device to run computations on ('cuda', 'mps', or 'cpu')
            normalize_obs: Whether to normalize observations
            normalize_reward: Whether to normalize rewards
            clip_obs: Clip observations to this range
            clip_reward: Clip rewards to this range
            gamma: Discount factor for reward normalization
            epsilon: Small constant for numerical stability
        """
        self.num_envs = len(env_fns)
        self.envs = [fn() for fn in env_fns]
        self.device = torch.device(device)
        
        # Get spaces from first env
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        
        # Normalization parameters
        self.normalize_obs = normalize_obs
        self.normalize_reward = normalize_reward
        self.clip_obs = clip_obs
        self.clip_reward = clip_reward
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Running statistics for normalization (on GPU)
        if self.normalize_obs:
            obs_shape = self.observation_space.shape
            self.obs_mean = torch.zeros(obs_shape, device=self.device)
            self.obs_var = torch.ones(obs_shape, device=self.device)
            self.obs_count = torch.tensor(0.0, device=self.device)
        
        if self.normalize_reward:
            self.ret_mean = torch.zeros(1, device=self.device)
            self.ret_var = torch.ones(1, device=self.device)
            self.ret_count = torch.tensor(0.0, device=self.device)
            self.returns = torch.zeros(self.num_envs, device=self.device)
        
        self._episode_rewards = [0.0] * self.num_envs
        self._episode_lengths = [0] * self.num_envs
        
    def reset(self, seed: Optional[int] = None) -> torch.Tensor:
        """Reset all environments and return initial observations."""
        obs_list = []
        for i, env in enumerate(self.envs):
            if seed is not None:
                obs, _ = env.reset(seed=seed + i)
            else:
                obs, _ = env.reset()
            obs_list.append(obs)
        
        obs = np.stack(obs_list)
        obs_tensor = torch.from_numpy(obs).float().to(self.device)
        
        if self.normalize_obs:
            obs_tensor = self._normalize_obs(obs_tensor)
        
        if self.normalize_reward:
            self.returns.zero_()
        
        return obs_tensor
    
    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[Dict]]:
        """
        Step all environments with the given actions.
        
        Args:
            actions: Tensor of shape (num_envs, *action_shape)
            
        Returns:
            obs: Tensor of shape (num_envs, *obs_shape)
            rewards: Tensor of shape (num_envs,)
            dones: Tensor of shape (num_envs,)
            truncated: Tensor of shape (num_envs,)
            infos: List of info dicts
        """
        # Convert actions to numpy for gym environments
        if isinstance(actions, torch.Tensor):
            actions_np = actions.cpu().numpy()
        else:
            actions_np = actions
        
        obs_list = []
        rewards_list = []
        dones_list = []
        truncated_list = []
        infos = []
        
        for i, (env, action) in enumerate(zip(self.envs, actions_np)):
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Track episode statistics
            self._episode_rewards[i] += reward
            self._episode_lengths[i] += 1
            
            done = terminated or truncated
            
            if done:
                # Add episode info
                info['episode'] = {
                    'r': self._episode_rewards[i],
                    'l': self._episode_lengths[i],
                }
                self._episode_rewards[i] = 0.0
                self._episode_lengths[i] = 0
                
                # Reset environment
                obs, _ = env.reset()
            
            obs_list.append(obs)
            rewards_list.append(reward)
            dones_list.append(terminated)
            truncated_list.append(truncated)
            infos.append(info)
        
        # Convert to tensors on GPU
        obs = torch.from_numpy(np.stack(obs_list)).float().to(self.device)
        rewards = torch.tensor(rewards_list, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones_list, dtype=torch.bool, device=self.device)
        truncated = torch.tensor(truncated_list, dtype=torch.bool, device=self.device)
        
        # Normalize observations
        if self.normalize_obs:
            obs = self._normalize_obs(obs)
        
        # Normalize rewards
        if self.normalize_reward:
            rewards = self._normalize_reward(rewards, dones)
        
        return obs, rewards, dones, truncated, infos
    
    def _normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Normalize observations using running mean and std."""
        # Update running statistics
        batch_mean = obs.mean(dim=0)
        batch_var = obs.var(dim=0)
        batch_count = obs.shape[0]
        
        self._update_mean_var_count(
            batch_mean, batch_var, batch_count,
            self.obs_mean, self.obs_var, self.obs_count
        )
        
        # Normalize
        obs = (obs - self.obs_mean) / torch.sqrt(self.obs_var + self.epsilon)
        
        # Clip
        if self.clip_obs > 0:
            obs = torch.clamp(obs, -self.clip_obs, self.clip_obs)
        
        return obs
    
    def _normalize_reward(self, rewards: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        """Normalize rewards using running return statistics."""
        # Update returns
        self.returns = self.returns * self.gamma + rewards
        
        # Update running statistics before normalization
        self._update_mean_var_count(
            self.returns.mean(), 
            self.returns.var(), 
            float(self.num_envs),
            self.ret_mean, 
            self.ret_var, 
            self.ret_count
        )
        
        # Normalize
        rewards = rewards / torch.sqrt(self.ret_var + self.epsilon)
        
        # Clip
        if self.clip_reward > 0:
            rewards = torch.clamp(rewards, -self.clip_reward, self.clip_reward)
        
        # Reset returns for done episodes
        self.returns = torch.where(dones, torch.zeros_like(self.returns), self.returns)
        
        return rewards
    
    def _update_mean_var_count(
        self,
        batch_mean: torch.Tensor,
        batch_var: torch.Tensor,
        batch_count: float,
        mean: torch.Tensor,
        var: torch.Tensor,
        count: torch.Tensor,
    ):
        """Update running mean and variance using Welford's online algorithm."""
        delta = batch_mean - mean
        tot_count = count + batch_count
        
        mean.copy_(mean + delta * batch_count / tot_count)
        m_a = var * count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * count * batch_count / tot_count
        var.copy_(m2 / tot_count)
        count.copy_(tot_count)
    
    def get_normalization_stats(self) -> Dict[str, Any]:
        """Get current normalization statistics for saving."""
        stats = {}
        if self.normalize_obs:
            stats['obs_mean'] = self.obs_mean.cpu().numpy()
            stats['obs_var'] = self.obs_var.cpu().numpy()
            stats['obs_count'] = self.obs_count.cpu().item()
        if self.normalize_reward:
            stats['ret_mean'] = self.ret_mean.cpu().numpy()
            stats['ret_var'] = self.ret_var.cpu().numpy()
            stats['ret_count'] = self.ret_count.cpu().item()
        return stats
    
    def set_normalization_stats(self, stats: Dict[str, Any]):
        """Load normalization statistics from checkpoint."""
        if 'obs_mean' in stats and self.normalize_obs:
            self.obs_mean = torch.from_numpy(stats['obs_mean']).to(self.device)
            self.obs_var = torch.from_numpy(stats['obs_var']).to(self.device)
            self.obs_count = torch.tensor(stats['obs_count'], device=self.device)
        if 'ret_mean' in stats and self.normalize_reward:
            self.ret_mean = torch.from_numpy(stats['ret_mean']).to(self.device)
            self.ret_var = torch.from_numpy(stats['ret_var']).to(self.device)
            self.ret_count = torch.tensor(stats['ret_count'], device=self.device)
    
    def close(self):
        """Close all environments."""
        for env in self.envs:
            env.close()
    
    @property
    def unwrapped(self):
        """Return the unwrapped environments."""
        return self.envs
