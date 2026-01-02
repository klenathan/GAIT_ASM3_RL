"""
GPU-optimized PPO implementation using pure PyTorch.
Based on CleanRL patterns but integrated with arena training infrastructure.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class PPORolloutBuffer:
    """Storage for PPO rollout data on GPU."""
    obs: torch.Tensor
    actions: torch.Tensor
    logprobs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor
    advantages: Optional[torch.Tensor] = None
    returns: Optional[torch.Tensor] = None
    

class ActorCritic(nn.Module):
    """Combined actor-critic network for PPO."""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Tuple[int, ...] = (256, 256),
        activation: nn.Module = nn.SiLU,
    ):
        super().__init__()
        
        # Shared feature extractor
        layers = []
        prev_size = obs_dim
        for hidden_size in hidden_sizes[:-1]:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                activation(),
            ])
            prev_size = hidden_size
        
        self.shared_net = nn.Sequential(*layers) if layers else nn.Identity()
        
        # Policy head
        self.actor = nn.Sequential(
            nn.Linear(prev_size, hidden_sizes[-1]),
            activation(),
            nn.Linear(hidden_sizes[-1], action_dim),
        )
        
        # Value head
        self.critic = nn.Sequential(
            nn.Linear(prev_size, hidden_sizes[-1]),
            activation(),
            nn.Linear(hidden_sizes[-1], 1),
        )
        
        # Initialize weights with orthogonal initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Orthogonal initialization for better training stability."""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning action logits and value."""
        features = self.shared_net(obs)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value.squeeze(-1)
    
    def get_action_and_value(
        self, 
        obs: torch.Tensor, 
        action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log probability, entropy, and value.
        
        Returns:
            action: Sampled action (if action=None) or input action
            logprob: Log probability of the action
            entropy: Entropy of the action distribution
            value: State value estimate
        """
        logits, value = self(obs)
        probs = Categorical(logits=logits)
        
        if action is None:
            action = probs.sample()
        
        logprob = probs.log_prob(action)
        entropy = probs.entropy()
        
        return action, logprob, entropy, value
    
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get state value estimate."""
        _, value = self(obs)
        return value


class TorchPPO:
    """
    GPU-optimized PPO implementation.
    
    Key improvements over SB3:
    - All rollout collection and training on GPU
    - No CPU multiprocessing overhead
    - Direct tensor operations without numpy conversions
    - Efficient batched GAE computation
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        device: str = "cuda",
        # Network architecture
        hidden_sizes: Tuple[int, ...] = (256, 256),
        activation: str = "silu",
        # PPO hyperparameters
        learning_rate: float = 3e-4,
        n_steps: int = 4096,
        batch_size: int = 256,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        clip_range_vf: Optional[float] = None,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: Optional[float] = 0.03,
        normalize_advantage: bool = True,
    ):
        self.device = torch.device(device)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Hyperparameters
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.normalize_advantage = normalize_advantage
        
        # Network
        activation_fn = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'silu': nn.SiLU,
            'swish': nn.SiLU,
        }.get(activation.lower(), nn.SiLU)
        
        self.policy = ActorCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
            activation=activation_fn,
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate, eps=1e-5)
        
        # Training stats
        self.num_timesteps = 0
        self._current_lr = learning_rate
        
    def collect_rollouts(
        self,
        env,
        n_steps: int,
    ) -> Tuple[PPORolloutBuffer, List[Dict]]:
        """
        Collect rollouts using the current policy.
        All operations performed on GPU for maximum efficiency.
        
        Returns:
            rollout: PPORolloutBuffer with collected data
            episode_infos: List of completed episode info dicts
        """
        num_envs = env.num_envs
        
        # Pre-allocate tensors on GPU
        obs_buffer = torch.zeros((n_steps, num_envs, self.obs_dim), device=self.device)
        actions_buffer = torch.zeros((n_steps, num_envs), dtype=torch.long, device=self.device)
        logprobs_buffer = torch.zeros((n_steps, num_envs), device=self.device)
        rewards_buffer = torch.zeros((n_steps, num_envs), device=self.device)
        dones_buffer = torch.zeros((n_steps, num_envs), dtype=torch.bool, device=self.device)
        values_buffer = torch.zeros((n_steps, num_envs), device=self.device)
        
        # Track episode information
        episode_infos = []
        
        # Get initial observation
        if self.num_timesteps == 0:
            obs = env.reset()
        else:
            # Use the observation from the end of last rollout
            obs = getattr(self, '_last_obs', env.reset())
        
        self.policy.eval()
        with torch.no_grad():
            for step in range(n_steps):
                obs_buffer[step] = obs
                
                # Get action from policy
                action, logprob, _, value = self.policy.get_action_and_value(obs)
                
                actions_buffer[step] = action
                logprobs_buffer[step] = logprob
                values_buffer[step] = value
                
                # Step environment
                obs, reward, done, truncated, infos = env.step(action)
                
                rewards_buffer[step] = reward
                dones_buffer[step] = done
                
                # Collect episode statistics
                for info in infos:
                    if 'episode' in info:
                        episode_infos.append(info)
                
                self.num_timesteps += num_envs
            
            # Store last observation for next rollout
            self._last_obs = obs
            
            # Get final value for GAE computation
            next_value = self.policy.get_value(obs)
        
        self.policy.train()
        
        # Compute advantages and returns using GAE
        advantages, returns = self._compute_gae(
            rewards_buffer,
            values_buffer,
            dones_buffer,
            next_value,
        )
        
        rollout = PPORolloutBuffer(
            obs=obs_buffer.reshape(-1, self.obs_dim),
            actions=actions_buffer.reshape(-1),
            logprobs=logprobs_buffer.reshape(-1),
            rewards=rewards_buffer.reshape(-1),
            dones=dones_buffer.reshape(-1),
            values=values_buffer.reshape(-1),
            advantages=advantages.reshape(-1),
            returns=returns.reshape(-1),
        )
        
        return rollout, episode_infos
    
    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: Shape (n_steps, num_envs)
            values: Shape (n_steps, num_envs)
            dones: Shape (n_steps, num_envs)
            next_value: Shape (num_envs,)
        
        Returns:
            advantages: Shape (n_steps, num_envs)
            returns: Shape (n_steps, num_envs)
        """
        n_steps, num_envs = rewards.shape
        advantages = torch.zeros_like(rewards)
        lastgaelam = 0
        
        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                nextnonterminal = ~dones[t]
                nextvalues = next_value
            else:
                nextnonterminal = ~dones[t]
                nextvalues = values[t + 1]
            
            delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
        
        returns = advantages + values
        return advantages, returns
    
    def train_on_rollout(self, rollout: PPORolloutBuffer) -> Dict[str, float]:
        """
        Update policy using the collected rollout.
        
        Returns:
            Dictionary of training statistics
        """
        # Normalize advantages
        if self.normalize_advantage:
            advantages = (rollout.advantages - rollout.advantages.mean()) / (rollout.advantages.std() + 1e-8)
        else:
            advantages = rollout.advantages
        
        # Training statistics
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        total_loss = 0.0
        total_kl = 0.0
        total_clipfrac = 0.0
        n_updates = 0
        
        # Multiple epochs over the same data
        for epoch in range(self.n_epochs):
            # Random permutation for mini-batch SGD
            indices = torch.randperm(len(rollout.obs), device=self.device)
            
            # Process mini-batches
            for start_idx in range(0, len(rollout.obs), self.batch_size):
                end_idx = start_idx + self.batch_size
                batch_indices = indices[start_idx:end_idx]
                
                # Get batch data
                batch_obs = rollout.obs[batch_indices]
                batch_actions = rollout.actions[batch_indices]
                batch_logprobs = rollout.logprobs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = rollout.returns[batch_indices]
                batch_values = rollout.values[batch_indices]
                
                # Forward pass
                _, newlogprob, entropy, newvalue = self.policy.get_action_and_value(
                    batch_obs, batch_actions
                )
                
                # Policy loss (PPO clip objective)
                logratio = newlogprob - batch_logprobs
                ratio = logratio.exp()
                
                approx_kl = ((ratio - 1) - logratio).mean()
                
                # Clipped surrogate objective
                policy_loss_1 = -batch_advantages * ratio
                policy_loss_2 = -batch_advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                policy_loss = torch.max(policy_loss_1, policy_loss_2).mean()
                
                # Value loss
                if self.clip_range_vf is not None:
                    value_pred_clipped = batch_values + torch.clamp(
                        newvalue - batch_values,
                        -self.clip_range_vf,
                        self.clip_range_vf,
                    )
                    value_loss_1 = (newvalue - batch_returns).pow(2)
                    value_loss_2 = (value_pred_clipped - batch_returns).pow(2)
                    value_loss = 0.5 * torch.max(value_loss_1, value_loss_2).mean()
                else:
                    value_loss = 0.5 * (newvalue - batch_returns).pow(2).mean()
                
                # Entropy loss (encourage exploration)
                entropy_loss = entropy.mean()
                
                # Total loss
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy_loss
                
                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Tracking
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                total_loss += loss.item()
                total_kl += approx_kl.item()
                
                # Calculate clip fraction
                with torch.no_grad():
                    clipfrac = ((ratio - 1.0).abs() > self.clip_range).float().mean().item()
                    total_clipfrac += clipfrac
                
                n_updates += 1
            
            # Early stopping based on KL divergence
            if self.target_kl is not None and total_kl / n_updates > self.target_kl:
                print(f"Early stopping at epoch {epoch + 1} due to reaching target KL")
                break
        
        # Return training statistics
        return {
            'train/policy_loss': total_policy_loss / n_updates,
            'train/value_loss': total_value_loss / n_updates,
            'train/entropy_loss': total_entropy_loss / n_updates,
            'train/total_loss': total_loss / n_updates,
            'train/approx_kl': total_kl / n_updates,
            'train/clip_fraction': total_clipfrac / n_updates,
            'train/learning_rate': self._current_lr,
        }
    
    def set_learning_rate(self, lr: float):
        """Update learning rate (for schedules)."""
        self._current_lr = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'num_timesteps': self.num_timesteps,
        }, path)
    
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.num_timesteps = checkpoint.get('num_timesteps', 0)
    
    @torch.no_grad()
    def predict(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Predict action for given observation."""
        self.policy.eval()
        
        if not isinstance(obs, torch.Tensor):
            obs = torch.from_numpy(obs).float().to(self.device)
        
        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)
        
        logits, _ = self.policy(obs)
        
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            probs = Categorical(logits=logits)
            action = probs.sample()
        
        return action.cpu().numpy()
