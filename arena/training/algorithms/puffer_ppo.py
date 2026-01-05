"""
PufferLib-optimized PPO Trainer.
Implementation adapted from CleanRL and PufferLib examples.
"""

import os
import time
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import pufferlib
import pufferlib.emulation
import pufferlib.vector
import pufferlib.models

from arena.core.config import (
    TrainerConfig,
    NUM_ENVS_DEFAULT_CUDA,
    NUM_ENVS_DEFAULT_MPS,
    NUM_ENVS_DEFAULT_CPU,
)
from arena.core.device import DeviceManager
from arena.core.env_puffer import make_puffer_env
from arena.training.registry import AlgorithmRegistry


@AlgorithmRegistry.register("puffer_ppo")
class PufferPPOTrainer:
    def __init__(self, config: TrainerConfig):
        self.config = config
        
        # Simple device selection
        if config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            if self.device == "cpu" and torch.backends.mps.is_available():
                self.device = "mps"
        else:
            self.device = config.device

        # PPO Hyperparameters from config
        self.total_timesteps = config.total_timesteps
        self.learning_rate = config.learning_rate
        
        # Set num_envs based on config or defaults
        if config.num_envs:
            self.num_envs = config.num_envs
        else:
            if self.device == "cuda":
                self.num_envs = NUM_ENVS_DEFAULT_CUDA
            elif self.device == "mps":
                self.num_envs = NUM_ENVS_DEFAULT_MPS
            else:
                self.num_envs = NUM_ENVS_DEFAULT_CPU
                
        self.num_steps = config.num_steps
        self.anneal_lr = True
        self.gamma = config.gamma
        self.gae_lambda = config.gae_lambda
        self.num_minibatches = config.num_minibatches
        self.update_epochs = config.update_epochs
        self.norm_adv = True
        self.clip_coef = config.clip_coef
        self.clip_vloss = True
        self.ent_coef = config.ent_coef
        self.vf_coef = config.vf_coef
        self.max_grad_norm = config.max_grad_norm
        self.target_kl = None
        
        # Mixed Precision Training Scaler
        if self.device == "cuda":
            self.scaler = torch.amp.GradScaler("cuda")
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=False)

        self.batch_size = int(self.num_envs * self.num_steps)
        self.minibatch_size = int(self.batch_size // self.num_minibatches)

        self.run_name = f"puffer_ppo_style{config.style}_{int(time.time())}"
        self.log_path = os.path.join(config.runs_dir, self.run_name)
        self.writer = SummaryWriter(self.log_path)

        print(f"Initializing PufferPPOTrainer on {self.device} with {self.num_envs} envs")

    def train(self):
        # Vectorization
        # PufferLib v3: Multiprocessing(env_creators, env_args, env_kwargs, num_envs, num_workers=...)
        
        # Use all available CPU cores for workers
        num_workers = os.cpu_count() or 1
        
        # Enable overwork to allow more envs than cores (essential for Colab)
        vec_env = pufferlib.vector.Multiprocessing(
            [make_puffer_env] * self.num_envs,
            [[] for _ in range(self.num_envs)],
            [{"config": self.config} for _ in range(self.num_envs)],
            self.num_envs,
            num_workers=num_workers,
            overwork=True,
        )

        # Policy Network
        # Using Default PufferLib Policy
        policy = pufferlib.models.Default(vec_env.driver_env, hidden_size=64).to(
            self.device
        )
        optimizer = optim.Adam(policy.parameters(), lr=self.learning_rate, eps=1e-5)

        # Storage based on single space
        obs_shape = vec_env.single_observation_space.shape
        action_shape = vec_env.single_action_space.shape
        obs = torch.zeros((self.num_steps, self.num_envs) + obs_shape).to(self.device)
        actions = torch.zeros((self.num_steps, self.num_envs) + action_shape).to(
            self.device
        )
        logprobs = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        rewards = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        dones = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        values = torch.zeros((self.num_steps, self.num_envs)).to(self.device)

        # Env Reset
        global_step = 0
        start_time = time.time()
        next_obs, _ = vec_env.reset()
        next_obs = torch.Tensor(next_obs).to(self.device)
        next_done = torch.zeros(self.num_envs).to(self.device)
        next_lstm_state = None

        num_updates = self.total_timesteps // self.batch_size

        print(
            f"Starting training loop for {self.total_timesteps} timesteps ({num_updates} updates)"
        )

        for update in range(1, num_updates + 1):
            # Annealing the rate if instructed to do so.
            if self.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = frac * self.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

            # Rollout
            for step in range(0, self.num_steps):
                global_step += 1 * self.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                with torch.no_grad():
                    logits, value = policy(next_obs)
                    probs = torch.distributions.Categorical(logits=logits)
                    action = probs.sample()
                    logprob = probs.log_prob(action)
                    values[step] = value.flatten()

                actions[step] = action
                logprobs[step] = logprob

                # Execute game step
                # PufferLib vector clients expect numpy actions
                cpu_action = action.cpu().numpy()
                next_obs, reward, done, truncated, info = vec_env.step(cpu_action)

                rewards[step] = torch.tensor(reward).to(self.device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(
                    self.device
                ), torch.Tensor(done).to(self.device)

                # Logging
                for item in info:
                    if "episode" in item.keys():
                        print(
                            f"global_step={global_step}, episode_reward={item['episode']['r']}"
                        )
                        self.writer.add_scalar(
                            "charts/episodic_return", item["episode"]["r"], global_step
                        )
                        self.writer.add_scalar(
                            "charts/episodic_length", item["episode"]["l"], global_step
                        )
                        break

            # Bootstrap value if not done
            with torch.no_grad():
                logits, next_value = policy(next_obs)
                next_value = next_value.reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(self.device)
                lastgaelam = 0
                for t in reversed(range(self.num_steps)):
                    if t == self.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = (
                        rewards[t]
                        + self.gamma * nextvalues * nextnonterminal
                        - values[t]
                    )
                    advantages[t] = lastgaelam = (
                        delta
                        + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                    )
                returns = advantages + values

            # Flatten batch
            b_obs = obs.reshape((-1,) + vec_env.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + vec_env.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(self.batch_size)
            clipfracs = []

            for epoch in range(self.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, self.batch_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    mb_inds = b_inds[start:end]

                    with torch.cuda.amp.autocast(enabled=(self.device == "cuda")):
                        logits, newvalue = policy(b_obs[mb_inds])
                        probs = torch.distributions.Categorical(logits=logits)
                        newlogprob = probs.log_prob(
                            b_actions[mb_inds].squeeze()
                        )  # Squeeze if needed since actions might be (N, 1) or (N)
                        entropy = probs.entropy()

                        logratio = newlogprob - b_logprobs[mb_inds]
                        ratio = logratio.exp()

                        with torch.no_grad():
                            # calculate approx_kl http://joschu.net/blog/kl-approx.html
                            old_approx_kl = (-logratio).mean()
                            approx_kl = ((ratio - 1) - logratio).mean()
                            clipfracs += [
                                ((ratio - 1.0).abs() > self.clip_coef).float().mean().item()
                            ]

                        mb_advantages = b_advantages[mb_inds]
                        if self.norm_adv:
                            mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                                mb_advantages.std() + 1e-8
                            )

                        # Policy loss
                        pg_loss1 = -mb_advantages * ratio
                        pg_loss2 = -mb_advantages * torch.clamp(
                            ratio, 1 - self.clip_coef, 1 + self.clip_coef
                        )
                        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                        # Value loss
                        newvalue = newvalue.view(-1)
                        if self.clip_vloss:
                            v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                            v_clipped = b_values[mb_inds] + torch.clamp(
                                newvalue - b_values[mb_inds],
                                -self.clip_coef,
                                self.clip_coef,
                            )
                            v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                            v_loss = 0.5 * v_loss_max.mean()
                        else:
                            v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                        entropy_loss = entropy.mean()
                        loss = (
                            pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef
                        )

                    optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(policy.parameters(), self.max_grad_norm)
                    self.scaler.step(optimizer)
                    self.scaler.update()

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = (
                np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            )

            # Logging metrics
            self.writer.add_scalar(
                "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
            )
            self.writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            self.writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            self.writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            self.writer.add_scalar(
                "losses/old_approx_kl", old_approx_kl.item(), global_step
            )
            self.writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            self.writer.add_scalar(
                "losses/explained_variance", explained_var, global_step
            )
            print(f"SPS: {int(global_step / (time.time() - start_time))}")
            self.writer.add_scalar(
                "charts/SPS", int(global_step / (time.time() - start_time)), global_step
            )

            # Checkpoint at frequency
            if update % 10 == 0:
                torch.save(
                    policy.state_dict(),
                    os.path.join(self.log_path, f"model_{global_step}.pt"),
                )

        # Final save
        torch.save(policy.state_dict(), os.path.join(self.log_path, "model_final.pt"))
        vec_env.close()
        self.writer.close()

        return policy
