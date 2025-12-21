"""
Custom callbacks for training monitoring and TensorBoard logging
"""

from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import time


class ArenaCallback(BaseCallback):
    """
    Custom callback for tracking arena-specific metrics during training
    Logs to TensorBoard for monitoring
    """
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0
        self.episode_wins = []
        
        # Episode tracking (supports VecEnv)
        self.current_episode_reward = None  # np.ndarray shape (n_envs,)
        self.current_episode_length = None  # np.ndarray shape (n_envs,)
        
        # Arena-specific metrics
        self.enemies_destroyed_total = 0
        self.spawners_destroyed_total = 0
        self.max_phase_reached = 0

    def _on_training_start(self) -> None:
        """Initialize per-env episode trackers for VecEnv."""
        try:
            n_envs = int(getattr(self.training_env, "num_envs", 1))
        except Exception:
            n_envs = 1
        self.current_episode_reward = np.zeros(n_envs, dtype=np.float32)
        self.current_episode_length = np.zeros(n_envs, dtype=np.int32)
        
    def _on_step(self) -> bool:
        """
        Called at every step
        """
        infos = self.locals.get("infos", None)
        rewards = self.locals.get("rewards", None)
        dones = self.locals.get("dones", None)

        if infos is None or rewards is None or dones is None:
            return True

        # Ensure per-env trackers exist (covers edge cases where _on_training_start didn't fire)
        if self.current_episode_reward is None or self.current_episode_length is None:
            try:
                n_envs = len(infos)
            except Exception:
                n_envs = 1
            self.current_episode_reward = np.zeros(n_envs, dtype=np.float32)
            self.current_episode_length = np.zeros(n_envs, dtype=np.int32)

        rewards_arr = np.asarray(rewards, dtype=np.float32).reshape(-1)
        dones_arr = np.asarray(dones, dtype=bool).reshape(-1)

        # Track episode progress for all envs
        n = min(len(rewards_arr), len(self.current_episode_reward))
        self.current_episode_reward[:n] += rewards_arr[:n]
        self.current_episode_length[:n] += 1

        # Only do heavier work when at least one episode ends (big overhead win on VecEnv)
        if not np.any(dones_arr[:n]):
            return True

        finished_idxs = np.flatnonzero(dones_arr[:n])

        # Aggregate finished episodes for logging
        finished_rewards = self.current_episode_reward[finished_idxs]
        finished_lengths = self.current_episode_length[finished_idxs]

        for ep_rew, ep_len in zip(finished_rewards, finished_lengths):
            self.episode_count += 1
            self.episode_rewards.append(float(ep_rew))
            self.episode_lengths.append(int(ep_len))
            if len(self.episode_rewards) > 100:
                self.episode_rewards.pop(0)
                self.episode_lengths.pop(0)

        # Arena-specific metrics from infos on done envs (sum/peak)
        enemies_destroyed = 0
        spawners_destroyed = 0
        phase_reached = None
        final_health = None
        wins = 0
        spawners_per_min = []
        time_to_first_kill = []
        time_to_win = []

        for i in finished_idxs:
            try:
                info = infos[i]
            except Exception:
                continue
            if isinstance(info, dict):
                enemies_destroyed += int(info.get("enemies_destroyed", 0))
                spawners_destroyed += int(info.get("spawners_destroyed", 0))
                if "phase" in info:
                    phase_val = int(info["phase"])
                    phase_reached = phase_val if phase_reached is None else max(phase_reached, phase_val)
                if "player_health" in info:
                    final_health = float(info["player_health"])

                win = bool(info.get("win", False))
                wins += int(win)

                ep_steps = int(info.get("episode_steps", 0))
                if ep_steps > 0:
                    # steps are frames; FPS is config.FPS (60 default)
                    spm = (float(info.get("spawners_destroyed", 0)) * float(60 * 60)) / float(ep_steps)
                    spawners_per_min.append(spm)

                t_first = int(info.get("first_spawner_kill_step", -1))
                if t_first >= 0:
                    time_to_first_kill.append(t_first)

                t_win = int(info.get("win_step", -1))
                if t_win >= 0:
                    time_to_win.append(t_win)

        self.enemies_destroyed_total += enemies_destroyed
        self.spawners_destroyed_total += spawners_destroyed
        if phase_reached is not None:
            self.max_phase_reached = max(self.max_phase_reached, phase_reached)

        # Log episode summary (means across envs that finished this step)
        self.logger.record("arena/ep_rew_mean", float(np.mean(finished_rewards)))
        self.logger.record("arena/ep_len_mean", float(np.mean(finished_lengths)))
        self.logger.record("arena/wins", int(wins))

        if len(spawners_per_min) > 0:
            self.logger.record("arena/spawners_per_min_mean", float(np.mean(spawners_per_min)))
        if len(time_to_first_kill) > 0:
            self.logger.record("arena/time_to_first_spawner_kill_mean", float(np.mean(time_to_first_kill)))
        if len(time_to_win) > 0:
            self.logger.record("arena/time_to_win_mean", float(np.mean(time_to_win)))

        if enemies_destroyed:
            self.logger.record("arena/enemies_destroyed", enemies_destroyed)
        if spawners_destroyed:
            self.logger.record("arena/spawners_destroyed", spawners_destroyed)
        if phase_reached is not None:
            self.logger.record("arena/phase_reached", phase_reached)
        if final_health is not None:
            self.logger.record("arena/final_health", final_health)

        # Cumulative + rolling averages
        self.logger.record("arena/total_enemies_destroyed", self.enemies_destroyed_total)
        self.logger.record("arena/total_spawners_destroyed", self.spawners_destroyed_total)
        self.logger.record("arena/max_phase_reached", self.max_phase_reached)
        if len(self.episode_rewards) > 0:
            self.logger.record("arena/avg_reward_100ep", float(np.mean(self.episode_rewards)))
            self.logger.record("arena/avg_length_100ep", float(np.mean(self.episode_lengths)))

        # Rolling win rate (100 episodes, across all envs)
        if len(finished_idxs) > 0:
            self.episode_wins.extend([1] * int(wins))
            self.episode_wins.extend([0] * int(len(finished_idxs) - wins))
            if len(self.episode_wins) > 100:
                self.episode_wins = self.episode_wins[-100:]
            self.logger.record("arena/win_rate_100ep", float(np.mean(self.episode_wins)) if self.episode_wins else 0.0)

        # Reset finished env trackers
        self.current_episode_reward[finished_idxs] = 0.0
        self.current_episode_length[finished_idxs] = 0

        if self.verbose > 0 and self.episode_count % 10 == 0:
            print(
                f"Episode {self.episode_count} | "
                f"Avg Reward(100): {np.mean(self.episode_rewards):.2f} | "
                f"Max Phase: {self.max_phase_reached}"
            )
        
        return True


class PerformanceCallback(BaseCallback):
    """
    Callback to track training performance metrics (FPS, timing, etc.)
    """
    
    def __init__(self, verbose=0, log_freq=1000):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.start_time = None
        self.last_log_time = None
        self.last_log_steps = 0
        
    def _on_training_start(self) -> None:
        """Called before the first rollout"""
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.last_log_steps = 0
        
    def _on_step(self) -> bool:
        """Track performance metrics"""
        if self.n_calls % self.log_freq == 0:
            current_time = time.time()
            
            # Calculate FPS since last log
            steps_since_last = self.num_timesteps - self.last_log_steps
            time_since_last = current_time - self.last_log_time
            
            if time_since_last > 0:
                fps = steps_since_last / time_since_last
                self.logger.record('performance/fps', fps)
                
                # Total elapsed time
                total_elapsed = current_time - self.start_time
                self.logger.record('performance/time_elapsed_seconds', total_elapsed)
                self.logger.record('performance/time_elapsed_minutes', total_elapsed / 60)
                
                # Average FPS over entire training
                avg_fps = self.num_timesteps / total_elapsed if total_elapsed > 0 else 0
                self.logger.record('performance/avg_fps', avg_fps)
                
                if self.verbose > 0:
                    print(f"Performance | FPS: {fps:.1f} | Avg FPS: {avg_fps:.1f} | "
                          f"Steps: {self.num_timesteps:,} | Time: {total_elapsed/60:.1f}m")
            
            # Update tracking
            self.last_log_time = current_time
            self.last_log_steps = self.num_timesteps
        
        return True


class RenderCallback(BaseCallback):
    """
    Callback to periodically render the environment during training
    """
    
    def __init__(self, render_freq=1000, verbose=0):
        super().__init__(verbose)
        self.render_freq = render_freq
        
    def _on_step(self) -> bool:
        """Render every N steps"""
        if self.n_calls % self.render_freq == 0:
            if hasattr(self.training_env, 'envs'):
                # VecEnv
                env = self.training_env.envs[0]
                if hasattr(env, 'render'):
                    env.render()
            elif hasattr(self.training_env, 'render'):
                # Single env
                self.training_env.render()
        
        return True
