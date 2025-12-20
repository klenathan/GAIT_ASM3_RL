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
        
        # Episode tracking
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
        # Arena-specific metrics
        self.enemies_destroyed_total = 0
        self.spawners_destroyed_total = 0
        self.max_phase_reached = 0
        
    def _on_step(self) -> bool:
        """
        Called at every step
        """
        # Get the most recent info
        if len(self.locals['infos']) > 0:
            info = self.locals['infos'][0]
            
            # Track episode progress
            self.current_episode_reward += self.locals['rewards'][0]
            self.current_episode_length += 1
            
            # Check if episode is done
            if self.locals['dones'][0]:
                self.episode_count += 1
                
                # Log episode metrics
                self.logger.record('rollout/ep_rew_mean', self.current_episode_reward)
                self.logger.record('rollout/ep_len_mean', self.current_episode_length)
                
                # Arena-specific metrics
                if 'enemies_destroyed' in info:
                    self.logger.record('arena/enemies_destroyed', info['enemies_destroyed'])
                    self.enemies_destroyed_total += info['enemies_destroyed']
                
                if 'spawners_destroyed' in info:
                    self.logger.record('arena/spawners_destroyed', info['spawners_destroyed'])
                    self.spawners_destroyed_total += info['spawners_destroyed']
                
                if 'phase' in info:
                    self.logger.record('arena/phase_reached', info['phase'])
                    self.max_phase_reached = max(self.max_phase_reached, info['phase'])
                
                if 'player_health' in info:
                    self.logger.record('arena/final_health', info['player_health'])
                
                # Cumulative metrics
                self.logger.record('arena/total_enemies_destroyed', self.enemies_destroyed_total)
                self.logger.record('arena/total_spawners_destroyed', self.spawners_destroyed_total)
                self.logger.record('arena/max_phase_reached', self.max_phase_reached)
                
                # Store for averaging
                self.episode_rewards.append(self.current_episode_reward)
                self.episode_lengths.append(self.current_episode_length)
                
                # Keep only last 100 episodes for averaging
                if len(self.episode_rewards) > 100:
                    self.episode_rewards.pop(0)
                    self.episode_lengths.pop(0)
                
                # Log averages
                if len(self.episode_rewards) > 0:
                    self.logger.record('arena/avg_reward_100ep', np.mean(self.episode_rewards))
                    self.logger.record('arena/avg_length_100ep', np.mean(self.episode_lengths))
                
                # Reset episode tracking
                self.current_episode_reward = 0
                self.current_episode_length = 0
                
                if self.verbose > 0 and self.episode_count % 10 == 0:
                    print(f"Episode {self.episode_count} | "
                          f"Avg Reward: {np.mean(self.episode_rewards):.2f} | "
                          f"Max Phase: {self.max_phase_reached}")
        
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
