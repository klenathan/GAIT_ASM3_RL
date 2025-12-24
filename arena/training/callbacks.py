"""
Custom callbacks for training monitoring and TensorBoard logging.
"""

from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
import numpy as np
import time
import os

from arena.training.training_state import save_training_state, get_training_state_path, TrainingState


class LearningRateCallback(BaseCallback):
    """Logs the current learning rate to TensorBoard."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
    
    def _on_step(self) -> bool:
        # Get current learning rate from the model's optimizer
        # SB3 stores this in model.lr_schedule which is called with progress_remaining
        if hasattr(self.model, "lr_schedule"):
            progress_remaining = 1.0 - (self.num_timesteps / self.model._total_timesteps)
            current_lr = self.model.lr_schedule(progress_remaining)
            self.logger.record("train/learning_rate", current_lr)
        return True


class ArenaCallback(BaseCallback):
    """Tracks arena-specific metrics and logs them to TensorBoard."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_enemies = []
        self.episode_spawners = []
        self.episode_wins = []
        self.episode_count = 0
        
        self.current_episode_reward = None
        self.current_episode_length = None
        self.current_episode_enemies = None
        self.current_episode_spawners = None
        
        self.enemies_destroyed_total = 0
        self.spawners_destroyed_total = 0
        self.max_phase_reached = 0

    def _on_training_start(self) -> None:
        n_envs = getattr(self.training_env, "num_envs", 1)
        self.current_episode_reward = np.zeros(n_envs, dtype=np.float32)
        self.current_episode_length = np.zeros(n_envs, dtype=np.int32)
        self.current_episode_enemies = np.zeros(n_envs, dtype=np.int32)
        self.current_episode_spawners = np.zeros(n_envs, dtype=np.int32)
        
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [])
        dones = self.locals.get("dones", [])

        if not infos or rewards is None or dones is None: return True

        self.current_episode_reward += np.asarray(rewards)
        self.current_episode_length += 1
        
        # Track increments for totals and episode-specific counts
        phases = []
        for i, info in enumerate(infos):
            e = int(info.get("enemies_destroyed", 0))
            s = int(info.get("spawners_destroyed", 0))
            self.current_episode_enemies[i] += e
            self.current_episode_spawners[i] += s
            self.enemies_destroyed_total += e
            self.spawners_destroyed_total += s
            
            phase = int(info.get("phase", 0))
            self.max_phase_reached = max(self.max_phase_reached, phase)
            phases.append(phase)
        
        mean_phase = float(np.mean(phases)) if phases else 0.0

        if not np.any(dones): return True

        finished = np.flatnonzero(dones)
        for i in finished:
            self.episode_count += 1
            self.episode_rewards.append(float(self.current_episode_reward[i]))
            self.episode_lengths.append(int(self.current_episode_length[i]))
            self.episode_enemies.append(int(self.current_episode_enemies[i]))
            self.episode_spawners.append(int(self.current_episode_spawners[i]))
            
            info = infos[i]
            win = bool(info.get("win", False))
            self.episode_wins.append(1 if win else 0)
            
            # Keep only last 100 episodes
            for lst in [self.episode_rewards, self.episode_lengths, 
                        self.episode_enemies, self.episode_spawners, self.episode_wins]:
                if len(lst) > 100: lst.pop(0)

            self.current_episode_reward[i] = 0.0
            self.current_episode_length[i] = 0
            self.current_episode_enemies[i] = 0
            self.current_episode_spawners[i] = 0

        # Persistent recording (ensures metrics appear in every TB dump)
        # Mean metrics (only available after at least one episode finishes)
        if len(self.episode_rewards) > 0:
            self.logger.record("arena/ep_rew_mean", float(np.mean(self.episode_rewards)))
            self.logger.record("arena/ep_len_mean", float(np.mean(self.episode_lengths)))
            self.logger.record("arena/ep_enemies_mean", float(np.mean(self.episode_enemies)))
            self.logger.record("arena/ep_spawners_mean", float(np.mean(self.episode_spawners)))
            self.logger.record("arena/win_rate_100ep", float(np.mean(self.episode_wins)))
            self.logger.record("arena/ep_enemies_last", self.episode_enemies[-1])
            self.logger.record("arena/ep_spawners_last", self.episode_spawners[-1])

        # Current episode progress (averaged across all environments)
        self.logger.record("arena/cur_ep_enemies", float(np.mean(self.current_episode_enemies)))
        self.logger.record("arena/cur_ep_spawners", float(np.mean(self.current_episode_spawners)))
        self.logger.record("arena/cur_ep_rew", float(np.mean(self.current_episode_reward)))

        # Global totals
        self.logger.record("arena/total_enemies_destroyed", self.enemies_destroyed_total)
        self.logger.record("arena/total_spawners_destroyed", self.spawners_destroyed_total)
        self.logger.record("arena/max_phase_reached", self.max_phase_reached)
        self.logger.record("arena/mean_phase_reached", mean_phase)

        return True

class PerformanceCallback(BaseCallback):
    """Tracks training performance (FPS, elapsed time)."""
    
    def __init__(self, log_freq=1000, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.start_time = None
        self.last_log_time = None
        self.last_log_steps = 0
        
    def _on_training_start(self) -> None:
        self.start_time = time.time()
        self.last_log_time = self.start_time
        
    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            now = time.time()
            dt = now - self.last_log_time
            if dt > 0:
                fps = (self.num_timesteps - self.last_log_steps) / dt
                self.logger.record('performance/fps', fps)
                self.logger.record('performance/time_min', (now - self.start_time) / 60)
            self.last_log_time, self.last_log_steps = now, self.num_timesteps
        return True

class HParamCallback(BaseCallback):
    """
    Logs hyperparameters to TensorBoard as formatted text.
    
    Hyperparameters are logged as a markdown table in the TEXT tab,
    providing a clean single-run view without duplicate entries.
    """
    def __init__(self, hparams: dict):
        super().__init__()
        self.hparams = hparams

    def _on_training_start(self) -> None:
        for output_format in self.logger.output_formats:
            if hasattr(output_format, "writer"):
                # Format hyperparameters as a markdown table
                hparam_lines = ["| Hyperparameter | Value |", "|----------------|-------|"]
                for k, v in sorted(self.hparams.items()):
                    # Format values appropriately
                    if isinstance(v, float):
                        value_str = f"{v:.6g}"  # Format floats with up to 6 significant digits
                    elif isinstance(v, (int, str, bool)):
                        value_str = str(v)
                    else:
                        value_str = str(v)
                    hparam_lines.append(f"| {k} | {value_str} |")
                
                hparam_text = "\n".join(hparam_lines)
                # Log at timestep 0 so it appears at the start of training
                output_format.writer.add_text("hyperparameters", hparam_text, global_step=0)
                break
                
    def _on_step(self) -> bool: return True


class CurriculumCallback(BaseCallback):
    """
    Tracks episode outcomes and manages curriculum progression.
    Reports to CurriculumManager and logs stage to TensorBoard.
    """
    
    def __init__(self, curriculum_manager, verbose=0):
        super().__init__(verbose)
        self.curriculum_manager = curriculum_manager
        self.current_episode_spawners = None
        self.current_episode_reward = None
        self.current_episode_length = None
        self.current_episode_wins = None
    
    def _on_training_start(self) -> None:
        if not self.curriculum_manager or not self.curriculum_manager.enabled:
            return
        n_envs = getattr(self.training_env, "num_envs", 1)
        self.current_episode_spawners = np.zeros(n_envs, dtype=np.int32)
        self.current_episode_reward = np.zeros(n_envs, dtype=np.float32)
        self.current_episode_length = np.zeros(n_envs, dtype=np.int32)
        self.current_episode_wins = np.zeros(n_envs, dtype=np.int32)
        
        # Log initial stage
        self.logger.record("curriculum/stage", self.curriculum_manager.current_stage_index)
        self.logger.record("curriculum/stage_name", self.curriculum_manager.current_stage.name)
        
        if self.verbose > 0:
            print(f"[Curriculum] Starting at stage {self.curriculum_manager.current_stage_index}: "
                  f"{self.curriculum_manager.current_stage.name}")
    
    def _on_step(self) -> bool:
        if not self.curriculum_manager or not self.curriculum_manager.enabled:
            return True
            
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [])
        dones = self.locals.get("dones", [])
        
        if not infos or rewards is None or dones is None:
            return True
        
        # Accumulate episode stats
        self.current_episode_reward += np.asarray(rewards)
        self.current_episode_length += 1
        
        for i, info in enumerate(infos):
            self.current_episode_spawners[i] += int(info.get("spawners_destroyed", 0))
            if info.get("win", False):
                self.current_episode_wins[i] = 1
        
        # Process finished episodes
        if np.any(dones):
            finished = np.flatnonzero(dones)
            for i in finished:
                # Report to curriculum manager
                self.curriculum_manager.record_episode(
                    spawners_killed=int(self.current_episode_spawners[i]),
                    won=bool(self.current_episode_wins[i]),
                    length=int(self.current_episode_length[i]),
                    reward=float(self.current_episode_reward[i])
                )
                
                # Reset counters for this env
                self.current_episode_spawners[i] = 0
                self.current_episode_reward[i] = 0.0
                self.current_episode_length[i] = 0
                self.current_episode_wins[i] = 0
            
            # Check for advancement
            if self.curriculum_manager.check_advancement():
                stage = self.curriculum_manager.current_stage
                if self.verbose > 0:
                    print(f"[Curriculum] Advanced to stage {self.curriculum_manager.current_stage_index}: {stage.name}")
        
        # Log curriculum status
        self.logger.record("curriculum/stage", self.curriculum_manager.current_stage_index)
        
        return True


class CheckpointWithStateCallback(CheckpointCallback):
    """
    Extended CheckpointCallback that also saves training state.
    Enables proper transfer learning by preserving curriculum progress.
    """
    
    def __init__(
        self,
        curriculum_manager,
        save_freq: int,
        save_path: str,
        name_prefix: str = "rl_model",
        save_replay_buffer: bool = False,
        save_vecnormalize: bool = False,
        verbose: int = 0,
    ):
        super().__init__(
            save_freq=save_freq,
            save_path=save_path,
            name_prefix=name_prefix,
            save_replay_buffer=save_replay_buffer,
            save_vecnormalize=save_vecnormalize,
            verbose=verbose,
        )
        self.curriculum_manager = curriculum_manager
    
    def _on_step(self) -> bool:
        # Call parent to handle standard checkpoint saving
        result = super()._on_step()
        
        # If a checkpoint was just saved, also save training state
        if self.n_calls % self.save_freq == 0:
            if self.curriculum_manager:
                # Get the path to the just-saved checkpoint
                checkpoint_path = os.path.join(
                    self.save_path,
                    f"{self.name_prefix}_{self.num_timesteps}_steps.zip"
                )
                
                # Save training state
                curriculum_dict = self.curriculum_manager.to_dict()
                state = TrainingState(
                    model_path=checkpoint_path,
                    algo=getattr(self.model, "model_class", "unknown"),
                    style=0,  # Not available here, but preserved in filename
                    total_timesteps_completed=self.num_timesteps,
                    total_episodes=0,
                    curriculum_stage_index=curriculum_dict["current_stage_index"],
                    curriculum_metrics=curriculum_dict["metrics"],
                )
                
                state_path = get_training_state_path(checkpoint_path)
                save_training_state(state_path, state)
                
                if self.verbose > 0:
                    print(f"Training state saved: {state_path}")
        
        return result

