"""
Headless Evaluation Script for Deep RL Arena.
Efficiently runs thousands of episodes and generates comprehensive statistics for model comparison.

Usage:
    # Evaluate a single model
    python -m arena.eval_headless --model runs/ppo/style1/ppo_style1_20251225_175203/final/ppo_style1_20251225_175203_final.zip --episodes 1000

    # Compare multiple models
    python -m arena.eval_headless --models model1.zip model2.zip model3.zip --episodes 500 --compare

    # Parallel evaluation for speed
    python -m arena.eval_headless --model model.zip --episodes 10000 --workers 8

    # Stochastic evaluation
    python -m arena.eval_headless --model model.zip --episodes 1000 --stochastic

    # Save results to file
    python -m arena.eval_headless --model model.zip --episodes 1000 --output results.json
"""

# Suppress warnings BEFORE any imports to prevent pkg_resources deprecation warning from pygame
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import argparse
import json
import os
import glob
import time
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

from arena.training.registry import AlgorithmRegistry
from arena.training.algorithms import dqn, ppo, ppo_lstm, ppo_dict, a2c  # noqa: F401 - Import to register algorithms
from arena.training.training_state import find_training_state
from arena.core.environment_dict import ArenaDictEnv
from arena.core.environment import ArenaEnv
from arena.core.curriculum import CurriculumManager, CurriculumConfig
from arena.core.device import DeviceManager
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize


@dataclass
class EpisodeStats:
    """Statistics for a single episode."""
    episode_reward: float
    episode_length: int
    win: bool
    win_step: int
    phase_reached: int
    spawners_destroyed: int
    enemies_destroyed: int
    final_player_health: int
    first_spawner_kill_step: int


@dataclass
class ModelEvaluation:
    """Comprehensive evaluation statistics for a model."""
    model_path: str
    model_name: str
    algorithm: str
    style: int
    deterministic: bool
    total_episodes: int
    eval_time_seconds: float
    
    # Win statistics
    win_rate: float
    wins: int
    
    # Reward statistics
    mean_reward: float
    std_reward: float
    median_reward: float
    min_reward: float
    max_reward: float
    
    # Episode length statistics
    mean_length: float
    std_length: float
    median_length: float
    min_length: int
    max_length: int
    
    # Performance metrics
    mean_spawners_destroyed: float
    std_spawners_destroyed: float
    mean_enemies_destroyed: float
    mean_phase_reached: float
    mean_final_health: float
    
    # Timing metrics
    mean_win_step: float  # Average step at which wins occur (among wins only)
    mean_first_spawner_kill: float  # Average step for first spawner kill
    
    # Detailed distributions
    phase_distribution: Dict[int, int]  # phase -> count
    spawner_kill_distribution: Dict[int, int]  # kills -> count
    
    # Raw episode data (optional, for detailed analysis)
    episodes: Optional[List[EpisodeStats]] = None


class HeadlessEvaluator:
    """Efficient headless evaluation of RL models."""
    
    def __init__(self, device: str = "auto", verbose: bool = True):
        self.device = DeviceManager.get_device(device)
        self.verbose = verbose
    
    def _log(self, msg: str):
        """Print message if verbose."""
        if self.verbose:
            print(msg)
    
    def _infer_algo(self, model_path: str) -> str:
        """Infer algorithm type from filename."""
        name = os.path.basename(model_path).lower()
        algos = sorted(AlgorithmRegistry.list_algorithms(), key=len, reverse=True)
        for algo in algos:
            if algo in name:
                return algo
        return "ppo"
    
    def _infer_style(self, model_path: str) -> int:
        """Infer control style from model path."""
        path_lower = model_path.lower()
        if 'style2' in path_lower:
            return 2
        elif 'style1' in path_lower:
            return 1
        else:
            # Default to style 1 if not found in path
            return 1
    
    def _find_vecnormalize_stats(self, model_path: str) -> Optional[str]:
        """Find VecNormalize stats file matching the model."""
        import re
        
        model_dir = os.path.dirname(model_path)
        model_name = os.path.basename(model_path).replace('.zip', '')
        
        # Try to extract step count from model filename (e.g., "3600000" from "model_3600000_steps.zip")
        step_match = re.search(r'_(\d+)_steps', model_name)
        model_steps = int(step_match.group(1)) if step_match else None
        
        # Extract run prefix (everything before _NUMBERS_steps or _final)
        # Format: {algo}_style{N}_{YYYYMMDD}_{HHMMSS}
        run_prefix = None
        if step_match:
            # Get everything before the step count
            run_prefix = model_name[:step_match.start()]
        elif model_name.endswith('_final'):
            # Remove the _final suffix to get the prefix
            run_prefix = model_name[:-6]  # Remove '_final'
        else:
            # Fallback: try to match the pattern {algo}_style{N}_{date}_{time}
            # This handles cases like ppo_lstm_style1_20251226_155516
            pattern_match = re.match(r'(.+_style\d+_\d{8}_\d{6})', model_name)
            if pattern_match:
                run_prefix = pattern_match.group(1)
        
        if run_prefix:
            # Special handling for final models
            if model_name.endswith('_final'):
                # Look for the exact final vecnormalize file first
                final_pattern = os.path.join(model_dir, f"{run_prefix}_vecnormalize_final.pkl")
                if os.path.exists(final_pattern):
                    return final_pattern
                
                # If final vecnormalize doesn't exist, look in checkpoints directory for latest
                parent_dir = os.path.dirname(model_dir)
                checkpoints_dir = os.path.join(parent_dir, 'checkpoints')
                if os.path.exists(checkpoints_dir):
                    pattern = os.path.join(checkpoints_dir, f"{run_prefix}_vecnormalize*.pkl")
                    matches = glob.glob(pattern)
                    if matches:
                        return self._sort_by_steps(matches)[0]
            
            # If we know the step count, look for exact match first
            if model_steps:
                exact_pattern = os.path.join(model_dir, f"{run_prefix}_vecnormalize_{model_steps}_steps.pkl")
                if os.path.exists(exact_pattern):
                    return exact_pattern
                
                # Search in parent directory structure for exact match
                parent_dir = os.path.dirname(model_dir)
                for subdir in ['checkpoints', 'final', '.']:
                    search_dir = os.path.join(parent_dir, subdir) if subdir != '.' else parent_dir
                    exact_pattern = os.path.join(search_dir, f"{run_prefix}_vecnormalize_{model_steps}_steps.pkl")
                    if os.path.exists(exact_pattern):
                        return exact_pattern
            
            # Fallback: find all matching vecnormalize files and pick closest <= model_steps
            pattern = os.path.join(model_dir, f"{run_prefix}_vecnormalize*.pkl")
            matches = glob.glob(pattern)
            if matches and model_steps:
                return self._find_closest_vecnormalize(matches, model_steps)
            elif matches:
                return self._sort_by_steps(matches)[0]
            
            # Search in parent directory structure
            parent_dir = os.path.dirname(model_dir)
            for subdir in ['checkpoints', 'final', '.']:
                search_dir = os.path.join(parent_dir, subdir) if subdir != '.' else parent_dir
                pattern = os.path.join(search_dir, f"{run_prefix}_vecnormalize*.pkl")
                matches = glob.glob(pattern)
                if matches and model_steps:
                    return self._find_closest_vecnormalize(matches, model_steps)
                elif matches:
                    return self._sort_by_steps(matches)[0]
        
        # Fallback: any vecnormalize in same directory
        pattern = os.path.join(model_dir, "*vecnormalize*.pkl")
        matches = glob.glob(pattern)
        if matches and model_steps:
            return self._find_closest_vecnormalize(matches, model_steps)
        return self._sort_by_steps(matches)[0] if matches else None
    
    def _find_closest_vecnormalize(self, file_paths: List[str], target_steps: int) -> str:
        """Find VecNormalize file with step count closest to (and <= if possible) target_steps."""
        import re
        
        def extract_steps(path: str) -> int:
            match = re.search(r'_(\d+)_steps', path)
            return int(match.group(1)) if match else 0
        
        # First, try to find exact match
        for path in file_paths:
            if extract_steps(path) == target_steps:
                return path
        
        # Otherwise, find closest (prefer <= target, but allow > if no <= exists)
        files_with_steps = [(path, extract_steps(path)) for path in file_paths]
        files_lte = [(p, s) for p, s in files_with_steps if s <= target_steps]
        
        if files_lte:
            # Pick highest step count that's <= target
            return max(files_lte, key=lambda x: x[1])[0]
        else:
            # No files <= target, pick the smallest one > target
            return min(files_with_steps, key=lambda x: x[1])[0]
    
    def _sort_by_steps(self, file_paths: List[str]) -> List[str]:
        """Sort file paths by step count (numerically, descending)."""
        import re
        
        def extract_steps(path: str) -> int:
            match = re.search(r'_(\d+)_steps', path)
            return int(match.group(1)) if match else 0
        
        return sorted(file_paths, key=extract_steps, reverse=True)
    
    def _load_model(self, model_path: str, algo: Optional[str] = None):
        """Load model and return (model, algo, is_recurrent)."""
        if not algo:
            algo = self._infer_algo(model_path)
        
        try:
            trainer_class = AlgorithmRegistry.get(algo)
            algo_class = trainer_class.algorithm_class
            model = algo_class.load(model_path, device=self.device)
            is_recurrent = "Lstm" in trainer_class.policy_type
            return model, algo, is_recurrent
        except Exception as e:
            # Try other algorithms
            for other_algo in AlgorithmRegistry.list_algorithms():
                if other_algo == algo:
                    continue
                try:
                    trainer_class = AlgorithmRegistry.get(other_algo)
                    algo_class = trainer_class.algorithm_class
                    model = algo_class.load(model_path, device=self.device)
                    is_recurrent = "Lstm" in trainer_class.policy_type
                    return model, other_algo, is_recurrent
                except:
                    continue
            raise ValueError(f"Failed to load model from {model_path}: {e}")
    
    def _create_env(self, style: int, algo: str, vecnorm_path: Optional[str] = None, 
                    curriculum_stage: Optional[int] = None, num_envs: int = 1):
        """Create environment with VecNormalize if available.
        
        Args:
            style: Control style (1 or 2)
            algo: Algorithm type
            vecnorm_path: Path to VecNormalize stats
            curriculum_stage: If specified, apply curriculum modifiers from this stage.
                              None means no curriculum (full difficulty).
            num_envs: Number of parallel environments to create.
        """
        # Create curriculum manager if stage specified
        curriculum_manager = None
        if curriculum_stage is not None:
            curriculum_manager = CurriculumManager(CurriculumConfig(enabled=True))
            curriculum_manager.current_stage_index = curriculum_stage
            self._log(f"[OK] Using curriculum stage {curriculum_stage}: {curriculum_manager.current_stage.name}")
        
        def make_env(env_id: int):
            """Create a single environment for parallel execution."""
            def _init():
                # Create fresh curriculum manager for each env
                cm = None
                if curriculum_stage is not None:
                    cm = CurriculumManager(CurriculumConfig(enabled=True))
                    cm.current_stage_index = curriculum_stage
                
                if algo == "ppo_dict":
                    return ArenaDictEnv(control_style=style, render_mode=None)
                else:
                    return ArenaEnv(control_style=style, render_mode=None, curriculum_manager=cm)
            return _init
        
        # Create vectorized environment
        if num_envs > 1:
            vec_env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
            self._log(f"[OK] Created {num_envs} parallel environments (SubprocVecEnv)")
        else:
            # Use appropriate environment for algorithm
            if algo == "ppo_dict":
                base_env = ArenaDictEnv(control_style=style, render_mode=None)
            else:
                base_env = ArenaEnv(control_style=style, render_mode=None, 
                                   curriculum_manager=curriculum_manager)
            vec_env = DummyVecEnv([lambda: base_env])
        
        # Load VecNormalize stats if available
        if vecnorm_path and os.path.exists(vecnorm_path):
            env = VecNormalize.load(vecnorm_path, vec_env)
            env.training = False
            env.norm_reward = False
            return env, True
        else:
            return vec_env, False
    
    def run_episode(
        self, 
        model, 
        env, 
        is_recurrent: bool, 
        deterministic: bool = True
    ) -> EpisodeStats:
        """Run a single episode and return statistics."""
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        
        lstm_states = None
        episode_start = np.array([True])
        
        episode_reward = 0
        episode_length = 0
        done = False
        
        # Track episode info
        win = False
        win_step = -1
        phase_reached = 0
        spawners_destroyed = 0
        enemies_destroyed = 0
        final_player_health = 0
        first_spawner_kill_step = -1
        
        while not done:
            # Predict action
            if is_recurrent:
                action, lstm_states = model.predict(
                    obs, 
                    state=lstm_states, 
                    episode_start=episode_start, 
                    deterministic=deterministic
                )
            else:
                action, _ = model.predict(obs, deterministic=deterministic)
            
            # Step environment
            obs, reward, done, info = env.step(action)
            
            # For VecEnv compatibility
            if isinstance(done, np.ndarray):
                done = done[0]
            if isinstance(reward, np.ndarray):
                reward = reward[0]
            if isinstance(info, list):
                info = info[0]
            
            episode_reward += reward
            episode_length += 1
            episode_start = np.array([False])
            
            # Update episode tracking
            if done:
                win = info.get('win', False)
                win_step = info.get('win_step', -1)
                phase_reached = info.get('phase', 0)
                spawners_destroyed = info.get('total_spawners_destroyed', 0)
                enemies_destroyed = info.get('total_enemies_destroyed', 0)
                final_player_health = info.get('player_health', 0)
                first_spawner_kill_step = info.get('first_spawner_kill_step', -1)
        
        return EpisodeStats(
            episode_reward=float(episode_reward),
            episode_length=episode_length,
            win=win,
            win_step=win_step,
            phase_reached=phase_reached,
            spawners_destroyed=spawners_destroyed,
            enemies_destroyed=enemies_destroyed,
            final_player_health=final_player_health,
            first_spawner_kill_step=first_spawner_kill_step
        )
    
    def run_episodes_parallel(
        self,
        model,
        env,
        num_envs: int,
        num_episodes: int,
        is_recurrent: bool,
        deterministic: bool = True
    ) -> List[EpisodeStats]:
        """Run episodes in parallel using vectorized environments.
        
        Args:
            model: The trained model
            env: Vectorized environment (SubprocVecEnv or DummyVecEnv with VecNormalize)
            num_envs: Number of parallel environments
            num_episodes: Total number of episodes to collect
            is_recurrent: Whether the model uses LSTM
            deterministic: Use deterministic policy
            
        Returns:
            List of EpisodeStats for all completed episodes
        """
        completed_episodes = []
        
        # Initialize observations and states
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        
        lstm_states = None
        episode_starts = np.ones(num_envs, dtype=bool)
        
        # Track running episode stats for each environment
        running_rewards = np.zeros(num_envs)
        running_lengths = np.zeros(num_envs, dtype=int)
        
        while len(completed_episodes) < num_episodes:
            # Predict actions for all environments
            if is_recurrent:
                actions, lstm_states = model.predict(
                    obs,
                    state=lstm_states,
                    episode_start=episode_starts,
                    deterministic=deterministic
                )
            else:
                actions, _ = model.predict(obs, deterministic=deterministic)
            
            # Step all environments
            obs, rewards, dones, infos = env.step(actions)
            
            # Update running stats
            running_rewards += rewards
            running_lengths += 1
            episode_starts = dones.copy()
            
            # Collect completed episodes
            for i, (done, info) in enumerate(zip(dones, infos)):
                if done and len(completed_episodes) < num_episodes:
                    episode_stats = EpisodeStats(
                        episode_reward=float(running_rewards[i]),
                        episode_length=int(running_lengths[i]),
                        win=info.get('win', False),
                        win_step=info.get('win_step', -1),
                        phase_reached=info.get('phase', 0),
                        spawners_destroyed=info.get('total_spawners_destroyed', 0),
                        enemies_destroyed=info.get('total_enemies_destroyed', 0),
                        final_player_health=info.get('player_health', 0),
                        first_spawner_kill_step=info.get('first_spawner_kill_step', -1)
                    )
                    completed_episodes.append(episode_stats)
                    
                    # Reset running stats for this env
                    running_rewards[i] = 0
                    running_lengths[i] = 0
                    
                    # Progress indicator
                    if len(completed_episodes) % max(1, num_episodes // 10) == 0:
                        self._log(f"  Progress: {len(completed_episodes)}/{num_episodes} episodes ({len(completed_episodes)/num_episodes*100:.1f}%)")
        
        return completed_episodes
    
    def evaluate_model(
        self,
        model_path: str,
        style: int = 1,
        num_episodes: int = 100,
        deterministic: bool = True,
        algo: Optional[str] = None,
        save_episodes: bool = False,
        curriculum_stage: Optional[int] = None,
        auto_curriculum: bool = False,
        num_workers: int = 1
    ) -> ModelEvaluation:
        """
        Evaluate a model over multiple episodes.
        
        Args:
            model_path: Path to model checkpoint
            style: Control style (1 or 2)
            num_episodes: Number of episodes to evaluate
            deterministic: Use deterministic policy
            algo: Algorithm type (auto-detected if None)
            save_episodes: Whether to include raw episode data in results
            curriculum_stage: Curriculum stage to use (0-5). None means no curriculum (full difficulty).
            auto_curriculum: If True, auto-detect curriculum stage from training state file.
            num_workers: Number of parallel environments (default: 1 = sequential)
        
        Returns:
            ModelEvaluation with comprehensive statistics
        """
        self._log(f"\n{'='*80}")
        self._log(f"Evaluating: {os.path.basename(model_path)}")
        self._log(f"Episodes: {num_episodes} | Style: {style} | Deterministic: {deterministic} | Workers: {num_workers}")
        self._log(f"{'='*80}")
        
        start_time = time.time()
        
        # Load model
        self._log("Loading model...")
        model, algo, is_recurrent = self._load_model(model_path, algo)
        self._log(f"[OK] Loaded {algo} model (Recurrent: {is_recurrent})")
        
        # Auto-detect curriculum stage from training state if requested
        effective_curriculum_stage = curriculum_stage
        if auto_curriculum and curriculum_stage is None:
            training_state = find_training_state(model_path)
            if training_state:
                effective_curriculum_stage = training_state.curriculum_stage_index
                self._log(f"[OK] Auto-detected curriculum stage: {effective_curriculum_stage}")
            else:
                self._log("⚠ No training state found, using full difficulty (no curriculum)")
        
        # Find and load VecNormalize stats
        vecnorm_path = self._find_vecnormalize_stats(model_path)
        if vecnorm_path:
            self._log(f"[OK] Found VecNormalize stats: {os.path.basename(vecnorm_path)}")
        else:
            self._log("⚠ WARNING: No VecNormalize stats found!")
        
        # Create environment with curriculum stage if specified
        env, has_vecnorm = self._create_env(style, algo, vecnorm_path, effective_curriculum_stage, num_workers)
        self._log(f"[OK] Environment created (VecNormalize: {has_vecnorm})")
        
        # Run episodes
        self._log(f"\nRunning {num_episodes} episodes with {num_workers} parallel environments...")
        
        if num_workers > 1:
            # Parallel episode execution
            episodes = self.run_episodes_parallel(
                model, env, num_workers, num_episodes, is_recurrent, deterministic
            )
        else:
            # Sequential episode execution (original behavior)
            episodes = []
            for i in range(num_episodes):
                episode_stats = self.run_episode(model, env, is_recurrent, deterministic)
                episodes.append(episode_stats)
                
                # Progress indicator
                if (i + 1) % max(1, num_episodes // 10) == 0:
                    self._log(f"  Progress: {i+1}/{num_episodes} episodes ({(i+1)/num_episodes*100:.1f}%)")
        
        env.close()
        eval_time = time.time() - start_time
        
        # Compute statistics
        self._log("\nComputing statistics...")
        evaluation = self._compute_statistics(
            episodes, model_path, algo, style, deterministic, eval_time, save_episodes
        )
        
        self._log("[OK] Evaluation complete!")
        return evaluation
    
    def _compute_statistics(
        self,
        episodes: List[EpisodeStats],
        model_path: str,
        algo: str,
        style: int,
        deterministic: bool,
        eval_time: float,
        save_episodes: bool
    ) -> ModelEvaluation:
        """Compute comprehensive statistics from episode data."""
        n = len(episodes)
        
        # Extract arrays for statistics
        rewards = np.array([e.episode_reward for e in episodes])
        lengths = np.array([e.episode_length for e in episodes])
        wins = np.array([e.win for e in episodes])
        spawners = np.array([e.spawners_destroyed for e in episodes])
        enemies = np.array([e.enemies_destroyed for e in episodes])
        phases = np.array([e.phase_reached for e in episodes])
        healths = np.array([e.final_player_health for e in episodes])
        win_steps = np.array([e.win_step for e in episodes if e.win])
        first_spawner_kills = np.array([e.first_spawner_kill_step for e in episodes if e.first_spawner_kill_step > 0])
        
        # Compute distributions
        phase_dist = {int(p): int(np.sum(phases == p)) for p in range(int(phases.max()) + 1)}
        spawner_dist = {int(s): int(np.sum(spawners == s)) for s in range(int(spawners.max()) + 1)}
        
        return ModelEvaluation(
            model_path=model_path,
            model_name=os.path.basename(model_path),
            algorithm=algo,
            style=style,
            deterministic=deterministic,
            total_episodes=n,
            eval_time_seconds=eval_time,
            
            # Win statistics
            win_rate=float(wins.mean()),
            wins=int(wins.sum()),
            
            # Reward statistics
            mean_reward=float(rewards.mean()),
            std_reward=float(rewards.std()),
            median_reward=float(np.median(rewards)),
            min_reward=float(rewards.min()),
            max_reward=float(rewards.max()),
            
            # Length statistics
            mean_length=float(lengths.mean()),
            std_length=float(lengths.std()),
            median_length=float(np.median(lengths)),
            min_length=int(lengths.min()),
            max_length=int(lengths.max()),
            
            # Performance metrics
            mean_spawners_destroyed=float(spawners.mean()),
            std_spawners_destroyed=float(spawners.std()),
            mean_enemies_destroyed=float(enemies.mean()),
            mean_phase_reached=float(phases.mean()),
            mean_final_health=float(healths.mean()),
            
            # Timing metrics
            mean_win_step=float(win_steps.mean()) if len(win_steps) > 0 else -1.0,
            mean_first_spawner_kill=float(first_spawner_kills.mean()) if len(first_spawner_kills) > 0 else -1.0,
            
            # Distributions
            phase_distribution=phase_dist,
            spawner_kill_distribution=spawner_dist,
            
            # Raw data
            episodes=episodes if save_episodes else None
        )


def print_evaluation_summary(eval_result: ModelEvaluation):
    """Print a human-readable summary of evaluation results."""
    print(f"\n{'='*80}")
    print(f"EVALUATION SUMMARY: {eval_result.model_name}")
    print(f"{'='*80}")
    print(f"Algorithm: {eval_result.algorithm} | Style: {eval_result.style} | Deterministic: {eval_result.deterministic}")
    print(f"Episodes: {eval_result.total_episodes} | Eval Time: {eval_result.eval_time_seconds:.2f}s")
    print(f"Speed: {eval_result.total_episodes / eval_result.eval_time_seconds:.1f} episodes/sec")
    
    print(f"\n{'-'*80}")
    print("WIN STATISTICS")
    print(f"{'-'*80}")
    print(f"Win Rate:        {eval_result.win_rate*100:6.2f}% ({eval_result.wins}/{eval_result.total_episodes})")
    if eval_result.mean_win_step > 0:
        print(f"Avg Win Step:    {eval_result.mean_win_step:6.0f} steps")
    
    print(f"\n{'-'*80}")
    print("REWARD STATISTICS")
    print(f"{'-'*80}")
    print(f"Mean:            {eval_result.mean_reward:8.2f} ± {eval_result.std_reward:.2f}")
    print(f"Median:          {eval_result.median_reward:8.2f}")
    print(f"Min/Max:         {eval_result.min_reward:8.2f} / {eval_result.max_reward:.2f}")
    
    print(f"\n{'-'*80}")
    print("EPISODE LENGTH")
    print(f"{'-'*80}")
    print(f"Mean:            {eval_result.mean_length:6.0f} ± {eval_result.std_length:.0f} steps")
    print(f"Median:          {eval_result.median_length:6.0f} steps")
    print(f"Min/Max:         {eval_result.min_length:6d} / {eval_result.max_length:d} steps")
    
    print(f"\n{'-'*80}")
    print("PERFORMANCE METRICS")
    print(f"{'-'*80}")
    print(f"Spawners/Episode:{eval_result.mean_spawners_destroyed:6.2f} ± {eval_result.std_spawners_destroyed:.2f}")
    print(f"Enemies/Episode: {eval_result.mean_enemies_destroyed:6.1f}")
    print(f"Avg Phase:       {eval_result.mean_phase_reached:6.2f}")
    print(f"Avg Final HP:    {eval_result.mean_final_health:6.0f}")
    if eval_result.mean_first_spawner_kill > 0:
        print(f"1st Spawner Kill:{eval_result.mean_first_spawner_kill:6.0f} steps")
    
    print(f"\n{'-'*80}")
    print("PHASE DISTRIBUTION")
    print(f"{'-'*80}")
    for phase in sorted(eval_result.phase_distribution.keys()):
        count = eval_result.phase_distribution[phase]
        pct = count / eval_result.total_episodes * 100
        bar = '#' * int(pct / 2)
        print(f"Phase {phase}:         {count:4d} episodes ({pct:5.1f}%) {bar}")
    
    print(f"\n{'-'*80}")
    print("SPAWNER KILLS DISTRIBUTION")
    print(f"{'-'*80}")
    for kills in sorted(eval_result.spawner_kill_distribution.keys()):
        count = eval_result.spawner_kill_distribution[kills]
        pct = count / eval_result.total_episodes * 100
        bar = '#' * int(pct / 2)
        print(f"{kills:2d} Spawners:    {count:4d} episodes ({pct:5.1f}%) {bar}")
    
    print(f"{'='*80}\n")


def print_comparison_table(evaluations: List[ModelEvaluation]):
    """Print a comparison table of multiple models."""
    print(f"\n{'='*120}")
    print("MODEL COMPARISON")
    print(f"{'='*120}")
    
    # Header
    print(f"{'Model':<50} {'WinRate':>8} {'MeanRwd':>10} {'Spawners':>9} {'Phase':>6} {'Episodes':>9}")
    print(f"{'-'*120}")
    
    # Rows
    for ev in evaluations:
        name = ev.model_name[:48]
        print(f"{name:<50} {ev.win_rate*100:7.2f}% {ev.mean_reward:10.1f} {ev.mean_spawners_destroyed:9.2f} {ev.mean_phase_reached:6.2f} {ev.total_episodes:9d}")
    
    print(f"{'='*120}\n")
    
    # Find best model by different metrics
    print("BEST MODELS BY METRIC:")
    print(f"{'-'*120}")
    
    best_winrate = max(evaluations, key=lambda e: e.win_rate)
    print(f"Best Win Rate:      {best_winrate.model_name} ({best_winrate.win_rate*100:.2f}%)")
    
    best_reward = max(evaluations, key=lambda e: e.mean_reward)
    print(f"Best Mean Reward:   {best_reward.model_name} ({best_reward.mean_reward:.2f})")
    
    best_spawners = max(evaluations, key=lambda e: e.mean_spawners_destroyed)
    print(f"Best Spawner Kills: {best_spawners.model_name} ({best_spawners.mean_spawners_destroyed:.2f})")
    
    best_phase = max(evaluations, key=lambda e: e.mean_phase_reached)
    print(f"Best Phase:         {best_phase.model_name} ({best_phase.mean_phase_reached:.2f})")
    
    print(f"{'='*120}\n")


def save_results(evaluations: List[ModelEvaluation], output_path: str):
    """Save evaluation results to JSON file."""
    data = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'num_models': len(evaluations),
        'evaluations': [asdict(ev) for ev in evaluations]
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"[OK] Results saved to: {output_path}")


def save_results_csv(evaluations: List[ModelEvaluation], output_path: str):
    """Save evaluation results to CSV file in a structured format.
    
    The CSV has models as rows and metrics as columns (transposed format).
    """
    import csv
    
    # Define the metrics to export (in order)
    metrics = [
        ('Algorithm', lambda e: e.algorithm.upper()),
        ('Style', lambda e: str(e.style)),
        ('Deterministic', lambda e: str(e.deterministic)),
        ('Episodes', lambda e: str(e.total_episodes)),
        ('Eval Time (s)', lambda e: f"{e.eval_time_seconds:.2f}"),
        ('Speed (eps/sec)', lambda e: f"{e.total_episodes / e.eval_time_seconds:.1f}"),
        ('Win Rate (%)', lambda e: f"{e.win_rate * 100:.1f}"),
        ('Avg Win Step', lambda e: f"{int(e.mean_win_step)}" if e.mean_win_step > 0 else "N/A"),
        ('Mean Reward', lambda e: f"{e.mean_reward:.2f} ± {e.std_reward:.2f}"),
        ('Median Reward', lambda e: f"{e.median_reward:.2f}"),
        ('Reward Min', lambda e: f"{e.min_reward:.2f}"),
        ('Reward Max', lambda e: f"{e.max_reward:.2f}"),
        ('Mean Episode Length', lambda e: f"{int(e.mean_length)} ± {int(e.std_length)}"),
        ('Median Episode Length', lambda e: f"{int(e.median_length)}"),
        ('Episode Length Min', lambda e: str(e.min_length)),
        ('Episode Length Max', lambda e: str(e.max_length)),
        ('Spawners / Episode', lambda e: f"{e.mean_spawners_destroyed:.2f} ± {e.std_spawners_destroyed:.2f}"),
        ('Enemies / Episode', lambda e: f"{e.mean_enemies_destroyed:.1f}"),
        ('Avg Phase Reached', lambda e: f"{e.mean_phase_reached:.2f}"),
        ('Avg Final HP', lambda e: f"{int(e.mean_final_health)}"),
        ('First Spawner Kill (steps)', lambda e: f"{int(e.mean_first_spawner_kill)}" if e.mean_first_spawner_kill > 0 else "N/A"),
    ]
    
    # Add phase distribution metrics
    max_phase = max(max(e.phase_distribution.keys()) for e in evaluations) if evaluations else 0
    for phase in range(1, max_phase + 1):
        metrics.append((
            f'Phase {phase} (%)',
            lambda e, p=phase: f"{e.phase_distribution.get(p, 0) / e.total_episodes * 100:.1f}"
        ))
    
    # Add spawner kill distribution metrics
    max_spawners = max(max(e.spawner_kill_distribution.keys()) for e in evaluations) if evaluations else 0
    for kills in range(max_spawners + 1):
        if any(e.spawner_kill_distribution.get(kills, 0) > 0 for e in evaluations):
            metrics.append((
                f'{kills} Spawners Killed (%)',
                lambda e, k=kills: f"{e.spawner_kill_distribution.get(k, 0) / e.total_episodes * 100:.1f}"
            ))
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header row with metric names (transposed: models as rows, metrics as columns)
        header = ['Model'] + [metric_name for metric_name, _ in metrics]
        writer.writerow(header)
        
        # Write each model as a row
        for ev in evaluations:
            row = [ev.model_name] + [metric_fn(ev) for _, metric_fn in metrics]
            writer.writerow(row)
    
    print(f"[OK] CSV results saved to: {output_path}")


def find_models_in_directory(directory: str, pattern: str = "*.zip") -> List[str]:
    """Recursively find model files in directory."""
    path = Path(directory)
    models = list(path.rglob(pattern))
    return [str(m) for m in models]


def main():
    parser = argparse.ArgumentParser(
        description="Headless evaluation script for Deep RL Arena models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Model selection
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument('--model', type=str, help='Path to single model file')
    model_group.add_argument('--models', nargs='+', help='Paths to multiple model files')
    model_group.add_argument('--directory', type=str, help='Directory to search for models')
    
    # Evaluation parameters
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes per model (default: 100)')
    parser.add_argument('--style', type=int, default=None, choices=[1, 2], help='Control style (auto-detected from model path if not specified)')
    parser.add_argument('--stochastic', action='store_true', help='Use stochastic policy (default: deterministic)')
    parser.add_argument('--algo', type=str, default=None, help='Algorithm type (auto-detected if not specified)')
    
    # Output options
    parser.add_argument('--output', type=str, default=None, help='Output JSON file path')
    parser.add_argument('--csv', type=str, default=None, help='Output CSV file path for tabular results')
    parser.add_argument('--save-episodes', action='store_true', help='Save raw episode data in output')
    parser.add_argument('--quiet', action='store_true', help='Suppress progress messages')
    parser.add_argument('--compare', action='store_true', help='Print comparison table for multiple models')
    
    # Performance options
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto/cpu/cuda)')
    parser.add_argument('--workers', type=int, default=1, help='Number of parallel workers (default: 1)')
    
    # Curriculum options
    parser.add_argument('--curriculum-stage', type=int, default=None, choices=[0, 1, 2, 3, 4, 5],
                       help='Curriculum stage (0-5). None=full difficulty. Auto-detect with --auto-curriculum.')
    parser.add_argument('--auto-curriculum', action='store_true',
                       help='Auto-detect curriculum stage from training state file')
    
    args = parser.parse_args()
    
    # Collect model paths
    if args.model:
        model_paths = [args.model]
    elif args.models:
        model_paths = args.models
    else:  # args.directory
        model_paths = find_models_in_directory(args.directory)
        if not model_paths:
            print(f"Error: No models found in {args.directory}")
            return
        print(f"Found {len(model_paths)} models in {args.directory}")
    
    # Validate model paths
    model_paths = [p for p in model_paths if os.path.exists(p)]
    if not model_paths:
        print("Error: No valid model paths provided")
        return
    
    # Create evaluator
    evaluator = HeadlessEvaluator(device=args.device, verbose=not args.quiet)
    
    # Run evaluations
    evaluations = []
    
    if args.workers > 1 and len(model_paths) > 1:
        # When evaluating multiple models with parallel workers, we use the workers
        # for parallel episodes within each model (not parallel models)
        print(f"Using {args.workers} parallel environments per model for evaluation...")
    
    for model_path in model_paths:
        try:
            # Auto-detect style from model path if not specified
            style = args.style
            if style is None:
                style = evaluator._infer_style(model_path)
                if not args.quiet:
                    print(f"Auto-detected style: {style} (from model path)")
            
            eval_result = evaluator.evaluate_model(
                model_path=model_path,
                style=style,
                num_episodes=args.episodes,
                deterministic=not args.stochastic,
                algo=args.algo,
                save_episodes=args.save_episodes,
                curriculum_stage=args.curriculum_stage,
                auto_curriculum=args.auto_curriculum,
                num_workers=args.workers
            )
            evaluations.append(eval_result)
            
            # Print summary after each model
            if not args.quiet:
                print_evaluation_summary(eval_result)
        
        except Exception as e:
            print(f"Error evaluating {model_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print comparison if multiple models
    if len(evaluations) > 1 and (args.compare or not args.quiet):
        print_comparison_table(evaluations)
    
    # Save results if requested
    if args.output:
        save_results(evaluations, args.output)
    
    # Save CSV results if requested
    if args.csv:
        save_results_csv(evaluations, args.csv)
    
    print(f"\n[OK] Evaluation complete! Evaluated {len(evaluations)} model(s) with {args.episodes} episodes each.")


if __name__ == "__main__":
    main()

