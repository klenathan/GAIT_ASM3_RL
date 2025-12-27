"""
Model evaluation logic for Deep RL Arena.
"""

from arena.training.algorithms import dqn, ppo, ppo_lstm, a2c
from arena.training.registry import AlgorithmRegistry
from arena.ui.model_output import ModelOutputExtractor
from arena.ui.menu import Menu
from arena.ui.renderer import ArenaRenderer
from arena.game.human_controller import HumanController
from arena.core.environment_dict import ArenaDictEnv
from arena.core.environment import ArenaEnv
from arena.core import config
from arena.core.device import DeviceManager
from arena.core.config import TrainerConfig
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import torch
import numpy as np
import pygame
import glob
import os
import warnings
# Suppress pkg_resources deprecation warning from pygame
warnings.filterwarnings(
    "ignore", message=".*pkg_resources is deprecated.*", category=UserWarning)


class Evaluator:
    """Handles model evaluation with visualization."""

    def __init__(self, requested_device: str = "auto"):
        self.device = DeviceManager.get_device(requested_device)
        self.renderer = None
        self.menu = None
        self.env = None
        self.model = None
        self.is_recurrent = False
        self.output_extractor = ModelOutputExtractor()
        self.current_algo = None

    def setup_ui(self):
        """Initialize renderer and menu."""
        self.renderer = ArenaRenderer()
        self.menu = Menu(self.renderer.screen)

    def _infer_algo(self, model_path: str) -> str:
        """Infer algorithm type from filename."""
        name = os.path.basename(model_path).lower()
        # Sort by length descending to match longest name first (e.g. ppo_lstm before ppo)
        algos = sorted(AlgorithmRegistry.list_algorithms(),
                       key=len, reverse=True)
        for algo in algos:
            if algo in name:
                return algo
        return "ppo"  # Default fallback

    def load_model(self, model_path: str, algo: str = None):
        """Load a trained model with automatic algorithm detection."""
        if not algo:
            algo = self._infer_algo(model_path)

        print(f"Loading model: {model_path} (Algo: {algo})")

        try:
            trainer_class = AlgorithmRegistry.get(algo)
            algo_class = trainer_class.algorithm_class
            self.model = algo_class.load(model_path, device=self.device)
            self.is_recurrent = "Lstm" in trainer_class.policy_type
            self.current_algo = algo
            return True
        except Exception as e:
            print(f"Failed to load as {algo}: {e}")
            # Try other loaders
            for other_algo in AlgorithmRegistry.list_algorithms():
                if other_algo == algo:
                    continue
                try:
                    trainer_class = AlgorithmRegistry.get(other_algo)
                    algo_class = trainer_class.algorithm_class
                    self.model = algo_class.load(
                        model_path, device=self.device)
                    self.is_recurrent = "Lstm" in trainer_class.policy_type
                    self.current_algo = other_algo
                    print(f"Successfully loaded as {other_algo}")
                    return True
                except:
                    continue
            return False

    def _find_vecnormalize_stats(self, model_path: str) -> str:
        """Find VecNormalize stats file matching the model."""
        import glob
        import re

        model_dir = os.path.dirname(model_path)
        model_name = os.path.basename(model_path).replace('.zip', '')

        # Extract run prefix based on model name format
        # Format: {algo}_style{N}_{YYYYMMDD}_{HHMMSS}[_NUMBERS_steps or _final]
        run_prefix = None

        # Try to extract step count to get prefix before it
        step_match = re.search(r'_(\d+)_steps', model_name)
        if step_match:
            run_prefix = model_name[:step_match.start()]
        elif model_name.endswith('_final'):
            # Remove the _final suffix to get the prefix
            run_prefix = model_name[:-6]  # Remove '_final'
        else:
            # Fallback: try to match the pattern {algo}_style{N}_{date}_{time}
            pattern_match = re.match(r'(.+_style\d+_\d{8}_\d{6})', model_name)
            if pattern_match:
                run_prefix = pattern_match.group(1)

        if run_prefix:
            # Special handling for final models
            if model_name.endswith('_final'):
                # Look for the exact final vecnormalize file first
                final_pattern = os.path.join(
                    model_dir, f"{run_prefix}_vecnormalize_final.pkl")
                if os.path.exists(final_pattern):
                    return final_pattern

                # If final vecnormalize doesn't exist, look in checkpoints directory for latest
                parent_dir = os.path.dirname(model_dir)
                checkpoints_dir = os.path.join(parent_dir, 'checkpoints')
                if os.path.exists(checkpoints_dir):
                    pattern = os.path.join(
                        checkpoints_dir, f"{run_prefix}_vecnormalize*.pkl")
                    matches = sorted(glob.glob(pattern), reverse=True)
                    if matches:
                        return matches[0]

            pattern = os.path.join(
                model_dir, f"{run_prefix}_vecnormalize*.pkl")
            matches = sorted(glob.glob(pattern), reverse=True)  # Latest first
            if matches:
                return matches[0]

            # Try in parent directory (for new unified structure where checkpoints/ and final/ are separate)
            parent_dir = os.path.dirname(model_dir)
            for subdir in ['checkpoints', 'final', '.']:
                search_dir = os.path.join(
                    parent_dir, subdir) if subdir != '.' else parent_dir
                pattern = os.path.join(
                    search_dir, f"{run_prefix}_vecnormalize*.pkl")
                matches = sorted(glob.glob(pattern), reverse=True)
                if matches:
                    return matches[0]

        # Fallback: find any vecnormalize file in the same directory
        pattern = os.path.join(model_dir, "*vecnormalize*.pkl")
        matches = sorted(glob.glob(pattern), reverse=True)
        return matches[0] if matches else None

    def run_session(self, model_path: str, style: int, deterministic: bool = True):
        """Run a single evaluation session."""
        if not self.load_model(model_path):
            return "menu"

        if self.env:
            self.env.close()

        # Create base environment - use ArenaDictEnv for ppo_dict, ArenaEnv for others
        if self.current_algo == "ppo_dict":
            base_env = ArenaDictEnv(control_style=style, render_mode=None)
        else:
            base_env = ArenaEnv(control_style=style, render_mode=None)
        base_env.render_mode = "human"
        base_env.renderer = self.renderer
        base_env._owns_renderer = False

        # Wrap in DummyVecEnv for VecNormalize compatibility
        vec_env = DummyVecEnv([lambda: base_env])

        # Try to load VecNormalize stats
        vecnorm_path = self._find_vecnormalize_stats(model_path)
        if vecnorm_path and os.path.exists(vecnorm_path):
            print(
                f"✓ Loading VecNormalize stats from: {os.path.basename(vecnorm_path)}")
            self.env = VecNormalize.load(vecnorm_path, vec_env)
            self.env.training = False  # Disable stats updates during eval
            self.env.norm_reward = False  # Don't normalize rewards during eval
            print(
                "✓ VecNormalize stats applied successfully (observations will be normalized)")
        else:
            print("⚠ WARNING: No VecNormalize stats found!")
            print("⚠ Model will receive raw observations (may cause poor performance)")
            print("⚠ This is expected for old models but indicates a bug for new models")
            self.env = vec_env

        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]  # Handle new gym API

        # Initialize LSTM states for recurrent models
        # RecurrentPPO expects lstm_states=None initially, which triggers initialization
        lstm_states = None
        # episode_start flag: True at episode start, False otherwise
        # VecEnv expects array matching num_envs (1 in our case)
        episode_start = np.array([True])

        running = True
        while running:
            # Handle Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return "quit"
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return "menu"
                    # UI Toggles
                    if event.key == pygame.K_h:
                        self.renderer.show_health = not self.renderer.show_health
                    if event.key == pygame.K_v:
                        self.renderer.show_vision = not self.renderer.show_vision
                    if event.key == pygame.K_d:
                        self.renderer.show_debug = not self.renderer.show_debug

            # Save LSTM states before predict (for proper extraction)
            lstm_states_for_extraction = lstm_states if self.is_recurrent else None

            # Predict action (obs is already an array from VecEnv)
            if self.is_recurrent:
                # RecurrentPPO: pass lstm_states and episode_start flag
                # lstm_states=None triggers initialization on first call or after reset
                action, lstm_states = self.model.predict(
                    obs, state=lstm_states, episode_start=episode_start, deterministic=deterministic
                )
            else:
                # Non-recurrent: standard prediction
                action, _ = self.model.predict(
                    obs, deterministic=deterministic)

            # Get scalar action for visualization (VecEnv returns arrays)
            action_scalar = action[0] if isinstance(
                action, np.ndarray) else action

            # Extract observation for single env (VecEnv returns arrays/dicts with batch dimension)
            if isinstance(obs, dict):
                # Dict observation: extract first element from each dict value
                obs_single = {key: value[0] if isinstance(value, np.ndarray) else value
                              for key, value in obs.items()}
            else:
                # Array observation: extract first element
                obs_single = obs[0]

            # Extract and visualize model output (use first element for single env)
            # For recurrent models, use lstm_states BEFORE predict for proper extraction
            # This ensures we extract with the same states that were used during predict
            output = self.output_extractor.extract(
                self.model, obs_single, action_scalar,
                lstm_states=lstm_states_for_extraction if self.is_recurrent else None,
                episode_start=episode_start if self.is_recurrent else None
            )
            self.renderer.set_model_output(output, style)

            # After first step, episode_start is False (unless episode resets)
            episode_start = np.array([False])

            # Step (VecEnv API: returns arrays, uses 'dones' not terminated/truncated)
            obs, rewards, dones, infos = self.env.step(action)

            # Handle episode reset for recurrent models
            if dones[0]:
                obs = self.env.reset()
                if isinstance(obs, tuple):
                    obs = obs[0]
                # Reset LSTM states on episode boundary
                # Setting to None will trigger re-initialization on next predict()
                if self.is_recurrent:
                    lstm_states = None
                    episode_start = np.array([True])

            # Render - get underlying env from VecEnv wrapper
            if hasattr(self.env, 'envs'):
                self.env.envs[0].render()
            elif hasattr(self.env, 'venv') and hasattr(self.env.venv, 'envs'):
                self.env.venv.envs[0].render()

        return "menu"

    def run_human_session(self, style: int):
        """Run a manual gameplay session."""
        if self.env:
            self.env.close()

        # Create base environment
        self.env = ArenaEnv(control_style=style, render_mode="human")
        self.env.renderer = self.renderer
        self.env._owns_renderer = False

        controller = HumanController(style=style)

        # Create a mock metrics object for the renderer
        metrics = {
            'episode': 1,
            'episode_reward': 0.0,
            'total_reward': 0.0,
            'is_human': True
        }

        obs, _ = self.env.reset()
        running = True

        while running:
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    return "quit"
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return "menu"
                    # UI Toggles
                    if event.key == pygame.K_h:
                        self.renderer.show_health = not self.renderer.show_health
                    if event.key == pygame.K_v:
                        self.renderer.show_vision = not self.renderer.show_vision
                    if event.key == pygame.K_d:
                        self.renderer.show_debug = not self.renderer.show_debug

            # Get action from human controller
            action = controller.get_action(events)

            # Step environment
            obs, reward, terminated, truncated, info = self.env.step(action)
            metrics['episode_reward'] += reward
            metrics['total_reward'] += reward

            # Update renderer info (no model output)
            self.renderer.set_model_output(None, style)

            if terminated or truncated:
                obs, _ = self.env.reset()
                metrics['episode'] += 1
                metrics['episode_reward'] = 0.0

            # Render
            self.renderer.render(self.env, training_metrics=metrics)

        return "menu"

    def main_loop(self):
        """Main interactive loop with menu."""
        self.setup_ui()
        state = "menu"

        try:
            while True:
                if state == "menu":
                    events = pygame.event.get()
                    for event in events:
                        if event.type == pygame.QUIT:
                            return

                    action = self.menu.update(events)
                    self.renderer.render_menu(self.menu)

                    if action == "start":
                        selection = self.menu.get_selection()
                        if selection:
                            if selection["mode"] == "Human Player":
                                state = self.run_human_session(
                                    selection["style"])
                            else:
                                state = self.run_session(
                                    selection["model"], selection["style"], selection["deterministic"]
                                )
                        else:
                            if self.menu.gameplay_modes[self.menu.selected_mode_idx] == "Model":
                                self.menu.set_status(
                                    "No models available for selection", "warning")
                            else:
                                self.menu.set_status(
                                    "No selection made", "warning")

                    if action == "quit":
                        break

                elif state == "quit":
                    break
        finally:
            if self.env:
                self.env.close()
            self.renderer.close()
