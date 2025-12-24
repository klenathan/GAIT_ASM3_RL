"""
Model evaluation logic for Deep RL Arena.
"""

import os
import glob
import pygame
import numpy as np
import torch

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from arena.core.config import TrainerConfig
from arena.core.device import DeviceManager
from arena.core import config
from arena.core.environment import ArenaEnv
from arena.ui.renderer import ArenaRenderer
from arena.ui.menu import Menu
from arena.ui.model_output import ModelOutputExtractor
from arena.training.registry import AlgorithmRegistry
from arena.training.algorithms import dqn, ppo, ppo_lstm, a2c

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
        
    def setup_ui(self):
        """Initialize renderer and menu."""
        self.renderer = ArenaRenderer()
        self.menu = Menu(self.renderer.screen)
        
    def _infer_algo(self, model_path: str) -> str:
        """Infer algorithm type from filename."""
        name = os.path.basename(model_path).lower()
        # Sort by length descending to match longest name first (e.g. ppo_lstm before ppo)
        algos = sorted(AlgorithmRegistry.list_algorithms(), key=len, reverse=True)
        for algo in algos:
            if algo in name:
                return algo
        return "ppo" # Default fallback

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
            return True
        except Exception as e:
            print(f"Failed to load as {algo}: {e}")
            # Try other loaders
            for other_algo in AlgorithmRegistry.list_algorithms():
                if other_algo == algo: continue
                try:
                    trainer_class = AlgorithmRegistry.get(other_algo)
                    algo_class = trainer_class.algorithm_class
                    self.model = algo_class.load(model_path, device=self.device)
                    self.is_recurrent = "Lstm" in trainer_class.policy_type
                    print(f"Successfully loaded as {other_algo}")
                    return True
                except: continue
            return False

    def _find_vecnormalize_stats(self, model_path: str) -> str:
        """Find VecNormalize stats file matching the model."""
        model_dir = os.path.dirname(model_path)
        model_name = os.path.basename(model_path).replace('.zip', '')
        
        # Extract run prefix (e.g., ppo_style2_20251222_154509)
        parts = model_name.split('_')
        if len(parts) >= 4:
            # Try to find matching vecnormalize file
            run_prefix = '_'.join(parts[:4])  # algo_styleX_YYYYMMDD_HHMMSS
            pattern = os.path.join(model_dir, f"{run_prefix}_vecnormalize*.pkl")
            matches = sorted(glob.glob(pattern), reverse=True)  # Latest first
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
            
        if self.env: self.env.close()
        
        # Create base environment
        base_env = ArenaEnv(control_style=style, render_mode=None)
        base_env.render_mode = "human"
        base_env.renderer = self.renderer
        base_env._owns_renderer = False
        
        # Wrap in DummyVecEnv for VecNormalize compatibility
        vec_env = DummyVecEnv([lambda: base_env])
        
        # Try to load VecNormalize stats
        vecnorm_path = self._find_vecnormalize_stats(model_path)
        if vecnorm_path and os.path.exists(vecnorm_path):
            print(f"Loading VecNormalize stats: {vecnorm_path}")
            self.env = VecNormalize.load(vecnorm_path, vec_env)
            self.env.training = False  # Disable stats updates during eval
            self.env.norm_reward = False  # Don't normalize rewards during eval
        else:
            print("No VecNormalize stats found, using raw observations")
            self.env = vec_env
        
        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]  # Handle new gym API
        lstm_states = None
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
            
            # Predict action (obs is already an array from VecEnv)
            if self.is_recurrent:
                action, lstm_states = self.model.predict(
                    obs, state=lstm_states, episode_start=episode_start, deterministic=deterministic
                )
            else:
                action, _ = self.model.predict(obs, deterministic=deterministic)
            
            # Get scalar action for visualization (VecEnv returns arrays)
            action_scalar = action[0] if isinstance(action, np.ndarray) else action
                
            # Extract and visualize model output (use first element for single env)
            output = self.output_extractor.extract(
                self.model, obs[0], action_scalar, 
                lstm_states=lstm_states, 
                episode_start=episode_start
            )
            self.renderer.set_model_output(output, style)
            
            episode_start = np.array([False])
            
            # Step (VecEnv API: returns arrays, uses 'dones' not terminated/truncated)
            obs, rewards, dones, infos = self.env.step(action)
            
            if dones[0]:
                obs = self.env.reset()
                if isinstance(obs, tuple):
                    obs = obs[0]
                lstm_states = None
                episode_start = np.array([True])
            
            # Render - get underlying env from VecEnv wrapper
            if hasattr(self.env, 'envs'):
                self.env.envs[0].render()
            elif hasattr(self.env, 'venv') and hasattr(self.env.venv, 'envs'):
                self.env.venv.envs[0].render()
            
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
                        if event.type == pygame.QUIT: return
                        
                    action = self.menu.update(events)
                    self.renderer.render_menu(self.menu)
                    
                    if action == "start":
                        selection = self.menu.get_selection()
                        if selection:
                            state = self.run_session(
                                selection["model"], selection["style"], selection["deterministic"]
                            )
                        else:
                            self.menu.set_status("No model selected", "warning")
                            
                    if action == "quit": break
                    
                elif state == "quit":
                    break
        finally:
            if self.env: self.env.close()
            self.renderer.close()
