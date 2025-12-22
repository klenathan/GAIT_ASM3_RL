"""
Model evaluation logic for Deep RL Arena.
"""

import os
import pygame
import numpy as np
import torch
from stable_baselines3 import DQN, PPO
from sb3_contrib import RecurrentPPO

from arena.core.config import TrainerConfig
from arena.core.device import DeviceManager
from arena.core import config
from arena.environment import ArenaEnv
from arena.ui.renderer import ArenaRenderer
from arena.ui.menu import Menu

class Evaluator:
    """Handles model evaluation with visualization."""
    
    def __init__(self, requested_device: str = "auto"):
        self.device = DeviceManager.get_device(requested_device)
        self.renderer = None
        self.menu = None
        self.env = None
        self.model = None
        self.is_recurrent = False
        
    def setup_ui(self):
        """Initialize renderer and menu."""
        self.renderer = ArenaRenderer()
        self.menu = Menu(self.renderer.screen)
        
    def _infer_algo(self, model_path: str) -> str:
        """Infer algorithm type from filename."""
        name = os.path.basename(model_path).lower()
        if "ppo_lstm" in name: return "ppo_lstm"
        if "ppo" in name: return "ppo"
        if "dqn" in name: return "dqn"
        return "ppo" # Default fallback

    def load_model(self, model_path: str, algo: str = None):
        """Load a trained model with automatic algorithm detection."""
        if not algo:
            algo = self._infer_algo(model_path)
            
        print(f"Loading model: {model_path} (Algo: {algo})")
        
        algo_class = PPO if algo == "ppo" else (RecurrentPPO if algo == "ppo_lstm" else DQN)
        self.is_recurrent = algo == "ppo_lstm"
        
        try:
            self.model = algo_class.load(model_path, device=self.device)
            return True
        except Exception as e:
            print(f"Failed to load as {algo}: {e}")
            # Try other loaders
            for other_algo in ["ppo", "ppo_lstm", "dqn"]:
                if other_algo == algo: continue
                try:
                    other_class = PPO if other_algo == "ppo" else (RecurrentPPO if other_algo == "ppo_lstm" else DQN)
                    self.model = other_class.load(model_path, device=self.device)
                    self.is_recurrent = other_algo == "ppo_lstm"
                    print(f"Successfully loaded as {other_algo}")
                    return True
                except: continue
            return False

    def run_session(self, model_path: str, style: int, deterministic: bool = True):
        """Run a single evaluation session."""
        if not self.load_model(model_path):
            return "menu"
            
        if self.env: self.env.close()
        self.env = ArenaEnv(control_style=style, render_mode=None)
        self.env.render_mode = "human"
        self.env.renderer = self.renderer
        self.env._owns_renderer = False
        
        obs, info = self.env.reset()
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
            
            # Predict action
            if self.is_recurrent:
                action, lstm_states = self.model.predict(
                    obs, state=lstm_states, episode_start=episode_start, deterministic=deterministic
                )
                episode_start = np.array([False])
            else:
                action, _ = self.model.predict(obs, deterministic=deterministic)
            
            # Step
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            if terminated or truncated:
                obs, info = self.env.reset()
                lstm_states = None
                episode_start = np.array([True])
            
            # Render
            self.env.render()
            
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
