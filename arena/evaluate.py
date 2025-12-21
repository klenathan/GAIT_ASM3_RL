import argparse
import time
import pygame
import sys
import os
import numpy as np
from stable_baselines3 import DQN, PPO

from arena.environment import ArenaEnv
from arena.renderer import ArenaRenderer
from arena.menu import Menu
from arena import config


def _infer_algo_from_model_path(model_path: str) -> str | None:
    """Infer algo from filename convention like 'ppo_*.zip' or 'dqn_*.zip'."""
    name = os.path.basename(model_path).lower()
    if name.startswith("ppo_"):
        return "ppo"
    if name.startswith("dqn_"):
        return "dqn"
    return None


def _pygame_display_ready() -> bool:
    """Return True if pygame and a display surface are initialized."""
    try:
        return (
            pygame.get_init()
            and pygame.display.get_init()
            and pygame.display.get_surface() is not None
        )
    except Exception:
        return False


def run_evaluation(model, env, renderer, deterministic=True):
    """Run an indefinite evaluation session"""
    print(f"\nStarting evaluation with model: {model.path if hasattr(model, 'path') else 'Loaded Model'}")
    
    obs, info = env.reset()
    done = False
    
    running = True
    while running:
        # If the pygame window was closed or pygame was quit, exit cleanly.
        if not _pygame_display_ready():
            return "quit"

        # Handle events
        try:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return "quit"
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return "menu"
                    # QoL Debug keys
                    if event.key == pygame.K_h:
                        renderer.show_health = not renderer.show_health
                    if event.key == pygame.K_v:
                        renderer.show_vision = not renderer.show_vision
                    if event.key == pygame.K_RIGHTBRACKET:
                        config.FPS = min(240, config.FPS + 10)
                    if event.key == pygame.K_LEFTBRACKET:
                        config.FPS = max(10, config.FPS - 10)
        except pygame.error:
            # Common when the display is closed or pygame was quit elsewhere.
            return "quit"

        # Get action from model
        action, _states = model.predict(obs, deterministic=deterministic)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # If dead, reset automatically after a short delay or immediately
        if terminated or truncated:
            # Show final stats briefly or just reset
            obs, info = env.reset()
        
        # Render
        try:
            env.render()
        except pygame.error:
            return "quit"
        
    return "menu"


def main():
    parser = argparse.ArgumentParser(description="Deep RL Arena - Interactive Evaluation")
    parser.add_argument("--model", type=str, help="Initial model path (optional)")
    parser.add_argument("--style", type=int, choices=[1, 2], help="Initial control style")
    parser.add_argument("--algo", type=str, choices=["dqn", "ppo"], help="Initial algorithm")
    args = parser.parse_args()

    # Initialize Pygame and Renderer
    pygame.init()
    renderer = ArenaRenderer()
    menu = Menu(renderer.screen)
    
    # Pre-select if args provided
    if args.model:
        # Try to find model in list by name
        model_name = args.model.split('/')[-1].replace('.zip', '')
        for i, m in enumerate(menu.models):
            if m["name"] == model_name:
                menu.selected_model_idx = i
                break
    
    # Note: algo and style selection is now automatic from model metadata
    # The sidebar toggles remain for backward compatibility but don't affect model selection
    if args.algo:
        if args.algo in menu.algos:
            menu.selected_algo_idx = menu.algos.index(args.algo)
            
    if args.style:
        if args.style in menu.styles:
            menu.selected_style_idx = menu.styles.index(args.style)

    state = "menu"
    current_env = None
    current_model = None

    try:
        while True:
            if state == "menu":
                if not _pygame_display_ready():
                    break

                try:
                    events = pygame.event.get()
                except pygame.error:
                    break

                for event in events:
                    if event.type == pygame.QUIT:
                        # Exit cleanly (let the finally block close pygame once).
                        return

                menu_action = menu.update(events)

                try:
                    renderer.render_menu(menu)
                except pygame.error:
                    break

                if menu_action == "start":
                    selection = menu.get_selection()
                    if selection:
                        # Show loading status
                        menu.set_status("Loading model...", "loading", duration=None)
                        try:
                            renderer.render_menu(menu)
                        except pygame.error:
                            pass
                        
                        # Validate model file exists
                        if not os.path.exists(selection["model"]):
                            menu.set_status(f"Error: Model file not found: {selection['model']}", 
                                          "error", duration=5000)
                            continue
                        
                        # Load model
                        print(f"Loading {selection['model']}...")
                        inferred_algo = _infer_algo_from_model_path(selection["model"])
                        if inferred_algo and inferred_algo != selection["algo"]:
                            print(
                                f"Note: model filename suggests '{inferred_algo.upper()}', "
                                f"overriding selected algo '{selection['algo'].upper()}' to avoid load errors."
                            )
                            selection["algo"] = inferred_algo
                            if inferred_algo in menu.algos:
                                menu.selected_algo_idx = menu.algos.index(inferred_algo)

                        algo_class = PPO if selection["algo"] == "ppo" else DQN
                        try:
                            try:
                                current_model = algo_class.load(selection["model"])
                                menu.set_status("Model loaded successfully!", "success", duration=2000)
                            except Exception as e:
                                # Common failure mode: user selects DQN but picks a PPO model (or vice versa).
                                other_algo = "dqn" if selection["algo"] == "ppo" else "ppo"
                                other_class = DQN if other_algo == "dqn" else PPO
                                print(
                                    f"Load failed with {selection['algo'].upper()} ({e}). "
                                    f"Trying {other_algo.upper()} loader..."
                                )
                                menu.set_status(f"Retrying with {other_algo.upper()}...", "loading", duration=None)
                                try:
                                    renderer.render_menu(menu)
                                except pygame.error:
                                    pass
                                
                                current_model = other_class.load(selection["model"])
                                selection["algo"] = other_algo
                                if other_algo in menu.algos:
                                    menu.selected_algo_idx = menu.algos.index(other_algo)
                                menu.set_status(f"Model loaded as {other_algo.upper()}!", "success", duration=2000)

                            # Create/Update environment (only after we successfully loaded a model)
                            if current_env:
                                current_env.close()
                            # Important: don't let the env create/own its own renderer here.
                            # We want a single shared pygame window (the menu window).
                            current_env = ArenaEnv(control_style=selection["style"], render_mode=None)
                            current_env.render_mode = "human"
                            current_env.renderer = renderer
                            # Ensure env.close() won't quit pygame (which would crash the menu/eval loop).
                            current_env._owns_renderer = False

                            state = "evaluating"
                        except Exception as e:
                            error_msg = str(e)
                            # Provide more helpful error messages
                            if "No such file" in error_msg or "FileNotFoundError" in error_msg:
                                menu.set_status("Error: Model file not found", "error", duration=5000)
                            elif "corrupted" in error_msg.lower() or "invalid" in error_msg.lower():
                                menu.set_status("Error: Model file may be corrupted", "error", duration=5000)
                            elif "algorithm" in error_msg.lower():
                                menu.set_status("Error: Algorithm mismatch. Try different algo.", "error", duration=5000)
                            else:
                                # Truncate long error messages
                                short_error = error_msg[:80] + "..." if len(error_msg) > 80 else error_msg
                                menu.set_status(f"Error: {short_error}", "error", duration=6000)
                            print(f"Error loading model: {e}")
                    else:
                        menu.set_status("No model selected", "warning", duration=3000)

                if menu_action == "quit":
                    break

            elif state == "evaluating":
                result = run_evaluation(current_model, current_env, renderer, menu.deterministic)
                if result == "menu":
                    state = "menu"
                elif result == "quit":
                    break
    finally:
        if current_env:
            current_env.close()
        renderer.close()


if __name__ == "__main__":
    print("Starting evaluation...")
    main()
