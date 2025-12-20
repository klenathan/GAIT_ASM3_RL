import argparse
import time
import pygame
import sys
import numpy as np
from stable_baselines3 import DQN, PPO

from arena.environment import ArenaEnv
from arena.renderer import ArenaRenderer
from arena.menu import Menu
from arena import config


def run_evaluation(model, env, renderer, deterministic=True):
    """Run an indefinite evaluation session"""
    print(f"\nStarting evaluation with model: {model.path if hasattr(model, 'path') else 'Loaded Model'}")
    
    obs, info = env.reset()
    done = False
    
    running = True
    while running:
        # Handle events
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

        # Get action from model
        action, _states = model.predict(obs, deterministic=deterministic)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # If dead, reset automatically after a short delay or immediately
        if terminated or truncated:
            # Show final stats briefly or just reset
            obs, info = env.reset()
        
        # Render
        env.render()
        
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
        # Try to find model in list
        model_name = args.model.split('/')[-1].replace('.zip', '')
        if model_name in menu.models:
            menu.selected_model_idx = menu.models.index(model_name)
    
    if args.algo:
        if args.algo in menu.algos:
            menu.selected_algo_idx = menu.algos.index(args.algo)
            
    if args.style:
        if args.style in menu.styles:
            menu.selected_style_idx = menu.styles.index(args.style)

    state = "menu"
    current_env = None
    current_model = None

    while True:
        if state == "menu":
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            
            menu_action = menu.update(events)
            
            # Handle Sidebar clicks separately for better UX if needed
            if any(e.type == pygame.MOUSEBUTTONDOWN for e in events):
                # We need to re-check clicks for sidebar toggles
                menu.handle_sidebar_clicks(pygame.mouse.get_pos())

            renderer.render_menu(menu)
            
            if menu_action == "start" or (any(e.type == pygame.KEYDOWN and e.key == pygame.K_RETURN for e in events)):
                selection = menu.get_selection()
                if selection:
                    # Load model
                    print(f"Loading {selection['model']}...")
                    algo_class = PPO if selection['algo'] == 'ppo' else DQN
                    try:
                        current_model = algo_class.load(selection['model'])
                        # Create/Update environment
                        if current_env:
                            current_env.close()
                        current_env = ArenaEnv(control_style=selection['style'], render_mode="human")
                        # Sync renderer
                        current_env.renderer = renderer
                        
                        state = "evaluating"
                    except Exception as e:
                        print(f"Error loading model: {e}")
                else:
                    print("No model selected or found.")
            
            if menu_action == "quit":
                break

        elif state == "evaluating":
            result = run_evaluation(current_model, current_env, renderer, menu.deterministic)
            if result == "menu":
                state = "menu"
            elif result == "quit":
                break

    if current_env:
        current_env.close()
    renderer.close()


if __name__ == "__main__":
    main()
