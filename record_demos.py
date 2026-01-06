#!/usr/bin/env python3
"""
Convenience script for recording human demonstrations.
This provides a simplified interface for demo recording.
"""

import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arena.evaluation.evaluator import Evaluator
from arena.core.environment import ArenaEnv
from arena.game.human_controller import HumanController
from arena.training.imitation_learning import DemonstrationRecorder
from arena.ui.renderer import ArenaRenderer
import pygame


def record_demonstrations(style: int, output_path: str = None):
    """
    Record human demonstrations for imitation learning.

    Args:
        style: Control style (1 or 2)
        output_path: Optional custom path for saving demonstrations
    """
    print("\n" + "=" * 70)
    print("DEMONSTRATION RECORDING FOR IMITATION LEARNING")
    print("=" * 70)
    print(f"Control Style: {style}")
    print("\nControls:")
    if style == 1:
        print("  W/↑: Thrust forward")
        print("  A/←: Rotate left")
        print("  D/→: Rotate right")
        print("  Space: Shoot")
    else:
        print("  W/↑: Move up")
        print("  S/↓: Move down")
        print("  A/←: Move left")
        print("  D/→: Move right")
        print("  Space: Shoot")
    print("\n  ESC: Stop recording and save")
    print("\nTips:")
    print("  - Try to win episodes (destroy all spawners)")
    print("  - Avoid taking damage when possible")
    print("  - Show diverse strategies")
    print("  - Record at least 5-10 successful episodes")
    print("=" * 70 + "\n")

    # Create environment and recorder
    renderer = ArenaRenderer()
    env = ArenaEnv(control_style=style, render_mode="human")
    env.renderer = renderer
    env._owns_renderer = False

    controller = HumanController(style=style)
    recorder = DemonstrationRecorder(control_style=style)

    # Metrics for display
    metrics = {
        "episode": 0,
        "episode_reward": 0.0,
        "total_reward": 0.0,
        "is_human": True,
        "is_recording": True,
        "num_demos": 0,
    }

    obs, info = env.reset()
    recorder.start_episode()
    metrics["episode"] = 1
    running = True

    try:
        while running:
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    # UI Toggles
                    if event.key == pygame.K_h:
                        renderer.show_health = not renderer.show_health
                    if event.key == pygame.K_v:
                        renderer.show_vision = not renderer.show_vision
                    if event.key == pygame.K_d:
                        renderer.show_debug = not renderer.show_debug
                # Handle scroll events
                if event.type == pygame.MOUSEWHEEL:
                    renderer.handle_scroll(event)

            # Get action from human
            action = controller.get_action(events)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)

            # Record step
            recorder.record_step(obs, action, reward, terminated or truncated)

            metrics["episode_reward"] += reward
            metrics["total_reward"] += reward

            if terminated or truncated:
                # End recording for this episode
                recorder.end_episode(info)
                metrics["num_demos"] = len(recorder.buffer)

                # Display episode summary
                print(f"\nEpisode {metrics['episode']} complete:")
                print(f"  Reward: {metrics['episode_reward']:.1f}")
                print(f"  Win: {info.get('win', False)}")
                print(f"  Total demos recorded: {metrics['num_demos']}")

                # Check for victory
                if info.get("win", False):
                    renderer.render(env, training_metrics=metrics)
                    renderer.draw_victory_screen(
                        info["win_step"],
                        metrics["episode_reward"],
                        env.current_phase,
                    )
                    pygame.display.flip()

                    # Wait briefly
                    waiting = True
                    wait_start = pygame.time.get_ticks()
                    while waiting and (pygame.time.get_ticks() - wait_start < 2000):
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                running = False
                                waiting = False
                            if event.type == pygame.KEYDOWN:
                                if event.key == pygame.K_ESCAPE:
                                    running = False
                                    waiting = False
                                if event.key == pygame.K_SPACE:
                                    waiting = False
                        renderer.clock.tick(30)

                if not running:
                    break

                # Start new episode
                obs, info = env.reset()
                recorder.start_episode()
                metrics["episode"] += 1
                metrics["episode_reward"] = 0.0

            # Render
            renderer.set_model_output(None, style)
            renderer.render(env, training_metrics=metrics)

    finally:
        # Save demonstrations
        if len(recorder.buffer) > 0:
            filepath = recorder.save_demonstrations(output_path)
            print("\n" + "=" * 70)
            print("RECORDING COMPLETE")
            print("=" * 70)
            stats = recorder.buffer.get_statistics()
            print(f"Saved to: {filepath}")
            print(f"\nStatistics:")
            print(f"  Episodes recorded: {stats['num_demonstrations']}")
            print(f"  Total transitions: {stats['total_transitions']}")
            print(
                f"  Average return: {stats['avg_return']:.2f} ± {stats['std_return']:.2f}"
            )
            print(f"  Win rate: {stats['win_rate'] * 100:.1f}%")
            print(f"  Best episode: {stats['max_return']:.2f}")
            print(f"\nNext steps:")
            print(
                f"  1. Train with BC: python arena/train.py --demo-path {filepath} --bc-pretrain"
            )
            print(f"  2. Or use evaluate.py to load and visualize")
            print("=" * 70 + "\n")
        else:
            print("\nNo demonstrations recorded.")

        env.close()
        renderer.close()


def main():
    parser = argparse.ArgumentParser(
        description="Record human demonstrations for imitation learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Record for control style 1 (rotation + thrust)
  python record_demos.py --style 1

  # Record for control style 2 (directional movement)
  python record_demos.py --style 2

  # Save to custom location
  python record_demos.py --style 1 --output ./my_demos/expert_demos.pkl
        """,
    )
    parser.add_argument(
        "--style",
        type=int,
        required=True,
        choices=[1, 2],
        help="Control style: 1 (rotation+thrust) or 2 (directional)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Custom output path for demonstrations (default: auto-generated)",
    )

    args = parser.parse_args()

    try:
        record_demonstrations(args.style, args.output)
    except KeyboardInterrupt:
        print("\n\nRecording interrupted by user.")
    except Exception as e:
        print(f"\n\nError during recording: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
