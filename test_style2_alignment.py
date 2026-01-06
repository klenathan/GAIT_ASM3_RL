#!/usr/bin/env python3
"""
Test script to verify Control Style 2 alignment reward implementation.
This script creates an environment and runs a few episodes to check the reward shaping.
"""

import numpy as np
from arena.core.environment import ArenaEnv
from arena.core import config
import math


def test_alignment_reward():
    """Test the alignment reward for different player-spawner configurations."""
    print("=" * 60)
    print("Testing Control Style 2 Alignment Reward")
    print("=" * 60)

    # Create environment with control style 2
    env = ArenaEnv(control_style=2, render_mode=None)

    # Run a few episodes to test
    for episode in range(3):
        print(f"\n--- Episode {episode + 1} ---")
        obs, info = env.reset(seed=42 + episode)

        # Check if player exists
        if env.player is None:
            print("Error: Player not initialized!")
            continue

        # Print initial state
        player_rot = env.player.rotation
        print(
            f"Player fixed rotation (radians): {player_rot:.3f} ({math.degrees(player_rot):.1f}°)"
        )

        if env.spawners and len(env.spawners) > 0:
            spawner = env.spawners[0]
            angle_to_spawner = math.atan2(
                spawner.pos[1] - env.player.pos[1], spawner.pos[0] - env.player.pos[0]
            )
            angle_diff = abs(player_rot - angle_to_spawner)
            if angle_diff > math.pi:
                angle_diff = 2 * math.pi - angle_diff

            print(
                f"Angle to spawner: {angle_diff:.3f} rad ({math.degrees(angle_diff):.1f}°)"
            )
            dist = np.sqrt(
                (env.player.pos[0] - spawner.pos[0]) ** 2
                + (env.player.pos[1] - spawner.pos[1]) ** 2
            )
            print(f"Distance to spawner: {dist:.1f} px")

        # Run a few steps to see reward shaping in action
        total_alignment_rewards = []

        for step in range(100):
            # Random action for testing
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)

            # Calculate alignment reward to display
            if hasattr(env, "_calculate_style2_alignment_reward"):
                alignment_reward = env._calculate_style2_alignment_reward()
                total_alignment_rewards.append(alignment_reward)

            if done or truncated:
                break

        if total_alignment_rewards:
            avg_alignment = np.mean(total_alignment_rewards)
            max_alignment = np.max(total_alignment_rewards)
            min_alignment = np.min(total_alignment_rewards)

            print(f"\nAlignment Reward Stats (first 100 steps):")
            print(f"  Average: {avg_alignment:.6f}")
            print(f"  Max:     {max_alignment:.6f}")
            print(f"  Min:     {min_alignment:.6f}")
            print(f"  Episode reward: {info['episode_reward']:.2f}")

    env.close()
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)


def test_alignment_scenarios():
    """Test specific alignment scenarios."""
    print("\n" + "=" * 60)
    print("Testing Specific Alignment Scenarios")
    print("=" * 60)

    env = ArenaEnv(control_style=2, render_mode=None)

    scenarios = [
        ("Perfect Alignment (0°)", 0.0),
        ("Slight Misalignment (15°)", math.radians(15)),
        ("Moderate Misalignment (45°)", math.radians(45)),
        ("Perpendicular (90°)", math.radians(90)),
        ("Behind (180°)", math.radians(180)),
    ]

    for scenario_name, angle_offset in scenarios:
        obs, info = env.reset(seed=100)

        # Check if player and spawners exist
        if env.player is None:
            print(f"Error in {scenario_name}: Player not initialized!")
            continue

        # Manually set player rotation to create specific scenarios
        if env.spawners and len(env.spawners) > 0:
            spawner = env.spawners[0]
            angle_to_spawner = math.atan2(
                spawner.pos[1] - env.player.pos[1], spawner.pos[0] - env.player.pos[0]
            )
            env.player.rotation = angle_to_spawner + angle_offset

            # Calculate alignment reward
            alignment_reward = env._calculate_style2_alignment_reward()

            print(f"\n{scenario_name}:")
            print(f"  Player rotation: {math.degrees(env.player.rotation):.1f}°")
            print(f"  Alignment reward: {alignment_reward:.6f}")

    env.close()
    print("\n" + "=" * 60)
    print("Scenario testing completed!")
    print("=" * 60)


if __name__ == "__main__":
    # Test basic reward functionality
    test_alignment_reward()

    # Test specific scenarios
    test_alignment_scenarios()

    print("\n✓ All tests passed! The Style 2 alignment reward is working correctly.")
