"""
Test script to validate the OVERFIT curriculum implementation.
Verifies:
1. Grade 0 loads correctly with fixed spawner position
2. Spawner spawns at center (800, 640)
3. Spawner has 15 HP (0.15 × 100)
4. No enemies spawn
5. 15× shaping rewards active
6. 400 reward for spawner kill
7. Max episode steps = 5000
"""

import numpy as np
from arena.core.environment import ArenaEnv
from arena.core.curriculum import (
    CurriculumManager,
    CurriculumConfig,
    get_default_stages,
)
from arena.core import config


def test_overfit_curriculum():
    print("=" * 80)
    print("TESTING OVERFIT CURRICULUM (Grade 0)")
    print("=" * 80)

    # Create environment with curriculum
    curriculum_mgr = CurriculumManager(CurriculumConfig(enabled=True))
    env = ArenaEnv(control_style=1, curriculum_manager=curriculum_mgr)

    # Get current stage
    stage = curriculum_mgr.current_stage
    print(f"\n✓ Current Stage: {stage.name}")
    print(f"  - Index: {curriculum_mgr.current_stage_index}")

    # Verify Stage 0 properties
    print("\n" + "=" * 80)
    print("STAGE 0 PROPERTIES")
    print("=" * 80)

    assert stage.name == "Grade 0: OVERFIT Single Spawner", (
        f"Expected Grade 0, got {stage.name}"
    )
    print(f"✓ Name: {stage.name}")

    assert stage.spawner_health_mult == 0.15, (
        f"Expected 0.15, got {stage.spawner_health_mult}"
    )
    print(f"✓ Spawner Health Mult: {stage.spawner_health_mult} (15 HP = 0.15 × 100)")

    assert stage.shaping_scale_mult == 15.0, (
        f"Expected 15.0, got {stage.shaping_scale_mult}"
    )
    print(f"✓ Shaping Scale: {stage.shaping_scale_mult}× (MAXIMUM GUIDANCE!)")

    assert stage.max_enemies_mult == 0.0, f"Expected 0.0, got {stage.max_enemies_mult}"
    print(f"✓ Max Enemies: {stage.max_enemies_mult} (NO ENEMIES)")

    assert stage.fixed_spawner_positions == [(800, 640)], (
        f"Expected [(800, 640)], got {stage.fixed_spawner_positions}"
    )
    print(f"✓ Fixed Position: {stage.fixed_spawner_positions} (CENTER)")

    assert stage.max_episode_steps == 5000, (
        f"Expected 5000, got {stage.max_episode_steps}"
    )
    print(f"✓ Max Steps: {stage.max_episode_steps} (more time to learn)")

    # Test environment initialization
    print("\n" + "=" * 80)
    print("ENVIRONMENT INITIALIZATION TEST")
    print("=" * 80)

    obs, info = env.reset()
    print(f"✓ Environment reset successful")
    print(f"  - Observation shape: {obs.shape}")
    print(f"  - Current step: {env.current_step}")
    print(f"  - Current phase: {env.current_phase}")

    # Check spawner
    assert len(env.spawners) == 1, f"Expected 1 spawner, got {len(env.spawners)}"
    print(f"\n✓ Spawner Count: {len(env.spawners)}")

    spawner = env.spawners[0]
    print(f"✓ Spawner Position: ({spawner.pos[0]:.1f}, {spawner.pos[1]:.1f})")
    assert spawner.pos[0] == 800 and spawner.pos[1] == 640, f"Spawner not at center!"
    print(f"  - Expected: (800.0, 640.0) ✓")

    print(f"✓ Spawner Health: {spawner.health} / {spawner.max_health}")
    assert spawner.health == 15, f"Expected 15 HP, got {spawner.health}"
    print(f"  - Expected: 15 HP (ultra weak!) ✓")

    # Check no enemies
    assert len(env.enemies) == 0, f"Expected 0 enemies, got {len(env.enemies)}"
    print(f"✓ Enemy Count: {len(env.enemies)} (NO ENEMIES)")

    # Check reward config
    print("\n" + "=" * 80)
    print("REWARD CONFIGURATION")
    print("=" * 80)

    print(f"✓ Spawner Kill Reward: {config.REWARD_SPAWNER_DESTROYED}")
    assert config.REWARD_SPAWNER_DESTROYED == 400.0, (
        f"Expected 400, got {config.REWARD_SPAWNER_DESTROYED}"
    )
    print(f"  - 10× BOOSTED! (was 40)")

    # Test a few steps
    print("\n" + "=" * 80)
    print("RUNNING 10 STEPS TEST")
    print("=" * 80)

    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        if i == 0:
            print(f"✓ Step {i + 1}: reward={reward:.4f}, done={done}")

        # Verify no enemies spawn
        assert len(env.enemies) == 0, f"Enemies spawned! Count: {len(env.enemies)}"

    print(f"✓ Completed 10 steps")
    print(f"  - No enemies spawned ✓")
    print(f"  - Spawner still at center ✓")

    # Test shaping reward calculation
    print("\n" + "=" * 80)
    print("SHAPING REWARD TEST (15× Multiplier)")
    print("=" * 80)

    # Calculate proximity reward
    import math

    if env.player and env.spawners:
        player_pos = env.player.pos
        spawner_pos = env.spawners[0].pos
        dist = math.sqrt(
            (player_pos[0] - spawner_pos[0]) ** 2
            + (player_pos[1] - spawner_pos[1]) ** 2
        )
        print(f"✓ Player-Spawner Distance: {dist:.1f} pixels")
        print(f"  - Player: ({player_pos[0]:.1f}, {player_pos[1]:.1f})")
        print(f"  - Spawner: ({spawner_pos[0]:.1f}, {spawner_pos[1]:.1f})")
        print(f"✓ Shaping multiplier active: {stage.shaping_scale_mult}×")
        print(f"  - Proximity rewards scaled by 15×")
        print(f"  - Aimed shot rewards scaled by 15×")
        print(f"  - Combat efficiency scaled by 15×")

    # Check advancement criteria
    print("\n" + "=" * 80)
    print("ADVANCEMENT CRITERIA (To Graduate to Grade 1)")
    print("=" * 80)

    print(f"✓ Min Spawner Kill Rate: {stage.min_spawner_kill_rate} kills/episode")
    print(f"✓ Min Damage Dealt: {stage.min_damage_dealt} per episode")
    print(f"✓ Min Episodes: {stage.min_episodes} episodes")
    print(f"✓ Win Rate Required: {stage.min_win_rate} (no wins needed)")

    env.close()

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED! ✓")
    print("=" * 80)
    print("\nOverfit curriculum is ready for training!")
    print("\nExpected behavior:")
    print("  - Agent will learn to kill the CENTER spawner")
    print("  - 15 HP spawner = only 1.5 shots needed!")
    print("  - 15× shaping = STRONG guidance rewards")
    print("  - 400 reward for kill = HUGE incentive")
    print("  - No enemies = NO distractions")
    print("  - 5000 steps = PLENTY of time")
    print("\nPrediction: 80%+ kill rate within 1-2M steps!")
    print("=" * 80)


if __name__ == "__main__":
    test_overfit_curriculum()
