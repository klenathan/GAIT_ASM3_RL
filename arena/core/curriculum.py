"""
Curriculum Learning System for Arena RL Training.
Uses Strategy pattern for advancement criteria.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Callable
import numpy as np


# =============================================================================
# STRATEGY PATTERN: Advancement Criteria
# =============================================================================


class AdvancementStrategy(ABC):
    """Abstract base class for curriculum advancement strategies."""

    @abstractmethod
    def should_advance(self, metrics: "CurriculumMetrics") -> bool:
        """Determine if the agent should advance to the next stage."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return strategy name for logging."""
        pass


class SpawnerKillRateStrategy(AdvancementStrategy):
    """Advance when spawner kill rate exceeds threshold."""

    def __init__(self, threshold: float = 0.3, min_episodes: int = 50):
        self.threshold = threshold
        self.min_episodes = min_episodes

    def should_advance(self, metrics: "CurriculumMetrics") -> bool:
        if len(metrics.spawner_kills) < self.min_episodes:
            return False
        kill_rate = np.mean(metrics.spawner_kills[-100:])
        return kill_rate >= self.threshold

    def get_name(self) -> str:
        return f"SpawnerKillRate(threshold={self.threshold})"


class WinRateStrategy(AdvancementStrategy):
    """Advance when win rate exceeds threshold."""

    def __init__(self, threshold: float = 0.2, min_episodes: int = 50):
        self.threshold = threshold
        self.min_episodes = min_episodes

    def should_advance(self, metrics: "CurriculumMetrics") -> bool:
        if len(metrics.wins) < self.min_episodes:
            return False
        win_rate = np.mean(metrics.wins[-100:])
        return win_rate >= self.threshold

    def get_name(self) -> str:
        return f"WinRate(threshold={self.threshold})"


class SurvivalTimeStrategy(AdvancementStrategy):
    """Advance when average survival time exceeds threshold."""

    def __init__(self, threshold_steps: int = 1000, min_episodes: int = 50):
        self.threshold_steps = threshold_steps
        self.min_episodes = min_episodes

    def should_advance(self, metrics: "CurriculumMetrics") -> bool:
        if len(metrics.episode_lengths) < self.min_episodes:
            return False
        avg_length = np.mean(metrics.episode_lengths[-100:])
        return avg_length >= self.threshold_steps

    def get_name(self) -> str:
        return f"SurvivalTime(threshold={self.threshold_steps})"


class CompositeStrategy(AdvancementStrategy):
    """Combine multiple strategies with AND/OR logic."""

    def __init__(
        self, strategies: List[AdvancementStrategy], require_all: bool = False
    ):
        self.strategies = strategies
        self.require_all = require_all

    def should_advance(self, metrics: "CurriculumMetrics") -> bool:
        results = [s.should_advance(metrics) for s in self.strategies]
        return all(results) if self.require_all else any(results)

    def get_name(self) -> str:
        logic = "AND" if self.require_all else "OR"
        names = [s.get_name() for s in self.strategies]
        return f"Composite({logic}: {names})"


class StageBasedStrategy(AdvancementStrategy):
    """
    Advancement strategy that uses criteria defined in the current CurriculumStage.
    This ensures each stage can have unique, progressively harder requirements.
    """

    def __init__(self, curriculum_manager: "CurriculumManager"):
        self.curriculum_manager = curriculum_manager

    def should_advance(self, metrics: "CurriculumMetrics") -> bool:
        stage = self.curriculum_manager.current_stage

        # Check minimum episodes requirement
        if len(metrics.spawner_kills) < stage.min_episodes:
            return False

        # Check all criteria (using last 100 episodes average)
        window = min(100, len(metrics.spawner_kills))

        spawner_kill_rate = np.mean(metrics.spawner_kills[-window:])
        win_rate = np.mean(metrics.wins[-window:])
        avg_survival = np.mean(metrics.episode_lengths[-window:])
        avg_enemy_kills = np.mean(metrics.enemy_kills[-window:])
        avg_damage_dealt = np.mean(metrics.damage_dealt[-window:])
        avg_damage_taken = np.mean(metrics.damage_taken[-window:])

        # Average win time (only from wins in window)
        recent_win_times = [
            t
            for t, w in zip(metrics.episode_lengths[-window:], metrics.wins[-window:])
            if w == 1
        ]
        avg_win_time = np.mean(recent_win_times) if recent_win_times else 999999

        # All criteria must be met
        criteria_met = (
            spawner_kill_rate >= stage.min_spawner_kill_rate
            and win_rate >= stage.min_win_rate
            and avg_survival >= stage.min_survival_steps
            and avg_survival <= stage.max_survival_steps  # Not too passive
            and avg_enemy_kills >= stage.min_enemy_kill_rate
            and avg_damage_dealt >= stage.min_damage_dealt
            and avg_damage_taken <= stage.max_damage_taken
            and avg_win_time <= stage.max_win_time  # Fast wins required
        )

        return criteria_met

    def get_name(self) -> str:
        stage = self.curriculum_manager.current_stage
        return (
            f"StageBased(spawners>={stage.min_spawner_kill_rate}, "
            f"wins>={stage.min_win_rate}, enemies>={stage.min_enemy_kill_rate})"
        )


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class CurriculumStage:
    """Defines difficulty modifiers and advancement criteria for a curriculum stage."""

    name: str
    spawn_cooldown_mult: float = 1.0  # Higher = slower spawns
    max_enemies_mult: float = 1.0  # Lower = fewer enemies
    spawner_health_mult: float = 1.0  # Lower = easier to kill
    enemy_speed_mult: float = 1.0  # Lower = slower enemies
    shaping_scale_mult: float = 1.0  # Higher = stronger guidance
    damage_penalty_mult: float = 1.0  # Higher = more severe damage penalties

    # Overfit training options
    fixed_spawner_positions: Optional[List[tuple]] = (
        None  # [(x, y), ...] or None for random
    )
    max_episode_steps: Optional[int] = None  # Override MAX_STEPS, None = use default

    # Advancement criteria (configurable per stage)
    min_spawner_kill_rate: float = 0.3  # Required avg spawner kills per episode
    min_win_rate: float = 0.0  # Required win rate to advance
    min_survival_steps: int = 500  # Required avg episode length
    # Maximum avg episode length (penalizes passivity)
    max_survival_steps: int = 999999
    min_enemy_kill_rate: float = 0.0  # Required avg enemy kills per episode
    min_damage_dealt: float = 0.0  # Required avg damage dealt per episode
    # Maximum avg damage taken (lower = better defense)
    max_damage_taken: float = 999999.0
    # Maximum avg steps to win (lower = faster wins)
    max_win_time: int = 999999
    min_episodes: int = 50  # Minimum episodes before advancement check

    def __repr__(self):
        return f"Stage({self.name})"


@dataclass
class CurriculumMetrics:
    """Tracks episode outcomes for advancement decisions."""

    spawner_kills: List[float] = field(default_factory=list)
    wins: List[int] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    episode_rewards: List[float] = field(default_factory=list)
    # New combat engagement metrics
    enemy_kills: List[int] = field(default_factory=list)
    damage_dealt: List[float] = field(default_factory=list)
    damage_taken: List[float] = field(default_factory=list)
    # Steps to win (only for wins)
    win_times: List[int] = field(default_factory=list)

    def record_episode(
        self,
        spawners_killed: int,
        won: bool,
        length: int,
        reward: float,
        enemy_kills: int = 0,
        damage_dealt: float = 0,
        damage_taken: float = 0,
    ):
        self.spawner_kills.append(float(spawners_killed))
        self.wins.append(1 if won else 0)
        self.episode_lengths.append(length)
        self.episode_rewards.append(reward)
        self.enemy_kills.append(enemy_kills)
        self.damage_dealt.append(damage_dealt)
        self.damage_taken.append(damage_taken)

        # Track time to win for successful episodes
        if won:
            self.win_times.append(length)

        # Keep only last 200 episodes to limit memory
        for lst in [
            self.spawner_kills,
            self.wins,
            self.episode_lengths,
            self.episode_rewards,
            self.enemy_kills,
            self.damage_dealt,
            self.damage_taken,
        ]:
            if len(lst) > 200:
                lst.pop(0)

        # Keep last 100 win times (less frequent)
        if len(self.win_times) > 100:
            self.win_times.pop(0)


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning."""

    enabled: bool = True
    stages: List[CurriculumStage] = field(default_factory=list)
    strategy: Optional[AdvancementStrategy] = None

    def __post_init__(self):
        if not self.stages:
            self.stages = get_default_stages()
        # Note: strategy will be set by CurriculumManager to use StageBasedStrategy


def get_default_stages() -> List[CurriculumStage]:
    """
    Return 50-stage ultra-granular curriculum for fast learning (<10M timesteps).

    STRUCTURE:
    - Stages 0-19: Single spawner positioning & shooting mastery (20 stages)
    - Stages 20-39: Multiple spawners management (20 stages)
    - Stages 40-49: Enemy handling with multiple spawners (10 stages)

    DESIGN PRINCIPLES:
    - Ultra-small steps for rapid progression
    - Fast advancement (min_episodes=100-150 per stage)
    - Each stage ~50-200K timesteps = 50 stages × 150K = 7.5M total
    """
    stages = []

    # =========================================================================
    # PART 1: SINGLE SPAWNER MASTERY (Stages 0-19) - 20 stages
    # Goal: Learn positioning, aiming, shooting at ONE target
    # =========================================================================

    # Stages 0-4: Ultra-weak spawners, learn basic positioning (5 stages)
    for i in range(5):
        stages.append(
            CurriculumStage(
                name=f"S{i}: Ultra Basic Positioning",
                spawn_cooldown_mult=999.0,  # No enemies
                max_enemies_mult=0.0,
                spawner_health_mult=0.08 + i * 0.02,  # 8, 10, 12, 14, 16 HP
                enemy_speed_mult=0.5,
                shaping_scale_mult=25.0 - i * 1.0,  # 25→21 (strong guidance)
                damage_penalty_mult=0.05,
                fixed_spawner_positions=[(800, 640)],  # Center only
                max_episode_steps=2000,
                min_spawner_kill_rate=0.5 + i * 0.05,  # 50%→70%
                min_win_rate=0.0,
                min_survival_steps=200,
                max_survival_steps=1800,
                min_damage_dealt=5.0 + i * 2.0,
                min_episodes=100,  # Fast progression
            )
        )

    # Stages 5-9: Multiple fixed positions, transfer positioning (5 stages)
    fixed_positions_list = [
        [(800, 640)],  # Center
        [(400, 400), (1200, 880)],  # 2 positions
        [(400, 400), (800, 640), (1200, 880)],  # 3 positions
        [(300, 300), (1300, 300), (300, 980), (1300, 980)],  # 4 corners
        None,  # Random!
    ]
    for i in range(5):
        stages.append(
            CurriculumStage(
                name=f"S{5 + i}: Position Transfer {i + 1}/5",
                spawn_cooldown_mult=999.0,
                max_enemies_mult=0.0,
                spawner_health_mult=0.15 + i * 0.03,  # 15→27 HP
                enemy_speed_mult=0.5,
                shaping_scale_mult=20.0 - i * 2.0,  # 20→12
                damage_penalty_mult=0.1,
                fixed_spawner_positions=fixed_positions_list[i],
                max_episode_steps=2500,
                min_spawner_kill_rate=0.65 + i * 0.03,  # 65%→77%
                min_win_rate=0.0,
                min_survival_steps=250,
                max_survival_steps=2200,
                min_damage_dealt=15.0 + i * 3.0,
                min_episodes=120,
            )
        )

    # Stages 10-14: Increase spawner health gradually (5 stages)
    for i in range(5):
        stages.append(
            CurriculumStage(
                name=f"S{10 + i}: Tougher Spawner {i + 1}/5",
                spawn_cooldown_mult=999.0,
                max_enemies_mult=0.0,
                spawner_health_mult=0.30 + i * 0.10,  # 30→70 HP
                enemy_speed_mult=0.5,
                shaping_scale_mult=12.0 - i * 1.5,  # 12→6
                damage_penalty_mult=0.2 + i * 0.05,
                fixed_spawner_positions=None,  # Random
                max_episode_steps=2800,
                min_spawner_kill_rate=0.75 + i * 0.02,  # 75%→83%
                min_win_rate=0.0,
                min_survival_steps=300,
                max_survival_steps=2500,
                min_damage_dealt=30.0 + i * 8.0,
                min_episodes=120,
            )
        )

    # Stages 15-19: Full health single spawner, efficiency (5 stages)
    for i in range(5):
        stages.append(
            CurriculumStage(
                name=f"S{15 + i}: Efficient Single Kill {i + 1}/5",
                spawn_cooldown_mult=999.0,
                max_enemies_mult=0.0,
                spawner_health_mult=0.75 + i * 0.05,  # 75→95 HP
                enemy_speed_mult=0.5,
                shaping_scale_mult=6.0 - i * 1.0,  # 6→2 (reduce guidance)
                damage_penalty_mult=0.4 + i * 0.05,
                fixed_spawner_positions=None,
                max_episode_steps=2500,
                min_spawner_kill_rate=0.85 + i * 0.02,  # 85%→93%
                min_win_rate=0.0,
                min_survival_steps=300,
                max_survival_steps=2200,
                min_damage_dealt=70.0 + i * 5.0,
                min_episodes=120,
            )
        )

    # =========================================================================
    # PART 2: MULTIPLE SPAWNERS (Stages 20-39) - 20 stages
    # Goal: Manage 2+ spawners, prioritize targets, efficiency
    # =========================================================================

    # Stages 20-24: Introduce 2nd spawner gradually (5 stages)
    for i in range(5):
        # Phase config will spawn 1 spawner by default, but we handle multiple phases
        # So agent needs to survive and clear phase 1, then phase 2
        stages.append(
            CurriculumStage(
                name=f"S{20 + i}: Two Spawners Intro {i + 1}/5",
                spawn_cooldown_mult=999.0,  # Still no enemies
                max_enemies_mult=0.0,
                spawner_health_mult=0.80 + i * 0.04,  # 80→96 HP (smooth from S19)
                enemy_speed_mult=0.5,
                shaping_scale_mult=5.0 - i * 0.5,  # 5→3
                damage_penalty_mult=0.5 + i * 0.05,
                fixed_spawner_positions=None,
                max_episode_steps=3500,
                min_spawner_kill_rate=1.0 + i * 0.05,  # 1.0→1.2 spawners/ep
                min_win_rate=0.0,
                min_survival_steps=400,
                max_survival_steps=3200,
                min_damage_dealt=60.0 + i * 10.0,
                min_episodes=120,  # Reduced from 150
            )
        )

    # Stages 25-29: Increase multi-spawner difficulty (5 stages)
    for i in range(5):
        stages.append(
            CurriculumStage(
                name=f"S{25 + i}: Multi-Spawner Progress {i + 1}/5",
                spawn_cooldown_mult=999.0,
                max_enemies_mult=0.0,
                spawner_health_mult=0.70 + i * 0.05,  # 70→90 HP
                enemy_speed_mult=0.5,
                shaping_scale_mult=3.0 - i * 0.3,  # 3→1.8
                damage_penalty_mult=0.6 + i * 0.05,
                fixed_spawner_positions=None,
                max_episode_steps=3500,
                min_spawner_kill_rate=1.2 + i * 0.08,  # 1.2→1.52
                min_win_rate=0.0,
                min_survival_steps=450,
                max_survival_steps=3000,
                min_damage_dealt=80.0 + i * 10.0,
                min_episodes=120,  # Reduced from 150
            )
        )

    # Stages 30-34: Full health multiple spawners (5 stages)
    for i in range(5):
        stages.append(
            CurriculumStage(
                name=f"S{30 + i}: Full HP Multi-Spawner {i + 1}/5",
                spawn_cooldown_mult=999.0,
                max_enemies_mult=0.0,
                spawner_health_mult=0.90 + i * 0.02,  # 90→98 HP
                enemy_speed_mult=0.5,
                shaping_scale_mult=1.8 - i * 0.2,  # 1.8→1.0
                damage_penalty_mult=0.7 + i * 0.05,
                fixed_spawner_positions=None,
                max_episode_steps=3200,
                min_spawner_kill_rate=1.5 + i * 0.05,  # 1.5→1.7
                min_win_rate=0.0,
                min_survival_steps=500,
                max_survival_steps=2800,
                min_damage_dealt=110.0 + i * 10.0,
                min_episodes=120,  # Reduced from 150
            )
        )

    # Stages 35-39: Efficient multi-spawner clearing (5 stages)
    for i in range(5):
        stages.append(
            CurriculumStage(
                name=f"S{35 + i}: Efficient Multi-Clear {i + 1}/5",
                spawn_cooldown_mult=999.0,
                max_enemies_mult=0.0,
                spawner_health_mult=1.0,  # Full 100 HP
                enemy_speed_mult=0.5,
                shaping_scale_mult=1.0,  # Minimal guidance
                damage_penalty_mult=0.8 + i * 0.03,
                fixed_spawner_positions=None,
                max_episode_steps=3000,
                min_spawner_kill_rate=1.7 + i * 0.1,  # 1.7→2.1
                min_win_rate=0.0,
                min_survival_steps=550,
                max_survival_steps=2600,
                min_damage_dealt=140.0 + i * 15.0,
                min_episodes=120,  # Reduced from 150
            )
        )

    # =========================================================================
    # PART 3: ENEMIES + MULTIPLE SPAWNERS (Stages 40-49) - 10 stages
    # Goal: Handle enemies while killing spawners, win consistently
    # =========================================================================

    # Stages 40-42: Introduce slow enemies (3 stages)
    for i in range(3):
        stages.append(
            CurriculumStage(
                name=f"S{40 + i}: Slow Enemies Intro {i + 1}/3",
                spawn_cooldown_mult=5.0 - i * 1.0,  # 5.0→3.0 (slow spawns)
                max_enemies_mult=0.15 + i * 0.10,  # 15%→35% enemies
                spawner_health_mult=1.0,
                enemy_speed_mult=0.40 + i * 0.10,  # 40%→60% speed
                shaping_scale_mult=1.0,
                damage_penalty_mult=0.85 + i * 0.05,
                fixed_spawner_positions=None,
                max_episode_steps=3000,
                min_spawner_kill_rate=1.5 + i * 0.1,  # 1.5→1.7
                min_win_rate=0.0,
                min_survival_steps=600,
                max_survival_steps=2500,
                min_enemy_kill_rate=0.5 + i * 0.3,  # 0.5→1.1
                min_damage_dealt=150.0 + i * 15.0,
                min_episodes=120,  # Reduced from 150
            )
        )

    # Stages 43-45: Moderate enemies (3 stages)
    for i in range(3):
        stages.append(
            CurriculumStage(
                name=f"S{43 + i}: Moderate Enemies {i + 1}/3",
                spawn_cooldown_mult=3.0 - i * 0.5,  # 3.0→2.0
                max_enemies_mult=0.40 + i * 0.15,  # 40%→70%
                spawner_health_mult=1.0,
                enemy_speed_mult=0.60 + i * 0.10,  # 60%→80%
                shaping_scale_mult=1.0,
                damage_penalty_mult=0.90 + i * 0.03,
                fixed_spawner_positions=None,
                max_episode_steps=3000,
                min_spawner_kill_rate=1.7 + i * 0.15,  # 1.7→2.0
                min_win_rate=0.02 + i * 0.02,  # 2%→6%
                min_survival_steps=650,
                max_survival_steps=2300,
                min_enemy_kill_rate=1.5 + i * 0.5,  # 1.5→2.5
                min_damage_dealt=180.0 + i * 20.0,
                min_episodes=120,  # Reduced from 150
            )
        )

    # Stages 46-49: Full difficulty ramp to mastery (4 stages)
    for i in range(4):
        stages.append(
            CurriculumStage(
                name=f"S{46 + i}: Path to Mastery {i + 1}/4",
                spawn_cooldown_mult=1.5 - i * 0.15,  # 1.5→1.05
                max_enemies_mult=0.75 + i * 0.08,  # 75%→99%
                spawner_health_mult=1.0,
                enemy_speed_mult=0.85 + i * 0.05,  # 85%→100%
                shaping_scale_mult=1.0,
                damage_penalty_mult=0.95 + i * 0.01,
                fixed_spawner_positions=None,
                max_episode_steps=3000,
                min_spawner_kill_rate=2.0 + i * 0.15,  # 2.0→2.45
                min_win_rate=0.08 + i * 0.04,  # 8%→20%
                min_survival_steps=700,
                max_survival_steps=2000 - i * 50,  # 2000→1850 (faster)
                min_enemy_kill_rate=3.0 + i * 1.0,  # 3.0→6.0
                min_damage_dealt=220.0 + i * 20.0,  # 220→280
                max_damage_taken=200.0 - i * 15.0,  # Better defense
                min_episodes=120,  # Reduced from 150
            )
        )

    return stages


# =============================================================================
# CURRICULUM MANAGER
# =============================================================================


class CurriculumManager:
    """
    Manages curriculum progression across training.

    Usage:
        manager = CurriculumManager(CurriculumConfig())

        # In environment: get current modifiers
        stage = manager.current_stage
        spawn_cooldown = base_cooldown * stage.spawn_cooldown_mult

        # After each episode: record outcome
        manager.record_episode(spawners_killed=2, won=False, length=500, reward=100)

        # Check for advancement
        if manager.check_advancement():
            print(f"Advanced to stage {manager.current_stage_index}")
    """

    def __init__(self, config: Optional[CurriculumConfig] = None):
        self.config = config or CurriculumConfig()
        self.current_stage_index = 0
        self.metrics = CurriculumMetrics()
        self._advancement_callbacks: List[Callable[[int, CurriculumStage], None]] = []

        # Set StageBasedStrategy as default if no strategy is configured
        if self.config.strategy is None:
            self.config.strategy = StageBasedStrategy(self)

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    @property
    def current_stage(self) -> CurriculumStage:
        return self.config.stages[self.current_stage_index]

    @property
    def max_stage(self) -> int:
        return len(self.config.stages) - 1

    @property
    def is_final_stage(self) -> bool:
        return self.current_stage_index >= self.max_stage

    def record_episode(
        self,
        spawners_killed: int,
        won: bool,
        length: int,
        reward: float,
        enemy_kills: int = 0,
        damage_dealt: float = 0,
        damage_taken: float = 0,
    ):
        """Record episode outcome for advancement evaluation."""
        self.metrics.record_episode(
            spawners_killed,
            won,
            length,
            reward,
            enemy_kills,
            damage_dealt,
            damage_taken,
        )

    def check_advancement(self) -> bool:
        """Check if agent should advance to next stage. Returns True if advanced."""
        if not self.enabled or self.is_final_stage:
            return False

        if self.config.strategy.should_advance(self.metrics):
            self.current_stage_index += 1
            self._notify_advancement()
            return True
        return False

    def on_advancement(self, callback: Callable[[int, CurriculumStage], None]):
        """Register callback for stage advancement events."""
        self._advancement_callbacks.append(callback)

    def _notify_advancement(self):
        """Notify all registered callbacks of advancement."""
        for cb in self._advancement_callbacks:
            cb(self.current_stage_index, self.current_stage)

    def get_modified_value(self, base_value: float, modifier_name: str) -> float:
        """Apply current stage modifier to a base value."""
        if not self.enabled:
            return base_value
        modifier = getattr(self.current_stage, modifier_name, 1.0)
        return base_value * modifier

    def get_status_dict(self) -> dict:
        """Return status for logging."""
        return {
            "stage_index": self.current_stage_index,
            "stage_name": self.current_stage.name,
            "episodes_recorded": len(self.metrics.spawner_kills),
            "strategy": self.config.strategy.get_name()
            if self.config.strategy
            else "None",
        }

    def to_dict(self) -> dict:
        """Serialize curriculum state for checkpointing."""
        return {
            "current_stage_index": self.current_stage_index,
            "metrics": {
                # Keep last 200 episodes to limit file size
                "spawner_kills": list(self.metrics.spawner_kills[-200:]),
                "wins": list(self.metrics.wins[-200:]),
                "episode_lengths": list(self.metrics.episode_lengths[-200:]),
                "episode_rewards": list(self.metrics.episode_rewards[-200:]),
                "enemy_kills": list(self.metrics.enemy_kills[-200:]),
                "damage_dealt": list(self.metrics.damage_dealt[-200:]),
                "damage_taken": list(self.metrics.damage_taken[-200:]),
                "win_times": list(self.metrics.win_times[-100:]),
            },
        }

    def load_from_dict(self, data: dict):
        """Restore curriculum state from checkpoint."""
        if not data:
            return

        self.current_stage_index = data.get("current_stage_index", 0)

        # Restore metrics if available
        if "metrics" in data:
            m = data["metrics"]
            self.metrics.spawner_kills = list(m.get("spawner_kills", []))
            self.metrics.wins = list(m.get("wins", []))
            self.metrics.episode_lengths = list(m.get("episode_lengths", []))
            self.metrics.episode_rewards = list(m.get("episode_rewards", []))
            self.metrics.enemy_kills = list(m.get("enemy_kills", []))
            self.metrics.damage_dealt = list(m.get("damage_dealt", []))
            self.metrics.damage_taken = list(m.get("damage_taken", []))
            self.metrics.win_times = list(m.get("win_times", []))
