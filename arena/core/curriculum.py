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
    Return default 8-stage curriculum with POSITIONING-FIRST strategy.

    CRITICAL INSIGHT: Agent must learn POSITIONING (move + aim) from the start.
    Fixed spawners = agent memorizes ONE solution (fails on random positions).

    Strategy - TEACH POSITIONING WITH VARIATION:
    - Grade 0: 5 FIXED POSITIONS (learn to reposition based on spawner location)
    - Grade 1: Random positions (transfer to infinite variations)
    - Grade 2-3: Kill more spawners, get efficient
    - Grade 4+: Add enemies progressively

    Why 5 positions works:
    - Forces agent to learn "IF spawner is at X, THEN move/rotate to Y"
    - Teaches decision-making based on observation (not memorization)
    - Smooth transfer to random positions (just more variations of same skill)
    """
    return [
        # =====================================================================
        # GRADE 0: MULTIPLE FIXED POSITIONS - Learn to Reposition!
        # =====================================================================
        # PHILOSOPHY: Teach positioning FROM THE START with LIMITED variation
        # Agent learns: "Observe spawner position → Calculate positioning → Execute"
        # 5 positions: top-left, top-right, center, bottom-left, bottom-right
        # Expected: 70%+ kill rate within 1.5M steps (learning to position!)
        CurriculumStage(
            name="Grade 0: Learn Positioning (5 Fixed Positions)",
            spawn_cooldown_mult=999.0,  # NO enemy spawning
            max_enemies_mult=0.0,  # NO ENEMIES AT ALL
            spawner_health_mult=0.12,  # ULTRA WEAK (12 HP = 1.2 shots!)
            enemy_speed_mult=0.5,  # N/A
            shaping_scale_mult=20.0,  # MAXIMUM GUIDANCE (20× proximity + aimed shot!)
            damage_penalty_mult=0.1,  # Minimal penalty - encourage exploration
            # 5 FIXED POSITIONS: Force agent to learn repositioning!
            fixed_spawner_positions=[
                (400, 320),  # Top-left quadrant
                (1200, 320),  # Top-right quadrant
                (800, 640),  # Center
                (400, 960),  # Bottom-left quadrant
                (1200, 960),  # Bottom-right quadrant
            ],
            max_episode_steps=4000,  # More time to explore positioning
            # Advancement: Must show consistent positioning skill
            min_spawner_kill_rate=0.7,  # 70% kill rate (learning positioning!)
            min_win_rate=0.0,  # No wins required
            min_survival_steps=250,  # Allow short episodes
            max_survival_steps=3500,  # Don't camp forever
            min_damage_dealt=12.0,  # Must deal some damage
            min_episodes=300,  # Train extensively to master repositioning
        ),
        # =====================================================================
        # GRADE 1: RANDOM POSITIONS - Transfer Positioning Skill!
        # =====================================================================
        # PHILOSOPHY: Agent already knows how to reposition (from 5-position training)
        # Now test if it generalizes to INFINITE variations
        # Expected: 50%+ kill rate within 500K steps (successful transfer!)
        CurriculumStage(
            name="Grade 1: Random Positions (Transfer Learning)",
            spawn_cooldown_mult=999.0,  # NO enemy spawning
            max_enemies_mult=0.0,  # NO ENEMIES
            spawner_health_mult=0.15,  # ULTRA WEAK (15 HP = 1.5 shots!)
            enemy_speed_mult=0.5,  # N/A
            shaping_scale_mult=15.0,  # HIGH guidance (maintain proximity + aimed shot!)
            damage_penalty_mult=0.2,  # Low penalty - encourage exploration
            # NO FIXED POSITIONS: Random spawner placement
            fixed_spawner_positions=None,  # Random positions!
            max_episode_steps=4000,  # Same as Grade 0
            # Advancement: Show positioning works on random locations
            min_spawner_kill_rate=0.6,  # 60% kill rate (transfer successful!)
            min_win_rate=0.0,  # No wins required
            min_survival_steps=250,  # Short episodes OK
            max_survival_steps=3500,  # Don't camp
            min_damage_dealt=15.0,  # Must deal damage
            min_episodes=250,  # Ensure transfer is solid
        ),
        # =====================================================================
        # GRADE 2: INCREASE SPAWNER DIFFICULTY - Get Efficient!
        # =====================================================================
        # Agent can position and kill weak spawners → now increase difficulty
        CurriculumStage(
            name="Grade 2: Tougher Spawners (Random)",
            spawn_cooldown_mult=999.0,  # Still no enemies
            max_enemies_mult=0.0,  # NO ENEMIES
            spawner_health_mult=0.30,  # Moderate (30 HP, 3 hits)
            enemy_speed_mult=0.5,  # N/A
            shaping_scale_mult=10.0,  # Good guidance
            damage_penalty_mult=0.3,  # Low penalty
            # Advancement: Must maintain kill rate with tougher spawners
            min_spawner_kill_rate=0.75,  # 75% kill rate
            min_win_rate=0.0,  # No wins required
            min_survival_steps=300,  # Short episodes
            max_survival_steps=3000,  # Efficient
            min_damage_dealt=30.0,  # More damage needed
            min_episodes=200,
        ),
        # =====================================================================
        # GRADE 3: KILL MULTIPLE SPAWNERS - Master Positioning!
        # =====================================================================
        # Goal: Kill spawners efficiently, handle phase transitions
        CurriculumStage(
            name="Grade 3: Kill Multiple Spawners",
            spawn_cooldown_mult=999.0,  # Still no enemies!
            max_enemies_mult=0.0,  # STILL NO ENEMIES
            spawner_health_mult=0.50,  # Moderate (50 HP, 5 hits)
            enemy_speed_mult=0.5,  # N/A
            shaping_scale_mult=7.0,  # Good guidance
            damage_penalty_mult=0.4,  # Moderate penalty
            # Advancement: Kill 1.2+ spawners per episode
            min_spawner_kill_rate=1.2,  # Must be efficient
            min_win_rate=0.0,  # No wins yet
            min_survival_steps=400,  # Reasonable time
            max_survival_steps=2500,  # Don't be passive
            min_damage_dealt=70.0,  # High damage
            min_episodes=300,
        ),
        # =====================================================================
        # GRADE 4: FULL HEALTH SPAWNERS + EFFICIENCY
        # =====================================================================
        # Goal: Kill spawners at full difficulty (health-wise), no enemies yet
        CurriculumStage(
            name="Grade 4: Full Health Spawners",
            spawn_cooldown_mult=999.0,  # Still no enemies!
            max_enemies_mult=0.0,  # STILL NO ENEMIES
            spawner_health_mult=0.75,  # Strong (75 HP, 7-8 hits)
            enemy_speed_mult=0.5,  # N/A
            shaping_scale_mult=5.0,  # Moderate guidance
            damage_penalty_mult=0.5,  # Moderate penalty
            # Advancement: Kill 1.5+ spawners per episode
            min_spawner_kill_rate=1.5,  # Efficient spawner destruction
            min_win_rate=0.0,  # No wins yet
            min_survival_steps=500,  # Reasonable time
            max_survival_steps=2300,  # Efficient play
            min_damage_dealt=100.0,  # High damage
            min_episodes=200,
        ),
        # =====================================================================
        # GRADE 5: INTRODUCE ENEMIES (Slow & Weak)
        # =====================================================================
        # Goal: Maintain spawner focus while avoiding enemies
        CurriculumStage(
            name="Grade 5: Introduce Enemies",
            spawn_cooldown_mult=4.0,  # VERY slow enemy spawns
            max_enemies_mult=0.25,  # FEW enemies (25%)
            spawner_health_mult=0.75,  # Strong (75 HP)
            enemy_speed_mult=0.5,  # SLOW enemies (50% speed)
            shaping_scale_mult=4.0,  # Moderate guidance
            damage_penalty_mult=0.7,  # Moderate penalty
            # Advancement: Must maintain spawner kills with enemies
            min_spawner_kill_rate=1.2,  # Kill 1.2+ spawners/episode
            min_win_rate=0.0,  # No wins required
            min_survival_steps=550,  # Survive with enemies
            max_survival_steps=2200,  # Reasonable pace
            min_enemy_kill_rate=1.0,  # Some enemy kills
            min_damage_dealt=110.0,  # Good damage
            min_episodes=200,
        ),
        # =====================================================================
        # GRADE 6: MORE ENEMIES, FASTER KILLS
        # =====================================================================
        # Goal: Kill spawners faster with more enemies present
        CurriculumStage(
            name="Grade 6: Fast Kills with Enemies",
            spawn_cooldown_mult=2.5,  # Faster spawns
            max_enemies_mult=0.45,  # More enemies (45%)
            spawner_health_mult=0.85,  # Harder spawners (85 HP)
            enemy_speed_mult=0.65,  # Moderate speed (65%)
            shaping_scale_mult=2.5,  # Less guidance
            damage_penalty_mult=0.85,  # Higher penalty
            # Advancement: Kill 1.8+ spawners with enemies
            min_spawner_kill_rate=1.8,  # Kill 1.8+ spawners/episode
            min_win_rate=0.05,  # Some wins required (5%)
            min_survival_steps=600,  # Good survival
            max_survival_steps=1900,  # Must be efficient
            min_enemy_kill_rate=2.5,  # More enemy kills
            min_damage_dealt=150.0,  # High damage
            min_episodes=200,
        ),
        # =====================================================================
        # GRADE 7: ELITE PERFORMANCE (Full Game)
        # =====================================================================
        # Goal: Master-level performance at full difficulty
        CurriculumStage(
            name="Grade 7: Elite Performance",
            spawn_cooldown_mult=1.0,  # Full spawn rate
            max_enemies_mult=1.0,  # Maximum enemies (100%)
            spawner_health_mult=1.0,  # Full spawner health (100 HP)
            enemy_speed_mult=1.0,  # Full enemy speed
            shaping_scale_mult=1.0,  # Minimal shaping
            damage_penalty_mult=1.0,  # Standard penalty
            # Final stage - elite performance
            min_spawner_kill_rate=2.5,  # Excellent spawner destruction
            min_win_rate=0.20,  # Good win rate (20%)
            min_survival_steps=650,  # Good survival
            max_survival_steps=1750,  # Fast, efficient wins
            min_enemy_kill_rate=8.0,  # High kill rate
            min_damage_dealt=250.0,  # Excellent damage
            max_damage_taken=140.0,  # Good defense
            min_episodes=300,
        ),
    ]


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
