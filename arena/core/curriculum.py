"""
Curriculum Learning System for Arena RL Training.
Uses Strategy pattern for advancement criteria.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Callable
import numpy as np
from arena.core import config


# =============================================================================
# STRATEGY PATTERN: Advancement Criteria
# =============================================================================

class AdvancementStrategy(ABC):
    """Abstract base class for curriculum advancement strategies."""

    @abstractmethod
    def should_advance(self, metrics: 'CurriculumMetrics') -> bool:
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

    def should_advance(self, metrics: 'CurriculumMetrics') -> bool:
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

    def should_advance(self, metrics: 'CurriculumMetrics') -> bool:
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

    def should_advance(self, metrics: 'CurriculumMetrics') -> bool:
        if len(metrics.episode_lengths) < self.min_episodes:
            return False
        avg_length = np.mean(metrics.episode_lengths[-100:])
        return avg_length >= self.threshold_steps

    def get_name(self) -> str:
        return f"SurvivalTime(threshold={self.threshold_steps})"


class CompositeStrategy(AdvancementStrategy):
    """Combine multiple strategies with AND/OR logic."""

    def __init__(self, strategies: List[AdvancementStrategy], require_all: bool = False):
        self.strategies = strategies
        self.require_all = require_all

    def should_advance(self, metrics: 'CurriculumMetrics') -> bool:
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

    def __init__(self, curriculum_manager: 'CurriculumManager'):
        self.curriculum_manager = curriculum_manager

    def should_advance(self, metrics: 'CurriculumMetrics') -> bool:
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
        recent_win_times = [t for t, w in zip(metrics.episode_lengths[-window:],
                                              metrics.wins[-window:]) if w == 1]
        avg_win_time = np.mean(
            recent_win_times) if recent_win_times else 999999

        # All criteria must be met
        criteria_met = (
            spawner_kill_rate >= stage.min_spawner_kill_rate and
            win_rate >= stage.min_win_rate and
            avg_survival >= stage.min_survival_steps and
            avg_survival <= stage.max_survival_steps and  # Not too passive
            avg_enemy_kills >= stage.min_enemy_kill_rate and
            avg_damage_dealt >= stage.min_damage_dealt and
            avg_damage_taken <= stage.max_damage_taken and
            avg_win_time <= stage.max_win_time  # Fast wins required
        )

        return criteria_met

    def get_name(self) -> str:
        stage = self.curriculum_manager.current_stage
        return (f"StageBased(spawners>={stage.min_spawner_kill_rate}, "
                f"wins>={stage.min_win_rate}, enemies>={stage.min_enemy_kill_rate})")


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class CurriculumStage:
    """Defines difficulty modifiers and advancement criteria for a curriculum stage."""
    name: str
    spawn_cooldown_mult: float = 1.0   # Higher = slower spawns
    max_enemies_mult: float = 1.0      # Lower = fewer enemies
    spawner_health_mult: float = 1.0   # Lower = easier to kill
    enemy_speed_mult: float = 1.0      # Lower = slower enemies
    shaping_scale_mult: float = 1.0    # Higher = stronger guidance
    damage_penalty_mult: float = 1.0   # Higher = more severe damage penalties

    # Advancement criteria (configurable per stage)
    min_spawner_kill_rate: float = 0.3    # Required avg spawner kills per episode
    min_win_rate: float = 0.0             # Required win rate to advance
    min_survival_steps: int = 500         # Required avg episode length
    # Maximum avg episode length (penalizes passivity)
    max_survival_steps: int = 999999
    min_enemy_kill_rate: float = 0.0     # Required avg enemy kills per episode
    min_damage_dealt: float = 0.0        # Required avg damage dealt per episode
    # Maximum avg damage taken (lower = better defense)
    max_damage_taken: float = 999999.0
    # Maximum avg steps to win (lower = faster wins)
    max_win_time: int = 999999
    min_episodes: int = 50               # Minimum episodes before advancement check

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

    def record_episode(self, spawners_killed: int, won: bool, length: int, reward: float,
                       enemy_kills: int = 0, damage_dealt: float = 0, damage_taken: float = 0):
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
        for lst in [self.spawner_kills, self.wins, self.episode_lengths, self.episode_rewards,
                    self.enemy_kills, self.damage_dealt, self.damage_taken]:
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
            if config.CURRICULUM_MODE == "auto":
                self.stages = get_auto_stages(config.AUTO_CURRICULUM_STAGES)
            else:
                self.stages = get_manual_stages()
        # Note: strategy will be set by CurriculumManager to use StageBasedStrategy


def get_auto_stages(num_stages: int = 10) -> List[CurriculumStage]:
    """Generate automated curriculum stages by interpolating difficulty."""
    stages = []

    # Difficulty parameters (Start -> End)
    # Start: Very easy, slow enemies, high guidance
    # End: Full difficulty, fast enemies, no guidance

    for i in range(num_stages):
        progress = i / (num_stages - 1) if num_stages > 1 else 1.0

        # Interpolate parameters
        spawn_cooldown = 3.0 - (2.0 * progress)      # 3.0 -> 1.0
        max_enemies = 0.2 + (0.8 * progress)         # 0.2 -> 1.0
        spawner_health = 0.3 + (0.7 * progress)      # 0.3 -> 1.0
        enemy_speed = 0.2 + (0.8 * progress)         # 0.2 -> 1.0
        shaping_scale = 4.0 - (3.5 * progress)       # 4.0 -> 0.5
        damage_penalty = 0.1 + (0.9 * progress)      # 0.1 -> 1.0

        # Advancement criteria (Start -> End)
        min_spawner_kills = 0.1 + (2.9 * progress)   # 0.1 -> 3.0
        min_win_rate = 0.0 + (0.6 * progress)        # 0.0 -> 0.6
        min_enemy_kills = 0.0 + (15.0 * progress)    # 0.0 -> 15.0

        stages.append(CurriculumStage(
            name=f"Auto Stage {i+1}/{num_stages}",
            spawn_cooldown_mult=spawn_cooldown,
            max_enemies_mult=max_enemies,
            spawner_health_mult=spawner_health,
            enemy_speed_mult=enemy_speed,
            shaping_scale_mult=shaping_scale,
            damage_penalty_mult=damage_penalty,
            min_spawner_kill_rate=min_spawner_kills,
            min_win_rate=min_win_rate,
            min_enemy_kill_rate=min_enemy_kills,
            min_survival_steps=400 + int(200 * progress),
            min_episodes=50 + int(50 * progress)
        ))

    return stages


def get_manual_stages() -> List[CurriculumStage]:
    """Return default handcrafted curriculum progression."""
    return [
        # Grade 1: Survival Basics
        # Behavior Focus: Basic movement, staying alive, avoiding damage
        # Low enemy count, slow enemies, reduced damage penalties, high shaping guidance
        CurriculumStage(
            name="Grade 1: Survival Basics",
            spawn_cooldown_mult=2.5,      # Very slow spawns - less pressure
            max_enemies_mult=0.3,         # Very few enemies - focus on survival
            # Easy spawners (but not the focus yet)
            spawner_health_mult=0.4,
            enemy_speed_mult=0.2,         # Slow enemies - easier to avoid
            shaping_scale_mult=4.0,       # High guidance for basic movement
            damage_penalty_mult=0.3,      # Low damage penalty - encourage exploration
            # Advancement: Must survive consistently
            min_spawner_kill_rate=0.1,    # Spawner kills not required
            min_win_rate=0.0,             # Wins not required
            min_survival_steps=400,       # Must survive ~400 steps consistently
            min_episodes=50,
        ),

        # Grade 2: Enemy Elimination
        # Behavior Focus: Targeting and destroying enemies
        # More enemies, moderate speed, emphasis on enemy kills
        CurriculumStage(
            name="Grade 2: Enemy Elimination",
            spawn_cooldown_mult=2.0,      # Slower spawns - focus on existing enemies
            max_enemies_mult=0.5,         # More enemies to practice on
            spawner_health_mult=0.5,      # Still easy spawners
            enemy_speed_mult=0.85,        # Moderate speed
            shaping_scale_mult=3.0,       # Good guidance for combat
            damage_penalty_mult=0.6,      # Moderate penalty - learn to fight safely
            # Advancement: Must kill enemies consistently
            min_spawner_kill_rate=0.3,    # Spawners still not focus
            min_win_rate=0.0,             # Wins not required
            min_survival_steps=600,       # Longer survival with combat
            min_enemy_kill_rate=3.0,      # MUST kill at least 3 enemies per episode on average
            min_damage_dealt=50.0,        # Must deal damage to enemies
            min_episodes=75,
        ),

        # Grade 3: Spawner Targeting
        # Behavior Focus: Destroying spawners to progress phases
        # Spawners are primary targets, moderate difficulty
        CurriculumStage(
            name="Grade 3: Spawner Targeting",
            spawn_cooldown_mult=1.8,      # Faster spawns - spawners matter
            max_enemies_mult=0.6,         # More enemies from spawners
            spawner_health_mult=0.65,     # Moderate spawner health
            enemy_speed_mult=0.9,         # Faster enemies
            shaping_scale_mult=2.5,       # Guidance for spawner focus
            damage_penalty_mult=0.9,      # Higher penalty - be careful
            # Advancement: Must kill spawners consistently
            min_spawner_kill_rate=0.8,    # Must kill spawners regularly
            min_win_rate=0.0,             # Wins not required yet
            min_survival_steps=800,       # Good survival while targeting spawners
            min_episodes=100,
        ),

        # Grade 4: Multi-Target Management
        # Behavior Focus: Handling multiple threats simultaneously
        # More enemies, faster spawns, need to balance priorities
        CurriculumStage(
            name="Grade 4: Multi-Target Management",
            spawn_cooldown_mult=1.4,      # Faster spawns - more pressure
            max_enemies_mult=0.75,        # Many enemies at once
            spawner_health_mult=0.8,      # Harder spawners
            enemy_speed_mult=0.95,        # Fast enemies
            shaping_scale_mult=1.8,       # Less guidance - more independent
            damage_penalty_mult=1.2,      # Significant penalty
            # Advancement: Must handle multiple targets and kill spawners
            # Multiple spawner kills (reduced from 1.5)
            min_spawner_kill_rate=1.2,
            # Some wins required (reduced from 0.15)
            min_win_rate=0.10,
            min_survival_steps=750,       # Good survival (reduced from 800)
            max_survival_steps=1500,      # Don't be too passive (more lenient)
            # Must actively fight enemies (reduced from 8.0)
            min_enemy_kill_rate=5.0,
            # Must deal damage (reduced from 150.0)
            min_damage_dealt=100.0,
            min_episodes=150,
        ),

        # Grade 5: Aggressive Combat
        # Behavior Focus: Balanced aggression - fast clears while staying alive
        # Emphasis on combat efficiency and win speed
        CurriculumStage(
            name="Grade 5: Aggressive Combat",
            spawn_cooldown_mult=1.15,     # Fast spawns
            max_enemies_mult=0.85,        # Many enemies
            spawner_health_mult=0.9,      # Strong spawners
            enemy_speed_mult=1.0,         # Full speed
            shaping_scale_mult=1.0,       # Minimal guidance - independent combat
            damage_penalty_mult=1.3,      # Moderate penalty - smart aggression
            # Advancement: Must be aggressive AND efficient
            min_spawner_kill_rate=2.2,    # High spawner kill rate
            min_win_rate=0.35,            # Decent win rate
            min_survival_steps=700,       # Moderate survival (don't hide)
            max_survival_steps=1200,      # Win quickly, don't drag out
            min_enemy_kill_rate=12.0,     # High enemy elimination
            min_damage_dealt=250.0,       # High damage output
            max_damage_taken=150.0,       # Good defense despite aggression
            max_win_time=1000,            # Fast wins (when winning)
            min_episodes=200,
        ),

        # Grade 6: Elite Performance
        # Behavior Focus: Speed, precision, efficiency - win fast and clean
        # Full difficulty, requires aggressive play with excellent execution
        CurriculumStage(
            name="Grade 6: Elite Performance",
            spawn_cooldown_mult=1.0,      # Full spawn rate
            max_enemies_mult=1.0,         # Maximum enemies
            spawner_health_mult=1.0,      # Full spawner health
            enemy_speed_mult=1.0,         # Full enemy speed
            shaping_scale_mult=0.5,       # Minimal shaping - pure skill
            damage_penalty_mult=1.0,      # Standard penalty
            # Final stage - elite combat performance required
            min_spawner_kill_rate=3.0,    # Excellent spawner destruction
            min_win_rate=0.6,             # High win rate
            min_survival_steps=600,       # Don't need long survival - win fast!
            max_survival_steps=950,       # Must win quickly, no passive play
            min_enemy_kill_rate=15.0,     # Very high kill rate
            min_damage_dealt=350.0,       # Excellent damage output
            max_damage_taken=120.0,       # Excellent defense
            max_win_time=850,             # Fast, efficient wins required
            min_episodes=250,
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
        self._advancement_callbacks: List[Callable[[
            int, CurriculumStage], None]] = []

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

    def record_episode(self, spawners_killed: int, won: bool, length: int, reward: float,
                       enemy_kills: int = 0, damage_dealt: float = 0, damage_taken: float = 0):
        """Record episode outcome for advancement evaluation."""
        self.metrics.record_episode(spawners_killed, won, length, reward,
                                    enemy_kills, damage_dealt, damage_taken)

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
            "strategy": self.config.strategy.get_name() if self.config.strategy else "None",
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
            }
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
