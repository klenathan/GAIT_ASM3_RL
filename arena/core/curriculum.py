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


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class CurriculumStage:
    """Defines difficulty modifiers for a curriculum stage."""
    name: str
    spawn_cooldown_mult: float = 1.0   # Higher = slower spawns
    max_enemies_mult: float = 1.0      # Lower = fewer enemies
    spawner_health_mult: float = 1.0   # Lower = easier to kill
    enemy_speed_mult: float = 1.0      # Lower = slower enemies
    shaping_scale_mult: float = 1.0    # Higher = stronger guidance
    
    def __repr__(self):
        return f"Stage({self.name})"


@dataclass
class CurriculumMetrics:
    """Tracks episode outcomes for advancement decisions."""
    spawner_kills: List[float] = field(default_factory=list)
    wins: List[int] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    episode_rewards: List[float] = field(default_factory=list)
    
    def record_episode(self, spawners_killed: int, won: bool, length: int, reward: float):
        self.spawner_kills.append(float(spawners_killed))
        self.wins.append(1 if won else 0)
        self.episode_lengths.append(length)
        self.episode_rewards.append(reward)
        
        # Keep only last 200 episodes to limit memory
        for lst in [self.spawner_kills, self.wins, self.episode_lengths, self.episode_rewards]:
            if len(lst) > 200:
                lst.pop(0)


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning."""
    enabled: bool = True
    stages: List[CurriculumStage] = field(default_factory=list)
    strategy: Optional[AdvancementStrategy] = None
    
    def __post_init__(self):
        if not self.stages:
            self.stages = get_default_stages()
        if self.strategy is None:
            self.strategy = SpawnerKillRateStrategy(threshold=0.3)


def get_default_stages() -> List[CurriculumStage]:
    """Return default 5-stage curriculum progression."""
    return [
        CurriculumStage(
            name="Beginner",
            spawn_cooldown_mult=2.0,
            max_enemies_mult=0.4,
            spawner_health_mult=0.5,
            enemy_speed_mult=0.8,
            shaping_scale_mult=3.0,
        ),
        CurriculumStage(
            name="Easy",
            spawn_cooldown_mult=1.5,
            max_enemies_mult=0.6,
            spawner_health_mult=0.7,
            enemy_speed_mult=0.9,
            shaping_scale_mult=2.0,
        ),
        CurriculumStage(
            name="Medium",
            spawn_cooldown_mult=1.25,
            max_enemies_mult=0.8,
            spawner_health_mult=0.85,
            enemy_speed_mult=0.95,
            shaping_scale_mult=1.5,
        ),
        CurriculumStage(
            name="Hard",
            spawn_cooldown_mult=1.1,
            max_enemies_mult=0.9,
            spawner_health_mult=0.95,
            enemy_speed_mult=1.0,
            shaping_scale_mult=1.2,
        ),
        CurriculumStage(
            name="Expert",
            spawn_cooldown_mult=1.0,
            max_enemies_mult=1.0,
            spawner_health_mult=1.0,
            enemy_speed_mult=1.0,
            shaping_scale_mult=1.0,
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
    
    def record_episode(self, spawners_killed: int, won: bool, length: int, reward: float):
        """Record episode outcome for advancement evaluation."""
        self.metrics.record_episode(spawners_killed, won, length, reward)
    
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
