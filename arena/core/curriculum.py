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
        
        # All criteria must be met
        criteria_met = (
            spawner_kill_rate >= stage.min_spawner_kill_rate and
            win_rate >= stage.min_win_rate and
            avg_survival >= stage.min_survival_steps
        )
        
        return criteria_met
    
    def get_name(self) -> str:
        stage = self.curriculum_manager.current_stage
        return (f"StageBased(spawners>={stage.min_spawner_kill_rate}, "
                f"winrate>={stage.min_win_rate}, survival>={stage.min_survival_steps})")


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
    min_episodes: int = 50                # Minimum episodes before advancement check
    
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
        # Note: strategy will be set by CurriculumManager to use StageBasedStrategy


def get_default_stages() -> List[CurriculumStage]:
    """Return default 5-stage curriculum progression with gradually increasing difficulty."""
    return [
        CurriculumStage(
            name="Beginner",
            spawn_cooldown_mult=2.0,
            max_enemies_mult=0.4,
            spawner_health_mult=0.5,
            enemy_speed_mult=0.8,
            shaping_scale_mult=3.0,
            damage_penalty_mult=0.5, 
            # Stage 0 -> 1: Easy requirements
            min_spawner_kill_rate=0.5,    
            min_win_rate=0.0,            
            min_survival_steps=300,      
            min_episodes=50,            
        ),
        CurriculumStage(
            name="Easy",
            spawn_cooldown_mult=1.5,
            max_enemies_mult=0.6,
            spawner_health_mult=0.7,
            enemy_speed_mult=0.9,
            shaping_scale_mult=2.0,
            damage_penalty_mult=1.0,  
            # Stage 1 -> 2: Moderate requirements
            min_spawner_kill_rate=1.0,   
            min_win_rate=0.05,          
            min_survival_steps=500,      
            min_episodes=100,           
        ),
        CurriculumStage(
            name="Medium",
            spawn_cooldown_mult=1.25,
            max_enemies_mult=0.8,
            spawner_health_mult=0.85,
            enemy_speed_mult=0.95,
            shaping_scale_mult=1.5,
            damage_penalty_mult=1.2,  
            # Stage 2 -> 3: Getting harder
            min_spawner_kill_rate=1.5,   
            min_win_rate=0.2,           
            min_survival_steps=750,      
            min_episodes=150,
        ),
        CurriculumStage(
            name="Hard",
            spawn_cooldown_mult=1.1,
            max_enemies_mult=0.9,
            spawner_health_mult=0.95,
            enemy_speed_mult=1.0,
            shaping_scale_mult=1.2,
            damage_penalty_mult=1.5,
            # Stage 3 -> 4: Challenging requirements
            min_spawner_kill_rate=2.0,   
            min_win_rate=0.3,           
            min_survival_steps=1000,     
            min_episodes=200,
        ),
        CurriculumStage(
            name="Expert",
            spawn_cooldown_mult=1.0,
            max_enemies_mult=1.0,
            spawner_health_mult=1.0,
            enemy_speed_mult=1.0,
            shaping_scale_mult=1.0,
            damage_penalty_mult=1.7,  
            # Final stage - no advancement needed
            min_spawner_kill_rate=3.0, 
            min_win_rate=0.5,
            min_survival_steps=2000,
            min_episodes=300,
        ),
        CurriculumStage(
            name="Master",
            spawn_cooldown_mult=1.0,
            max_enemies_mult=1.0,
            spawner_health_mult=1.0,
            enemy_speed_mult=1.0,
            shaping_scale_mult=0.8,
            damage_penalty_mult=1.8,  
            # Final stage - no advancement needed
            min_spawner_kill_rate=3.0, 
            min_win_rate=0.8,
            min_survival_steps=2000,
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
