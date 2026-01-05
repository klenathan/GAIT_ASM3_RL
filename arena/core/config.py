"""
Configuration for Deep RL Arena.
Contains game parameters, reward structure, and structured trainer configurations.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
import numpy as np

# ============================================================================
# GAME SETTINGS
# ============================================================================

GAME_WIDTH = 1600
GAME_HEIGHT = 1280

# Visual Scaling for smaller screens
RENDER_SCALE = 0.6  # Scale down visuals only (logic stays 1600x1280)
WINDOW_GAME_WIDTH = int(GAME_WIDTH * RENDER_SCALE)
WINDOW_GAME_HEIGHT = int(GAME_HEIGHT * RENDER_SCALE)
SIDEBAR_WIDTH = 300

SCREEN_WIDTH = WINDOW_GAME_WIDTH + SIDEBAR_WIDTH
SCREEN_HEIGHT = WINDOW_GAME_HEIGHT
FPS = 60

# Action and Observation Spaces
ACTION_SPACE_STYLE_1 = 5  # Rotation, Thrust, Shoot
ACTION_SPACE_STYLE_2 = 6  # Directional (4), Shoot

# Observation Space Layout (44 dims total) - Agent-Relative Coordinates:
# [0]     Distance from arena center (normalized to max_dist)
# [1]     Angle to arena center (normalized 0-1 from -π to π)
# [2]     Velocity magnitude (speed, normalized to max velocity)
# [3]     Velocity direction angle (absolute world angle, normalized 0-1)
# [4]     Player rotation (normalized)
# [5]     Player health ratio
# [6]     Shoot cooldown ratio (0 = ready)
# [7]     Current phase ratio
# [8]     Spawners remaining ratio
# [9]     Time remaining ratio
# [10-12] Nearest enemy 1 (dist, angle relative to player rotation, exists)
# [13-15] Nearest enemy 2 (dist, angle relative to player rotation, exists)
# [16-19] Nearest spawner 1 (dist, angle relative to player rotation, exists, health)
# [20-23] Nearest spawner 2 (dist, angle relative to player rotation, exists, health)
# [24-26] Nearest projectile 1 (dist, angle relative to player rotation, exists)
# [27-29] Nearest projectile 2 (dist, angle relative to player rotation, exists)
# [30-32] Nearest projectile 3 (dist, angle relative to player rotation, exists)
# [33-35] Nearest projectile 4 (dist, angle relative to player rotation, exists)
# [36-38] Nearest projectile 5 (dist, angle relative to player rotation, exists)
# [39-42] Wall distances (left, right, top, bottom) - distance-based
# [43]    Enemy count (normalized)
OBS_DIM = 44

# Threat detection
PROJECTILE_DANGER_RADIUS = 150  # Radius to count nearby projectiles
MIN_ENEMY_SPAWN_DISTANCE = 150  # Minimum distance between spawning enemies


# Colors
COLOR_BACKGROUND = (10, 10, 25)
COLOR_PLAYER = (50, 200, 255)
COLOR_ENEMY = (255, 50, 50)
COLOR_SPAWNER = (200, 50, 200)
COLOR_PROJECTILE = (255, 255, 100)
COLOR_PLAYER_PROJECTILE = (100, 255, 255)
COLOR_TEXT = (255, 255, 255)
COLOR_HEALTH_GOOD = (50, 255, 50)
COLOR_HEALTH_MEDIUM = (255, 200, 50)
COLOR_HEALTH_BAD = (255, 50, 50)
COLOR_PANEL_BG = (20, 20, 40)
COLOR_PANEL_BORDER = (100, 100, 150)

# Model Output Visualization Colors
COLOR_ACTION_BAR_HIGH = (50, 200, 100)  # Green - high probability
COLOR_ACTION_BAR_LOW = (100, 100, 120)  # Dim - low probability
COLOR_VALUE_POSITIVE = (100, 200, 255)  # Blue - positive value
COLOR_VALUE_NEGATIVE = (255, 100, 100)  # Red - negative value
COLOR_ACTION_SELECTED = (255, 255, 100)  # Yellow - selected action
COLOR_ENTROPY_HIGH = (100, 255, 100)  # Green - high entropy (exploring)
COLOR_ENTROPY_LOW = (255, 100, 100)  # Red - low entropy (confident)

# Entity Parameters
PLAYER_RADIUS = 15
PLAYER_MAX_HEALTH = 150
PLAYER_SPEED = 5.0
PLAYER_ROTATION_SPEED = 5.0
PLAYER_THRUST = 0.3
PLAYER_MAX_VELOCITY = 6.0
PLAYER_FRICTION = 0.97
PLAYER_SHOOT_COOLDOWN = 10

ENEMY_RADIUS = 12
ENEMY_HEALTH = 10
ENEMY_SPEED = 2.0
ENEMY_DAMAGE = 10
ENEMY_SHOOT_COOLDOWN = 60
ENEMY_SHOOT_PROBABILITY = 0.3

SPAWNER_RADIUS = 25
SPAWNER_HEALTH = 100
SPAWNER_SPAWN_COOLDOWN = 120
SPAWNER_MAX_ENEMIES = 8

PROJECTILE_RADIUS = 3
PROJECTILE_SPEED = 8.0
PROJECTILE_DAMAGE = 10
PROJECTILE_LIFETIME = 120

# Phase System
PHASE_CONFIG = [
    {"spawners": 1, "enemy_speed_mult": 1.0, "spawn_rate_mult": 1.0},
    {"spawners": 2, "enemy_speed_mult": 0.9, "spawn_rate_mult": 0.9},
    {"spawners": 3, "enemy_speed_mult": 0.8, "spawn_rate_mult": 0.85},
    {"spawners": 4, "enemy_speed_mult": 0.7, "spawn_rate_mult": 0.8},
    {"spawners": 5, "enemy_speed_mult": 0.75, "spawn_rate_mult": 0.75},
]
MAX_PHASES = len(PHASE_CONFIG)

# Reward Structure
MAX_STEPS = 3000

REWARD_ENEMY_DESTROYED = 5.0
REWARD_SPAWNER_DESTROYED = 20.0  # The main jackpot
REWARD_PHASE_COMPLETE = 0.0
REWARD_DAMAGE_TAKEN = -3.0  # Reduced from -50 (too harsh)
REWARD_DEATH = -20.0
REWARD_STEP_SURVIVAL = -0.1  # Slight hunger cost to prevent stalling
REWARD_HIT_ENEMY = 2.0
REWARD_HIT_SPAWNER = 5.0
REWARD_SHOT_FIRED = -0.01
REWARD_QUICK_SPAWNER_KILL = 20.0

# Activity Penalties (discourage passive/corner-hiding play)
PENALTY_INACTIVITY = -0.5  # Per-step penalty when not moving enough
PENALTY_CORNER = -0.1  # Per-step penalty when too close to edges
CORNER_MARGIN = 80  # Distance from edge to be considered "in corner"
INACTIVITY_VELOCITY_THRESHOLD = 0.5  # Minimum velocity magnitude to be "active"

# Reward Shaping
SHAPING_MODE = "delta"  # Simpler selection
SHAPING_SCALE = 1.0
SHAPING_CLIP = 0.2

# Curriculum Learning
CURRICULUM_ENABLED = True
CURRICULUM_ADVANCEMENT_THRESHOLD = 1  # Spawner kill rate to advance
CURRICULUM_MIN_EPISODES = 100  # Min episodes before advancing
CURRICULUM_WINDOW = 100  # Episodes for averaging

# Parallel Environments
NUM_ENVS_DEFAULT_MPS = 64
NUM_ENVS_DEFAULT_CUDA = 1024  # Balanced for Colab T4 (2 vCPUs + 1 GPU)
NUM_ENVS_DEFAULT_CPU = 12

# Storage - Unified directory structure
RUNS_DIR = "./runs"  # Base directory for all training runs
CHECKPOINT_FREQ = 10_000

# Legacy paths (for backward compatibility with old models)
TENSORBOARD_LOG_DIR = "./logs"  # Deprecated: kept for reference only
MODEL_SAVE_DIR = "./models"  # Deprecated: kept for reference only

# ============================================================================
# STRUCTURED CONFIGURATIONS
# ============================================================================

# ============================================================================
# STRUCTURED CONFIGURATIONS
# ============================================================================


@dataclass
class TrainerConfig:
    """Configuration for the training run."""

    algo: str
    style: int
    total_timesteps: int = 100_000
    device: str = "auto"
    num_envs: Optional[int] = None
    render: bool = False
    progress_bar: bool = True
    checkpoint_freq: int = 50_000
    # Unified directory for models and logs
    runs_dir: str = RUNS_DIR
    # Optional resume/transfer learning
    pretrained_model_path: Optional[str] = None

    # Learning rate schedule
    lr_schedule: str = "cosine"  # "constant", "linear", "exponential", "cosine"
    lr_end: Optional[float] = 1e-6  # Final LR; defaults to start_lr * 0.1
    # Fraction of training for warmup (0 = none)
    lr_warmup_fraction: float = 0.0

    # Puffer PPO specific (moved from hardcoded in trainer)
    learning_rate: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    clip_coef: float = 0.1
    update_epochs: int = 4
    num_minibatches: int = 4
    num_steps: int = 128
