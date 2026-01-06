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

# Observation Space Layout (44 dims total):
# [0-1]   Player position (x, y)
# [2-3]   Player velocity (vx, vy)
# [4]     Player rotation
# [5]     Player health ratio
# [6]     Shoot cooldown ratio (0 = ready)
# [7]     Current phase ratio
# [8]     Spawners remaining ratio
# [9]     Time remaining ratio
# [10-12] Nearest enemy 1 (dist, angle, exists)
# [13-15] Nearest enemy 2 (dist, angle, exists)
# [16-19] Nearest spawner 1 (dist, angle, exists, health)
# [20-23] Nearest spawner 2 (dist, angle, exists, health)
# [24-26] Nearest projectile 1 (dist, angle, exists)
# [27-29] Nearest projectile 2 (dist, angle, exists)
# [30-32] Nearest projectile 3 (dist, angle, exists)
# [33-35] Nearest projectile 4 (dist, angle, exists)
# [36-38] Nearest projectile 5 (dist, angle, exists)
# [39-42] Wall distances (left, right, top, bottom)
# [43]    Enemy count
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
PROJECTILE_LIFETIME = 600

# Phase System
PHASE_CONFIG = [
    {"spawners": 1, "enemy_speed_mult": 1.0, "spawn_rate_mult": 0.8},
    {"spawners": 1, "enemy_speed_mult": 0.9, "spawn_rate_mult": 0.9},
    {"spawners": 2, "enemy_speed_mult": 0.8, "spawn_rate_mult": 0.3},
    {"spawners": 3, "enemy_speed_mult": 0.7, "spawn_rate_mult": 0.2},
    {"spawners": 4, "enemy_speed_mult": 0.75, "spawn_rate_mult": 0.1},
]
MAX_PHASES = len(PHASE_CONFIG)

# Reward Structure
MAX_STEPS = 3000

MAX_STEPS = 3000
REWARD_SPAWNER_DESTROYED = 0.9
REWARD_PHASE_COMPLETE = 0.0
REWARD_DAMAGE_TAKEN = -0.01
REWARD_DEATH = -1.0
REWARD_STEP_SURVIVAL = -0.01
REWARD_HIT_ENEMY = 0.01
REWARD_HIT_SPAWNER = 0.2
REWARD_ENEMY_DESTROYED = 0.01
REWARD_SHOT_FIRED = 0.0
REWARD_QUICK_SPAWNER_KILL = 0.2
REWARD_AIMING = 0.01  # Reward for pointing at an enemy/spawner
AIMING_ANGLE_THRESHOLD = 0.1  # Radians (approx 11 degrees)
REWARD_ACCURATE_SHOT = 0.1  # Reward for shooting while aiming correctly
REWARD_APPROACH_SPAWNER = 0.0  # Reward for moving closer to spawners
REWARD_STYLE2_ALIGNMENT_BONUS = 0.8  # Extra reward for Style 2 alignment

# Activity Penalties (discourage passive/corner-hiding play)
PENALTY_INACTIVITY = 0  # Per-step penalty when not moving enough
PENALTY_CORNER = -0.1  # Per-step penalty when too close to edges
CORNER_MARGIN = 80  # Distance from edge to be considered "in corner"
PENALTY_WALL_PROXIMITY = -0.1  # Penalty for being too close to walls
WALL_MARGIN = 50  # Distance from wall to trigger penalty
INACTIVITY_VELOCITY_THRESHOLD = 0.5  # Minimum velocity magnitude to be "active"

# Reward Shaping
SHAPING_MODE = "delta"  # Simpler selection
SHAPING_SCALE = 1.0
SHAPING_CLIP = 0.5

# Curriculum Learning
CURRICULUM_ENABLED = True
CURRICULUM_MODE = "auto"  # "manual" or "auto"
AUTO_CURRICULUM_STAGES = 10  # Number of stages for automated curriculum
CURRICULUM_ADVANCEMENT_THRESHOLD = 1  # Spawner kill rate to advance
CURRICULUM_MIN_EPISODES = 100  # Min episodes before advancing
CURRICULUM_WINDOW = 100  # Episodes for averaging

# Parallel Environments
NUM_ENVS_DEFAULT_MPS = 20
NUM_ENVS_DEFAULT_CUDA = 20
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
    save_replay_buffer: bool = False
    save_vecnormalize: bool = True
    runs_dir: str = RUNS_DIR  # Unified directory for models and logs
    # Optional resume/transfer learning
    pretrained_model_path: Optional[str] = None
    # Whether to reset timesteps in SB3 learn(); when resuming, typically False
    reset_num_timesteps: bool = True
    # Transfer learning fine-grained control
    load_vecnormalize: bool = True  # Load VecNormalize stats if available
    load_curriculum: bool = True  # Restore curriculum progress
    load_replay_buffer: bool = True  # Load replay buffer (DQN only)

    # Learning rate schedule
    lr_schedule: str = "cosine"  # "constant", "linear", "exponential", "cosine"
    lr_end: Optional[float] = 1e-5  # Final LR; defaults to start_lr * 0.1
    # Fraction of training for warmup (0 = none)
    lr_warmup_fraction: float = 0.0

    # DQN specific
    dqn_hidden_layers: List[int] = field(default_factory=lambda: [256, 128, 64])
    dqn_activation: str = "SiLU"

    # PPO specific
    ppo_net_arch: Dict[str, List[int]] = field(
        default_factory=lambda: dict(pi=[256, 128, 64], vf=[256, 128, 64])
    )
    ppo_activation: str = "SiLU"

    # LSTM specific
    ppo_lstm_net_arch: List[int] = field(default_factory=lambda: [128, 64])
    ppo_lstm_hidden_size: int = 128
    ppo_lstm_n_layers: int = 1

    # A2C specific
    a2c_net_arch: Dict[str, List[int]] = field(
        default_factory=lambda: dict(
            pi=[384, 256, 256, 128, 64], vf=[256, 128, 128, 64]
        )
    )
    a2c_activation: str = "SiLU"

    # Scaled DQN specific architecture configuration
    scaled_dqn_shared_layers: List[int] = field(
        default_factory=lambda: [512, 512, 384, 384, 256]
    )
    scaled_dqn_value_layers: List[int] = field(default_factory=lambda: [256, 128])
    scaled_dqn_advantage_layers: List[int] = field(default_factory=lambda: [256, 128])
    scaled_dqn_activation: str = "SiLU"
    scaled_dqn_use_layer_norm: bool = True
    scaled_dqn_use_residual: bool = True


@dataclass
class DQNHyperparams:
    learning_rate: float = 3e-4
    # buffer_size: int = 100_000
    buffer_size: int = 500_000
    batch_size: int = 64
    gamma: float = 0.99
    exploration_fraction: float = 0.2
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.05
    target_update_interval: int = 1000
    train_freq: int = 4
    gradient_steps: int = 1
    learning_starts: int = 1000
    verbose: int = 1


@dataclass
class PPOHyperparams:
    learning_rate: float = 5e-4
    n_steps: int = 4096
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.1
    ent_coef: float = 0.05
    vf_coef: float = 1.0
    max_grad_norm: float = 0.5
    target_kl: float = 0.03
    verbose: int = 0


@dataclass
class A2CHyperparams:
    learning_rate: float = 2.5e-4
    n_steps: int = 128
    gamma: float = 0.99
    gae_lambda: float = 1.0
    ent_coef: float = 0.05  # Higher entropy for better exploration
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    rms_prop_eps: float = 1e-5
    use_rms_prop: bool = True
    normalize_advantage: bool = True  # Stabilizes training
    verbose: int = 1


@dataclass
class PPOLSTMHyperparams(PPOHyperparams):
    learning_rate: float = 1e-4
    n_steps: int = 512
    batch_size: int = 32
    n_epochs: int = 5
    gamma: float = 0.99
    gae_lambda: float = 0.98
    clip_range: float = 0.2
    ent_coef: float = 0.05
    vf_coef: float = 1.0
    max_grad_norm: float = 0.5
    target_kl: float = 0.03
    verbose: int = 0


# Default hyperparameter instances (equivalent to old config)
DQN_DEFAULT = DQNHyperparams()
DQN_GPU_DEFAULT = DQNHyperparams(
    buffer_size=200_000, batch_size=256, gradient_steps=2, learning_starts=2000
)

PPO_DEFAULT = PPOHyperparams()
PPO_GPU_DEFAULT = PPOHyperparams(batch_size=256)

PPO_LSTM_DEFAULT = PPOLSTMHyperparams()
PPO_LSTM_GPU_DEFAULT = PPOLSTMHyperparams(batch_size=64)

A2C_DEFAULT = A2CHyperparams()
A2C_GPU_DEFAULT = A2CHyperparams(n_steps=256)


# ============================================================================
# SCALED DQN CONFIGURATION
# ============================================================================


@dataclass
class ScaledDQNHyperparams:
    """
    Hyperparameters for Scaled DQN with deep 10-layer dueling architecture.

    These hyperparameters are optimized for deep networks with:
    - 10 layers (5 shared + 3 value + 3 advantage)
    - ~1.2M parameters (6x more than standard DQN)
    - Residual connections and layer normalization
    - Industry-standard values for stable training
    """

    # Learning rate (lower for deep networks)
    learning_rate: float = 1e-4

    # Replay buffer (larger for better sampling diversity)
    buffer_size: int = 1_000_000
    batch_size: int = 256  # Larger batches for gradient stability

    # Discount factor
    gamma: float = 0.99

    # Exploration (epsilon-greedy)
    exploration_fraction: float = 0.15  # 15% of training
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.01  # Lower final epsilon

    # Target network updates
    target_update_interval: int = 2000  # More frequent updates

    # Training frequency
    train_freq: int = 4  # Train every 4 steps
    gradient_steps: int = 2  # Multiple gradient steps per update
    learning_starts: int = 10_000  # More warmup for deep network

    # Gradient clipping (prevent exploding gradients)
    max_grad_norm: float = 10.0

    # Other settings
    verbose: int = 1


# Default instances for CPU and GPU
SCALED_DQN_DEFAULT = ScaledDQNHyperparams()

SCALED_DQN_GPU_DEFAULT = ScaledDQNHyperparams(
    learning_rate=2e-4,  # Higher LR for GPU (larger batches)
    buffer_size=2_000_000,  # Larger buffer on GPU
    batch_size=512,  # Larger batches on GPU
    gradient_steps=4,  # More gradient steps on GPU
    learning_starts=20_000,  # More warmup on GPU
)
