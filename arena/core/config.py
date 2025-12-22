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

SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
FPS = 60

GAME_WIDTH = 1000
GAME_HEIGHT = 800

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
ENEMY_HEALTH = 30
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
    {"spawners": 2, "enemy_speed_mult": 1.1, "spawn_rate_mult": 0.9},
    {"spawners": 2, "enemy_speed_mult": 1.2, "spawn_rate_mult": 0.85},
    {"spawners": 3, "enemy_speed_mult": 1.3, "spawn_rate_mult": 0.8},
    {"spawners": 3, "enemy_speed_mult": 1.4, "spawn_rate_mult": 0.75},
]
MAX_PHASES = len(PHASE_CONFIG)

# Reward Structure
MAX_STEPS = 3000
STEP_REWARD = 0.1
REWARD_ENEMY_DESTROYED = 5.0
REWARD_SPAWNER_DESTROYED = 150.0
REWARD_PHASE_COMPLETE = 200.0
REWARD_DAMAGE_TAKEN = -2.0
REWARD_DEATH = -150.0
REWARD_STEP_SURVIVAL = 0.0
REWARD_HIT_ENEMY = 2.0
REWARD_HIT_SPAWNER = 10.0
REWARD_SHOT_FIRED = 0.0

# Reward Shaping
SHAPING_MODE = "delta"
SHAPING_SCALE = 3.0
SHAPING_CLIP = 0.3

# Parallel Environments
NUM_ENVS_DEFAULT_MPS = 4
NUM_ENVS_DEFAULT_CUDA = 20
NUM_ENVS_DEFAULT_CPU = 12

# Storage
TENSORBOARD_LOG_DIR = "./logs"
MODEL_SAVE_DIR = "./models"
CHECKPOINT_FREQ = 10_000

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
    save_vecnormalize: bool = False
    tensorboard_log_dir: str = TENSORBOARD_LOG_DIR
    model_save_dir: str = MODEL_SAVE_DIR
    
    # DQN specific
    dqn_hidden_layers: List[int] = field(default_factory=lambda: [256, 128, 64])
    dqn_activation: str = "SiLU"
    
    # PPO specific
    ppo_net_arch: List[Dict[str, List[int]]] = field(default_factory=lambda: [dict(pi=[256, 128, 64], vf=[256, 128, 64])])
    ppo_activation: str = "SiLU"
    
    # LSTM specific
    ppo_lstm_net_arch: List[int] = field(default_factory=lambda: [128, 64])
    ppo_lstm_hidden_size: int = 128
    ppo_lstm_n_layers: int = 1

@dataclass
class DQNHyperparams:
    learning_rate: float = 3e-4
    buffer_size: int = 100_000
    batch_size: int = 64
    gamma: float = 0.99
    exploration_fraction: float = 0.5
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.05
    target_update_interval: int = 1000
    train_freq: int = 4
    gradient_steps: int = 1
    learning_starts: int = 1000
    verbose: int = 1

@dataclass
class PPOHyperparams:
    learning_rate: float = 3e-5
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.03
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    verbose: int = 1

@dataclass
class PPOLSTMHyperparams(PPOHyperparams):
    learning_rate: float = 1e-4
    n_steps: int = 512
    batch_size: int = 32
    n_epochs: int = 5
    ent_coef: float = 0.01

# Default hyperparameter instances (equivalent to old config)
DQN_DEFAULT = DQNHyperparams()
DQN_GPU_DEFAULT = DQNHyperparams(
    buffer_size=200_000,
    batch_size=256,
    gradient_steps=2,
    learning_starts=2000
)

PPO_DEFAULT = PPOHyperparams()
PPO_GPU_DEFAULT = PPOHyperparams(batch_size=256)

PPO_LSTM_DEFAULT = PPOLSTMHyperparams()
PPO_LSTM_GPU_DEFAULT = PPOLSTMHyperparams(batch_size=64)
