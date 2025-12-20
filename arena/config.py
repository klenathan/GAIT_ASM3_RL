"""
Configuration file for Deep RL Arena
Contains all game parameters, reward structure, and neural network hyperparameters
"""

import numpy as np

# ============================================================================
# GAME SETTINGS
# ============================================================================

# Display
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
FPS = 60

# Game area (keep entities within this zone)
GAME_WIDTH = 1000
GAME_HEIGHT = 800

# Colors (RGB)
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

# ============================================================================
# ENTITY PARAMETERS
# ============================================================================

# Player
PLAYER_RADIUS = 15
PLAYER_MAX_HEALTH = 100
PLAYER_SPEED = 5.0  # For directional movement (style 2)
PLAYER_ROTATION_SPEED = 5.0  # Degrees per frame (style 1)
PLAYER_THRUST = 0.3  # Acceleration (style 1)
PLAYER_MAX_VELOCITY = 6.0  # Maximum velocity magnitude
PLAYER_FRICTION = 0.97  # Velocity decay
PLAYER_SHOOT_COOLDOWN = 10  # Frames between shots

# Enemy
ENEMY_RADIUS = 12
ENEMY_HEALTH = 30
ENEMY_SPEED = 2.5
ENEMY_DAMAGE = 10  # Damage on collision with player
ENEMY_SHOOT_COOLDOWN = 60  # Frames between enemy shots
ENEMY_SHOOT_PROBABILITY = 0.3  # Chance to shoot when cooldown ready

# Spawner
SPAWNER_RADIUS = 25
SPAWNER_HEALTH = 100
SPAWNER_SPAWN_COOLDOWN = 120  # Frames between spawns
SPAWNER_MAX_ENEMIES = 8  # Max enemies per spawner

# Projectile
PROJECTILE_RADIUS = 3
PROJECTILE_SPEED = 8.0
PROJECTILE_DAMAGE = 10
PROJECTILE_LIFETIME = 120  # Frames before auto-destroy

# ============================================================================
# PHASE SYSTEM
# ============================================================================

# Phase configuration: [num_spawners, enemy_speed_multiplier, spawn_rate_multiplier]
PHASE_CONFIG = [
    {"spawners": 1, "enemy_speed_mult": 1.0, "spawn_rate_mult": 1.0},
    {"spawners": 2, "enemy_speed_mult": 1.1, "spawn_rate_mult": 0.9},
    {"spawners": 2, "enemy_speed_mult": 1.2, "spawn_rate_mult": 0.85},
    {"spawners": 3, "enemy_speed_mult": 1.3, "spawn_rate_mult": 0.8},
    {"spawners": 3, "enemy_speed_mult": 1.4, "spawn_rate_mult": 0.75},
]

MAX_PHASES = len(PHASE_CONFIG)

# ============================================================================
# EPISODE SETTINGS
# ============================================================================

MAX_STEPS = 3000  # Maximum steps per episode
STEP_REWARD = 0.1  # Small reward for surviving each step

# ============================================================================
# REWARD STRUCTURE
# ============================================================================

REWARD_ENEMY_DESTROYED = 10.0
REWARD_SPAWNER_DESTROYED = 50.0
REWARD_PHASE_COMPLETE = 100.0
REWARD_DAMAGE_TAKEN = -5.0
REWARD_DEATH = -100.0
REWARD_STEP_SURVIVAL = 0.1

# Shaping rewards (optional but encouraged)
REWARD_APPROACH_SPAWNER = 1.0  # For getting closer to nearest spawner
REWARD_QUICK_SPAWNER_KILL = 5.0  # Bonus for destroying spawner efficiently

# ============================================================================
# OBSERVATION SPACE (14 dimensions)
# ============================================================================

OBS_DIM = 14
"""
Observation vector components:
0: Player X position (normalized 0-1)
1: Player Y position (normalized 0-1)
2: Player velocity X (normalized -1 to 1)
3: Player velocity Y (normalized -1 to 1)
4: Player rotation angle (normalized 0-1, for style 1)
5: Player health (normalized 0-1)
6: Current phase (normalized 0-1)
7: Nearest enemy distance (normalized 0-1)
8: Nearest enemy angle (relative, -pi to pi, normalized)
9: Nearest enemy exists (binary 0 or 1)
10: Nearest spawner distance (normalized 0-1)
11: Nearest spawner angle (relative, -pi to pi, normalized)
12: Nearest spawner exists (binary 0 or 1)
13: Number of active enemies (normalized 0-1)
"""

# ============================================================================
# ACTION SPACES
# ============================================================================

# Control Style 1: Rotation + Thrust
ACTION_SPACE_STYLE_1 = 5
ACTIONS_STYLE_1 = {
    0: "NONE",
    1: "THRUST",
    2: "ROTATE_LEFT",
    3: "ROTATE_RIGHT",
    4: "SHOOT"
}

# Control Style 2: Directional Movement
ACTION_SPACE_STYLE_2 = 6
ACTIONS_STYLE_2 = {
    0: "NONE",
    1: "UP",
    2: "DOWN",
    3: "LEFT",
    4: "RIGHT",
    5: "SHOOT"
}

# ============================================================================
# NEURAL NETWORK ARCHITECTURE
# ============================================================================

# DQN Architecture
DQN_HIDDEN_LAYERS = [256, 128, 64]
DQN_ACTIVATION = "relu"

# PPO Architecture
PPO_NET_ARCH = [dict(pi=[256, 128, 64], vf=[256, 128, 64])]
PPO_ACTIVATION = "relu"

# ============================================================================
# TRAINING HYPERPARAMETERS - DQN (CPU/Single Environment)
# ============================================================================

DQN_HYPERPARAMS = {
    "learning_rate": 3e-4,
    "buffer_size": 100_000,
    "batch_size": 64,
    "gamma": 0.99,
    "exploration_fraction": 0.5,  # 50% of training for exploration
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.05,
    "target_update_interval": 1000,
    "train_freq": 4,
    "gradient_steps": 1,
    "learning_starts": 1000,
    "verbose": 1,
}

# DQN GPU-Optimized Hyperparameters (larger batches for better GPU utilization)
DQN_HYPERPARAMS_GPU = {
    "learning_rate": 3e-4,
    "buffer_size": 200_000,  # Increased buffer
    "batch_size": 256,  # 4x larger for GPU
    "gamma": 0.99,
    "exploration_fraction": 0.5,
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.05,
    "target_update_interval": 1000,
    "train_freq": 4,
    "gradient_steps": 2,  # More gradient steps with larger batch
    "learning_starts": 2000,  # More warmup for larger buffer
    "verbose": 1,
}

# ============================================================================
# TRAINING HYPERPARAMETERS - PPO (CPU/Single Environment)
# ============================================================================

PPO_HYPERPARAMS = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,  # Entropy coefficient for exploration
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "verbose": 1,
}

# PPO GPU-Optimized Hyperparameters (larger batches for better GPU utilization)
PPO_HYPERPARAMS_GPU = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 256,  # 4x larger for GPU
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "verbose": 1,
}

# ============================================================================
# PARALLEL ENVIRONMENTS CONFIGURATION
# ============================================================================

# Number of parallel environments to use by default for each device type
NUM_ENVS_DEFAULT_MPS = 4   # Mac Silicon (MPS) - balanced for memory
NUM_ENVS_DEFAULT_CUDA = 8  # NVIDIA GPU - can handle more environments
NUM_ENVS_DEFAULT_CPU = 2   # CPU - limited parallelization

# ============================================================================
# TENSORBOARD & LOGGING
# ============================================================================

TENSORBOARD_LOG_DIR = "./logs"
MODEL_SAVE_DIR = "./models"
CHECKPOINT_FREQ = 10_000  # Save model every N steps
