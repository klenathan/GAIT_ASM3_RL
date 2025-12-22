# Deep RL Arena

A Pygame-based Deep Reinforcement Learning environment for training intelligent agents to navigate and survive in a dynamic combat arena.

## Features

- **Continuous Action Space Arena**: Real-time combat with enemies, spawners, and projectiles
- **Dual Control Schemes**: 
  - Style 1: Rotation + Thrust (space-like controls)
  - Style 2: Directional Movement (WASD-like)
- **Deep RL Algorithms**: DQN and PPO support via Stable Baselines3
- **Beautiful Visualization**: Real-time training metrics displayed on-screen
- **TensorBoard Integration**: Comprehensive logging and monitoring
- **Hyperparameter Tuning**: Automated optimization with Optuna
- **Phase Progression System**: Increasing difficulty as agent improves

## Installation

```bash
# Using uv (recommended)
uv run python -m arena.train --help

# Traditional installation
pip install -e .
```

## Quick Start

### Training

Train an agent using the modular trainer:
```bash
# DQN with rotation controls (Style 1)
uv run python -m arena.train --algo dqn --style 1 --steps 100000

# PPO with directional controls (Style 2)
uv run python -m arena.train --algo ppo --style 2 --steps 100000

# Recurrent PPO (LSTM)
uv run python -m arena.train --algo ppo_lstm --style 1
```

Monitor training with TensorBoard:
```bash
tensorboard --logdir logs
```

### Evaluation

Launch the interactive evaluation menu:
```bash
uv run python -m arena.evaluate
```

## Architecture

The project follows a modular, industry-standard architecture:

### Directory Structure

```
arena/
├── core/             # Base classes, hardware management, environment
│   ├── config.py     # Dataclass-based configurations
│   ├── device.py     # Hardware auto-detection (CUDA/MPS)
│   └── environment.py # Gymnasium environment implementation
├── game/             # Pure game logic
│   ├── entities.py   # Refactored Player, Enemy, Spawner, Projectile
│   └── utils.py      # Math & physics utilities
├── training/         # Training orchestration
│   ├── algorithms/   # Registered RL trainers (DQN, PPO, etc.)
│   ├── base.py       # Abstract BaseTrainer class
│   ├── callbacks.py  # Custom TensorBoard callbacks
│   └── runner.py     # Global training execution logic
├── evaluation/       # Inference components
│   └── evaluator.py  # Interactive evaluation manager
├── ui/               # Visualization
│   ├── menu.py       # Pygame model selection menu
│   └── renderer.py   # High-performance Pygame renderer
├── train.py          # Unified training CLI
└── evaluate.py       # Unified evaluation CLI
```

### Core Features

- **Algorithm Registry**: Effortlessly plug in new SB3 algorithms.
- **Hardware Acceleration**: Automatic detection of CUDA (NVIDIA) or MPS (Apple Silicon).
- **Type Safety**: Full dataclass support for hyperparameters and trainer settings.
- **Interactive Evaluation**: Sleek menu for selecting and playing against trained models.

## Environment Details

### Observation Space (14 dimensions)
(Same as before: Normalized positions, velocities, rotation, health, phase, and nearest entity sensors)

### Action Spaces
- **Style 1 (Rotation + Thrust)**: 5 actions
- **Style 2 (Directional)**: 6 actions

## Tips for Better Training

1. **Start with PPO**: Generally more stable than DQN for this environment.
2. **Try PPO_LSTM**: Excellent for overcoming POMDP issues if the sensors are limited.
3. **Monitor TensorBoard**: Watch the `arena/` metrics for phase progression and win rates.
4. **Hardware Selection**: The trainer defaults to the best available device but can be overridden with `--device cpu`.

## Assignment Requirements Met

✅ Controllable player with movement and shooting  
✅ Enemy spawners that periodically create enemies  
✅ Enemies that navigate toward player  
✅ Player and enemy health systems  
✅ Projectile collision detection  
✅ Phase progression system  
✅ Continuous/semi-continuous movement  
✅ Visual rendering with training metrics  
✅ Gym API (reset, step, render)  
✅ 14-dimensional observation vector  
✅ Two distinct control schemes  
✅ DQN and PPO support  
✅ Reward function with shaping  
✅ TensorBoard logging  
✅ Neural network architecture  
✅ Model saving and evaluation  

## Advanced Features

- **Hyperparameter Optimization**: Automated tuning with Optuna
- **Custom Callbacks**: Track arena-specific metrics
- **Phase System**: Curriculum-like difficulty progression
- **Beautiful Rendering**: Real-time training visualization
- **Comprehensive Logging**: TensorBoard integration with custom metrics

## License

Educational project for Game AI Design coursework.
