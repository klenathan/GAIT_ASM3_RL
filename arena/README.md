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
# Install dependencies
pip install -e .

# Or with uv
uv pip install -e .
```

## Quick Start

### Training

Train a DQN agent with rotation controls:
```bash
python -m arena.train --algo dqn --style 1 --steps 100000
```

Train a PPO agent with directional controls:
```bash
python -m arena.train --algo ppo --style 2 --steps 100000
```

Monitor training with TensorBoard:
```bash
tensorboard --logdir logs
```

### Evaluation

Evaluate a trained model:
```bash
python -m arena.evaluate --model models/dqn_style1_final.zip --episodes 5
```

### Hyperparameter Tuning

Optimize hyperparameters with Optuna:
```bash
python -m arena.tune_hyperparameters --algo ppo --style 2 --n_trials 50 --timesteps 50000
```

## Environment Details

### Observation Space (14 dimensions)

1. Player X position (normalized)
2. Player Y position (normalized)
3. Player velocity X
4. Player velocity Y
5. Player rotation angle
6. Player health (0-1)
7. Current phase (0-1)
8. Nearest enemy distance
9. Nearest enemy angle
10. Nearest enemy exists (binary)
11. Nearest spawner distance
12. Nearest spawner angle
13. Nearest spawner exists (binary)
14. Number of active enemies

### Action Spaces

**Style 1 (Rotation + Thrust)**:
- 0: No action
- 1: Thrust forward
- 2: Rotate left
- 3: Rotate right
- 4: Shoot

**Style 2 (Directional)**:
- 0: No action
- 1: Move up
- 2: Move down
- 3: Move left
- 4: Move right
- 5: Shoot

### Reward Structure

- **Enemy destroyed**: +10
- **Spawner destroyed**: +50
- **Phase complete**: +100
- **Damage taken**: -5
- **Death**: -100
- **Survival**: +0.1 per step
- **Approach spawner**: +1 (shaping reward)
- **Quick kill bonus**: +5

## Neural Network Architecture

### DQN
- Input: 14-dimensional observation
- Hidden layers: [256, 128, 64]
- Activation: ReLU
- Output: 5 or 6 actions (depending on control style)

### PPO
- Shared feature extractor: [256, 128]
- Policy head: 64 units
- Value head: 64 units
- Orthogonal initialization
- Activation: ReLU

## Hyperparameters

### DQN (Default)
- Learning rate: 3e-4
- Buffer size: 100,000
- Batch size: 64
- Gamma: 0.99
- Exploration: ε 1.0 → 0.05 over 50% of training

### PPO (Default)
- Learning rate: 3e-4
- N steps: 2048
- Batch size: 64
- N epochs: 10
- Gamma: 0.99
- GAE lambda: 0.95
- Clip range: 0.2
- Entropy coefficient: 0.01

## File Structure

```
arena/
├── __init__.py           # Package initialization
├── config.py             # Configuration and hyperparameters
├── entities.py           # Game entities (Player, Enemy, Spawner, Projectile)
├── environment.py        # Gym environment implementation
├── utils.py              # Utility functions
├── renderer.py           # Pygame visualization
├── callbacks.py          # Training callbacks
├── train.py              # Training script
├── evaluate.py           # Evaluation script
└── tune_hyperparameters.py  # Optuna hyperparameter tuning
```

## Tips for Better Training

1. **Start with PPO**: Generally more stable than DQN for this environment
2. **Use hyperparameter tuning**: Run Optuna to find optimal parameters
3. **Monitor TensorBoard**: Watch for reward improvements and phase progression
4. **Adjust reward shaping**: Modify config.py to emphasize different behaviors
5. **Train longer**: Complex behaviors may require 500k+ timesteps
6. **Try both control styles**: Style 2 (directional) is often easier to learn

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
