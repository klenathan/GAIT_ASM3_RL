# Deep RL Arena - Space Combat Environment

A challenging reinforcement learning environment where agents pilot spaceships to destroy enemy spawners across multiple phases.

## Features

- **Two Control Styles**: Rotation+Thrust (physics-based) or Directional movement (arcade-style)
- **Multiple RL Algorithms**: PPO, A2C, DQN, RecurrentPPO, and variants
- **Curriculum Learning**: Progressive difficulty scaling
- **Imitation Learning**: Bootstrap agents from human demonstrations
- **Rich Observation Space**: 44-dimensional state including enemies, spawners, projectiles, and spatial info
- **Phase-Based Gameplay**: 5 phases with increasing difficulty
- **Real-time Visualization**: Watch your agents learn with detailed metrics

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd GAIT_ASM3_RL

# Install dependencies
pip install -r requirements.txt
```

### Training an Agent

```bash
# Train with PPO (recommended for beginners)
python arena/train.py --algo ppo --style 1 --steps 1000000

# Train with A2C
python arena/train.py --algo a2c --style 2 --steps 500000

# Train with DQN
python arena/train.py --algo dqn --style 1 --steps 500000
```

### Evaluating Trained Models

```bash
# Launch interactive evaluation interface
python arena/evaluate.py
```

Use the GUI to:
- Select trained models
- Watch agents play
- Play manually
- Record demonstrations for imitation learning

### Playing Manually

```bash
python arena/evaluate.py
```

Select "Human Player" mode and choose your control style.

**Controls:**
- **Style 1** (Rotation + Thrust): W=thrust, A/D=rotate, Space=shoot
- **Style 2** (Directional): WASD=move, Space=shoot
- **ESC**: Return to menu

## Imitation Learning (NEW!)

Bootstrap your RL agents with human demonstrations:

### 1. Record Demonstrations

```bash
python record_demos.py --style 1
```

Play the game and your actions will be recorded. Press ESC to save.

### 2. Train with Behavioral Cloning

```bash
python arena/train.py \
  --algo ppo \
  --style 1 \
  --demo-path ./demonstrations/demo_style1_TIMESTAMP.pkl \
  --bc-pretrain \
  --bc-epochs 20 \
  --steps 1000000
```

The agent will first learn from your demonstrations, then improve with RL.

**See [IL_QUICKSTART.md](IL_QUICKSTART.md) for quick commands and [IMITATION_LEARNING.md](IMITATION_LEARNING.md) for full documentation.**

## Project Structure

```
GAIT_ASM3_RL/
├── arena/
│   ├── core/              # Environment, configuration, curriculum
│   ├── training/          # RL algorithms and trainers
│   │   ├── algorithms/    # PPO, A2C, DQN implementations
│   │   └── imitation_learning.py  # Behavioral cloning module
│   ├── evaluation/        # Model evaluation and demo recording
│   ├── game/              # Game entities and human controller
│   ├── ui/                # Rendering and menu systems
│   ├── train.py           # Training entry point
│   └── evaluate.py        # Evaluation entry point
├── record_demos.py        # Standalone demo recording script
└── demonstrations/        # Recorded human demonstrations
```

## Environment Details

### Control Styles

**Style 1: Rotation + Thrust (Physics-based)**
- Action space: 5 discrete actions
  - 0: Idle
  - 1: Thrust forward
  - 2: Rotate left
  - 3: Rotate right
  - 4: Shoot
- Realistic momentum and inertia
- Requires planning and precise control

**Style 2: Directional Movement (Arcade-style)**
- Action space: 6 discrete actions
  - 0: Idle
  - 1: Move up
  - 2: Move down
  - 3: Move left
  - 4: Move right
  - 5: Shoot
- Fixed shooting direction (randomized at episode start)
- Easier to learn initially

### Observation Space

44-dimensional continuous observation:
- Player state (position, velocity, rotation, health, cooldown)
- Phase progress and time remaining
- Nearest enemies (2) with distance, angle, existence
- Nearest spawners (2) with distance, angle, existence, health
- Nearest projectiles (5) with distance, angle, existence
- Wall distances (4 directions)
- Enemy count

### Reward Structure

- **Win bonus**: +500 (complete all phases)
- **Phase completion**: +50 per phase
- **Spawner destroyed**: +400
- **Enemy destroyed**: +5
- **Damage dealt**: +2-6 per hit
- **Health bonuses**: +2-5 for completing phases with high HP
- **Penalties**: -200 death, -5 damage taken, -0.02 per step

## Advanced Features

### Curriculum Learning

Agents start with easier scenarios and progressively face harder challenges:
- Fewer enemies initially
- Reduced spawner health
- Longer episode timeouts
- Gradual scaling of difficulty

Automatically enabled by default. Disable with curriculum configuration.

### Algorithms

| Algorithm | Best For | Speed | Sample Efficiency |
|-----------|----------|-------|-------------------|
| PPO | Stable training, good all-around | Medium | Medium |
| A2C | Fast experimentation | Fast | Low |
| DQN | Sample efficiency | Slow | High |
| RecurrentPPO | Partial observability | Medium | Medium |

### Hyperparameter Tuning

```bash
# Custom learning rate
python arena/train.py --algo ppo --lr 0.0001

# More parallel environments
python arena/train.py --algo ppo --num-envs 32

# Longer training
python arena/train.py --algo ppo --steps 5000000
```

### Resume Training

```bash
python arena/train.py \
  --algo ppo \
  --style 1 \
  --load-model ./runs/ppo/style1/RUN_NAME/final/MODEL.zip \
  --steps 1000000
```

## Documentation

- **[IL_QUICKSTART.md](IL_QUICKSTART.md)** - Quick reference for imitation learning
- **[IMITATION_LEARNING.md](IMITATION_LEARNING.md)** - Complete IL documentation
- **[IMITATION_LEARNING_IMPLEMENTATION.md](IMITATION_LEARNING_IMPLEMENTATION.md)** - Technical implementation details

## Tips for Training

1. **Start with PPO and Style 2** - Easier to learn
2. **Use imitation learning** - Record 10-20 good episodes to bootstrap
3. **Monitor TensorBoard** - Watch learning curves in `./runs/`
4. **Increase training steps** - Complex task needs 1M+ steps
5. **Experiment with hyperparameters** - Learning rate, entropy coefficient
6. **Use curriculum learning** - Enabled by default, helps initial learning
7. **Try different algorithms** - Each has strengths

## Performance Benchmarks

Typical training times (on M1 Mac with 20 envs):
- 1M steps: ~20-30 minutes
- 5M steps: ~2-3 hours

Expected win rates after training:
- PPO (1M steps): 10-30%
- PPO (5M steps): 40-60%
- PPO with IL (1M steps): 30-50%
- PPO with IL (5M steps): 60-80%

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Agent doesn't improve | Increase training steps, check reward scale |
| Training too slow | Reduce num_envs or disable rendering |
| Agent camps/hides | Adjust reward penalties for inactivity |
| Out of memory | Reduce num_envs or batch_size |

## Requirements

- Python 3.8+
- PyTorch 1.12+
- Stable-Baselines3 2.0+
- Gymnasium 0.28+
- Pygame 2.0+
- NumPy, Matplotlib

See `requirements.txt` for complete list.

## Citation

If you use this environment in your research, please cite:

```
@software{deep_rl_arena,
  title={Deep RL Arena: Space Combat Environment},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/GAIT_ASM3_RL}
}
```

## License

[Your chosen license]

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Acknowledgments

- Built with [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
- Inspired by classic space combat games
- Developed for reinforcement learning education and research
