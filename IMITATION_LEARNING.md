# Imitation Learning for Deep RL Arena

This document describes how to use Imitation Learning (Behavioral Cloning) to bootstrap your RL agents with human demonstrations.

## Overview

Imitation Learning allows you to:
1. **Record human demonstrations** - Play the game yourself and record your gameplay
2. **Pretrain agents** - Use these demonstrations to give your agent a head start before RL training
3. **Accelerate learning** - Agents start with good behaviors instead of random exploration

## Quick Start

### 1. Record Human Demonstrations

Launch the evaluation interface and select "Record Demos":

```bash
python arena/evaluate.py
```

In the menu:
1. Select **"Record Demos"** mode
2. Choose your **control style** (1 or 2)
3. Click **"Start"**

**Controls:**
- **Control Style 1** (Rotation + Thrust):
  - W/↑: Thrust forward
  - A/←: Rotate left
  - D/→: Rotate right
  - Space: Shoot
  - ESC: Stop recording and save

- **Control Style 2** (Directional):
  - W/↑: Move up
  - S/↓: Move down
  - A/←: Move left
  - D/→: Move right
  - Space: Shoot
  - ESC: Stop recording and save

Play as many episodes as you like. The more demonstrations, the better!

**Tips for good demonstrations:**
- Try to win episodes (destroy all spawners)
- Avoid taking damage when possible
- Show diverse strategies (different approaches to combat)
- Record at least 5-10 successful episodes for best results

Demonstrations are automatically saved to `./demonstrations/demo_style{N}_TIMESTAMP.pkl`

### 2. Train with Behavioral Cloning

Now use your demonstrations to pretrain an agent:

```bash
python arena/train.py \
  --algo ppo \
  --style 1 \
  --demo-path ./demonstrations/demo_style1_20260106_143022.pkl \
  --bc-pretrain \
  --bc-epochs 20 \
  --steps 1000000
```

**Arguments:**
- `--demo-path`: Path to your recorded demonstrations
- `--bc-pretrain`: Enable behavioral cloning pretraining
- `--bc-epochs`: Number of BC training epochs (default: 10)
- `--bc-lr`: BC learning rate (default: 1e-3)
- `--bc-batch-size`: BC batch size (default: 64)

The agent will first learn from your demonstrations via behavioral cloning, then continue with standard RL training.

## How It Works

### 1. Recording Phase

When you play in "Record Demos" mode, the system captures:
- **Observations**: The state of the game at each timestep (44-dim vector)
- **Actions**: Your control inputs (discrete actions)
- **Rewards**: Feedback from the environment
- **Episode info**: Win/loss, total reward, episode length

All data is saved to a `.pkl` file along with metadata (JSON).

### 2. Behavioral Cloning Phase

During BC pretraining:
1. Your demonstrations are loaded and converted to a training dataset
2. A supervised learning problem is created: predict actions from observations
3. The agent's policy network is trained to mimic your behavior
4. Training metrics (loss, accuracy) are displayed

**Expected Results:**
- Good accuracy (>80%) indicates the agent learned your patterns
- Lower accuracy might mean demonstrations are inconsistent or the task is hard

### 3. Reinforcement Learning Phase

After BC pretraining:
1. The agent starts RL with a warm-started policy (not random)
2. It explores and improves beyond the demonstrations
3. Typically learns faster and achieves better performance

## Advanced Usage

### Standalone BC Training Script

You can also use BC without the full training pipeline:

```python
from arena.training.imitation_learning import (
    DemonstrationBuffer,
    pretrain_from_demonstrations
)
from stable_baselines3 import PPO
from arena.core.environment import ArenaEnv

# Load demonstrations
buffer = DemonstrationBuffer.load("./demonstrations/demo_style1_20260106_143022.pkl")

# Print statistics
stats = buffer.get_statistics()
print(f"Demonstrations: {stats['num_demonstrations']}")
print(f"Total transitions: {stats['total_transitions']}")
print(f"Average return: {stats['avg_return']:.2f}")
print(f"Win rate: {stats['win_rate']*100:.1f}%")

# Create a model
env = ArenaEnv(control_style=1)
model = PPO("MlpPolicy", env)

# Pretrain with BC
history = pretrain_from_demonstrations(
    model=model,
    demo_path="./demonstrations/demo_style1_20260106_143022.pkl",
    num_epochs=20,
    learning_rate=1e-3,
    batch_size=64,
    device="auto",
    verbose=True
)

# Now train with RL
model.learn(total_timesteps=1000000)
```

### Recording Programmatically

You can also record demonstrations programmatically:

```python
from arena.training.imitation_learning import DemonstrationRecorder
from arena.core.environment import ArenaEnv
from arena.game.human_controller import HumanController
import pygame

env = ArenaEnv(control_style=1, render_mode="human")
controller = HumanController(style=1)
recorder = DemonstrationRecorder(control_style=1)

# Record multiple episodes
for episode in range(10):
    obs, _ = env.reset()
    recorder.start_episode()
    done = False
    
    while not done:
        action = controller.get_action(pygame.event.get())
        obs, reward, terminated, truncated, info = env.step(action)
        recorder.record_step(obs, action, reward, terminated or truncated)
        done = terminated or truncated
        env.render()
    
    recorder.end_episode(info)

# Save all demonstrations
filepath = recorder.save_demonstrations()
print(f"Saved to: {filepath}")
```

## File Format

Demonstration files (`.pkl`) contain:

```python
{
    'demonstrations': [
        {
            'observations': [[...], [...], ...],  # List of observation arrays
            'actions': [2, 4, 1, ...],            # List of action indices
            'rewards': [0.1, -0.02, 5.0, ...],    # List of rewards
            'dones': [False, False, True],        # Episode termination flags
            'info': {
                'control_style': 1,
                'win': True,
                'win_step': 450,
                ...
            }
        },
        # ... more demonstrations
    ],
    'metadata': {
        'control_style': 1,
        'timestamp': '2026-01-06T14:30:22',
        'num_demonstrations': 10,
        'total_transitions': 4523,
        'avg_return': 245.3,
        'win_rate': 0.8
    }
}
```

Metadata is also saved as JSON (`.json`) for easy inspection.

## Troubleshooting

### "No demonstrations found in buffer"
- Make sure you played at least one complete episode before pressing ESC
- Check that the demonstration file exists and is not corrupted

### Low BC accuracy (<50%)
- Your demonstrations might be inconsistent (different strategies)
- Try recording more episodes with a consistent strategy
- Increase BC epochs or learning rate

### Agent doesn't improve after BC
- BC gives a head start but might not find optimal policy
- Make sure RL training continues for enough steps
- Try adjusting RL hyperparameters (learning rate, entropy coefficient)

### Control style mismatch
- Always use the same control style for recording and training
- Check that `--style` matches the style used during recording

## Best Practices

1. **Quality over quantity**: 5-10 good demonstrations beat 50 mediocre ones
2. **Consistent strategy**: Try to play similarly across episodes
3. **Show success**: Include winning episodes in your demonstrations
4. **Diverse situations**: Show how to handle different enemy configurations
5. **BC as initialization**: Use BC epochs to get ~70-80% accuracy, then let RL improve
6. **Monitor BC metrics**: High accuracy doesn't guarantee good RL performance

## Implementation Details

### Behavioral Cloning

The BC implementation uses:
- **Algorithm**: Supervised learning with cross-entropy loss
- **Network**: Reuses the policy network from the RL algorithm
- **Optimizer**: Adam with configurable learning rate
- **Dataset**: State-action pairs flattened from all demonstrations
- **Validation**: Tracks loss and accuracy during training

### Integration with SB3

The imitation learning module integrates with Stable-Baselines3:
- Works with PPO, A2C, DQN, and variants
- Extracts and trains the policy network directly
- Preserves SB3 model structure for seamless RL training
- Compatible with all SB3 features (callbacks, tensorboard, etc.)

## Citation

If you use this imitation learning implementation in your research, please cite:

```
@software{arena_imitation_learning,
  title={Imitation Learning for Deep RL Arena},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/arena}
}
```

## References

- Behavioral Cloning: Pomerleau (1991) - "Efficient Training of Artificial Neural Networks for Autonomous Navigation"
- Imitation Learning Survey: Hussein et al. (2017) - "Imitation Learning: A Survey of Learning Methods"
- Stable-Baselines3: Raffin et al. (2021) - https://github.com/DLR-RM/stable-baselines3
