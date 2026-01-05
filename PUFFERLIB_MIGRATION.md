# PufferLib Migration Complete

This project has been successfully migrated from Stable-Baselines3 (SB3) to PufferLib for high-performance reinforcement learning training.

## Major Changes

### 🎯 What Changed

1. **Training Framework**: Replaced SB3 with PufferLib's explicit PPO implementation
2. **Model Format**: Changed from `.zip` (SB3) to `.pt` (PyTorch) checkpoints
3. **Algorithms**: Simplified to PPO-only (removed DQN, A2C)
4. **Performance**: Expected 10-100x speedup in training throughput

### 📂 New File Structure

```
arena/
├── core/
│   └── puffer_envs.py          # PufferLib environment wrappers
├── training/
│   ├── policies/               # NEW: PyTorch policy models
│   │   ├── mlp_policy.py      # Feedforward policy
│   │   ├── lstm_policy.py     # Recurrent policy
│   │   └── cnn_policy.py      # CNN policy for image obs
│   └── pufferl_trainer.py     # NEW: Main PufferLib trainer
├── evaluation/
│   └── evaluator.py            # Updated for .pt checkpoints
├── train.py                    # NEW: PufferLib training CLI
└── eval_headless.py            # Updated evaluation script
```

### 🗑️ Removed Files

- `arena/training/base.py` (SB3 base trainer)
- `arena/training/callbacks.py` (SB3 callbacks)
- `arena/training/cnn_extractor.py` (SB3 feature extractor)
- `arena/training/algorithms/*.py` (All SB3 algorithm wrappers)
- `arena/ui/extractors/*` (SB3 model extractors)

### 📦 Dependencies

**Old**: `stable-baselines3[extra]`, `sb3-contrib`
**New**: `pufferlib`, `wandb`

Update with:
```bash
uv pip install -r requirements.txt
```

## Usage Guide

### Training

**Basic training:**
```bash
python -m arena.train --style 1 --steps 1000000
```

**With custom hyperparameters:**
```bash
python -m arena.train \
    --style 1 \
    --policy mlp \
    --env-type standard \
    --lr 3e-4 \
    --num-envs 16 \
    --batch-size 512 \
    --device cuda
```

**LSTM policy:**
```bash
python -m arena.train --style 1 --policy lstm
```

**CNN policy (with heatmap observations):**
```bash
python -m arena.train --style 1 --policy cnn --env-type cnn
```

**Resume from checkpoint:**
```bash
python -m arena.train --load-checkpoint runs/ppo/style1/experiment_name/checkpoints/checkpoint_500000.pt
```

### Evaluation

**Evaluate latest model:**
```bash
python -m arena.eval_headless --model latest --style 1 --episodes 100
```

**Evaluate specific checkpoint:**
```bash
python -m arena.eval_headless \
    --model runs/ppo/style1/experiment_name/checkpoints/checkpoint_500000.pt \
    --episodes 100 \
    --deterministic
```

### Command-Line Options

**Training options:**
- `--style`: Control style (1 or 2)
- `--env-type`: Observation type (`standard`, `dict`, `cnn`)
- `--policy`: Policy architecture (`mlp`, `lstm`, `cnn`)
- `--steps`: Total training timesteps
- `--device`: Device (`cuda`, `cpu`, `mps`)
- `--num-envs`: Number of parallel environments
- `--lr`: Learning rate
- `--batch-size`: Steps per update
- `--no-normalize-obs`: Disable observation normalization
- `--no-curriculum`: Disable curriculum learning
- `--wandb`: Enable Weights & Biases logging

## Migration Notes

### ⚠️ Breaking Changes

1. **Old models incompatible**: SB3 `.zip` checkpoints cannot be loaded with the new system
2. **Different hyperparameters**: PufferLib PPO uses slightly different defaults
3. **No DQN/A2C**: Only PPO is supported (PufferLib's focus)
4. **Manual curriculum**: Curriculum logic is now inline in training loop

### ✅ What Still Works

- All three environment types (standard, dict, CNN)
- Both control styles (rotation+thrust, directional)
- Curriculum learning system
- Observation and reward normalization
- Learning rate schedules
- Checkpointing and resume training

### 🚀 Performance Improvements

PufferLib achieves 10-100x speedup through:
- **Zero-copy batching**: Single shared memory buffer
- **Optimized IPC**: Busy-wait flags instead of pipes
- **Async vectorization**: EnvPool-style simulation
- **Efficient LSTM**: Uses `LSTMCell` during rollouts, `LSTM` for training

### 📊 Logging

PufferLib supports multiple logging backends:
- **TensorBoard**: Default (logs in `runs/*/logs/`)
- **Weights & Biases**: Use `--wandb --wandb-project your-project`
- **Neptune**: Use `--neptune --neptune-project your-project`

### 🔍 Checkpoints

**New format:**
```python
checkpoint = {
    'global_step': int,
    'model_state_dict': OrderedDict,  # PyTorch state_dict
    'optimizer_state_dict': OrderedDict,
    'config': dict,  # Training configuration
    'obs_normalizer': dict,  # Optional normalization stats
    'reward_normalizer': dict,
}
```

**Location:**
- Checkpoints: `runs/ppo/styleN/experiment_name/checkpoints/checkpoint_*.pt`
- Final model: `runs/ppo/styleN/experiment_name/final/model.pt`
- Config: `runs/ppo/styleN/experiment_name/config.json`

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError: No module named 'pufferlib'`:
```bash
uv pip install pufferlib
```

### CUDA Out of Memory

Reduce batch size or number of environments:
```bash
python -m arena.train --num-envs 8 --batch-size 256
```

### Slow Training on CPU

Use fewer environments and serial backend:
```bash
python -m arena.train --device cpu --num-envs 4 --vec-backend Serial
```

### Old Checkpoints

Old SB3 `.zip` models cannot be loaded. You must retrain with PufferLib.
Backup files are preserved as `*_old.py` for reference.

## Development

### Adding Custom Policies

Create a new policy in `arena/training/policies/`:

```python
import torch
import torch.nn as nn

class MyCustomPolicy(nn.Module):
    def __init__(self, observation_space, action_space, **kwargs):
        super().__init__()
        # Your architecture here
    
    def get_action_and_value(self, obs, actions=None, deterministic=False):
        # Return actions, log_probs, entropy, values
        pass
```

Then use with: `--policy custom` (after registering in `policies/__init__.py`)

### Modifying Training Loop

Edit `arena/training/pufferl_trainer.py`:
- `collect_batch()`: Modify experience collection
- `train_batch()`: Modify PPO update logic
- `compute_gae()`: Modify advantage computation

### Custom Environments

Environments remain unchanged! Just add PufferLib wrapper:

```python
from arena.core.puffer_envs import make_env

env = make_env(env_type='standard', control_style=1)
```

## Resources

- **PufferLib Docs**: https://puffer.ai/docs.html
- **PufferLib GitHub**: https://github.com/PufferAI/PufferLib
- **Arena Documentation**: See `README.md` for game mechanics

## Changelog

### 2026-01-05: Complete Migration to PufferLib

**Added:**
- PufferLib trainer infrastructure (`pufferl_trainer.py`)
- PyTorch policy models (MLP, LSTM, CNN)
- Environment wrappers for PufferLib
- New evaluation system for `.pt` checkpoints
- Updated CLI with PufferLib options

**Removed:**
- All Stable-Baselines3 code
- SB3-contrib dependencies
- DQN and A2C algorithms
- Old callback system
- SB3 feature extractors

**Changed:**
- Checkpoint format: `.zip` → `.pt`
- Training framework: SB3 → PufferLib
- Policy implementation: SB3 wrappers → Raw PyTorch
- Normalization: VecNormalize → Custom normalizers

---

For questions or issues, refer to the PufferLib documentation or check the backup files (`*_old.py`).
