# Quick Start - PufferLib Training

## Installation

```bash
# Install dependencies
uv pip install -r requirements.txt

# Or manually:
uv pip install pufferlib wandb torch gymnasium pygame numpy
```

## Basic Training

### Train with default settings (MLP policy, Style 1)
```bash
python -m arena.train --style 1 --steps 1000000
```

### Train with LSTM (recurrent policy)
```bash
python -m arena.train --style 1 --policy lstm --steps 1000000
```

### Train with CNN (for heatmap observations)
```bash
python -m arena.train --style 1 --policy cnn --env-type cnn --steps 1000000
```

### Train with custom hyperparameters
```bash
python -m arena.train \
    --style 1 \
    --policy mlp \
    --lr 3e-4 \
    --num-envs 16 \
    --batch-size 512 \
    --gamma 0.99 \
    --device cuda
```

## Evaluation

### Evaluate latest model
```bash
python -m arena.eval_headless --model latest --style 1 --episodes 100
```

### Evaluate specific checkpoint
```bash
python -m arena.eval_headless \
    --model runs/ppo/style1/experiment/checkpoints/checkpoint_500000.pt \
    --episodes 100
```

## Key Options

### Training
- `--style 1|2`: Control scheme (1=rotation+thrust, 2=directional)
- `--policy mlp|lstm|cnn`: Policy architecture
- `--env-type standard|dict|cnn`: Observation space type
- `--steps N`: Total training timesteps
- `--num-envs N`: Parallel environments (more = faster but more memory)
- `--device cuda|cpu|mps`: Compute device
- `--lr FLOAT`: Learning rate
- `--batch-size N`: Steps per update
- `--no-curriculum`: Disable curriculum learning
- `--wandb`: Enable Weights & Biases logging

### Evaluation
- `--model PATH`: Path to `.pt` checkpoint (or "latest")
- `--episodes N`: Number of test episodes
- `--deterministic`: Use deterministic policy (default)
- `--stochastic`: Use stochastic policy (for exploration analysis)

## Quick Examples

**Fast training for testing (CPU, few envs):**
```bash
python -m arena.train --style 1 --steps 100000 --num-envs 4 --device cpu
```

**High-performance training (GPU, many envs):**
```bash
python -m arena.train --style 1 --steps 10000000 --num-envs 32 --device cuda
```

**Resume training from checkpoint:**
```bash
python -m arena.train \
    --load-checkpoint runs/ppo/style1/exp_name/checkpoints/checkpoint_500000.pt \
    --steps 2000000
```

**Train both styles:**
```bash
python -m arena.train --style 1 --steps 1000000 &
python -m arena.train --style 2 --steps 1000000 &
```

## Output Structure

```
runs/ppo/styleN/experiment_name/
├── checkpoints/
│   ├── checkpoint_100000.pt
│   ├── checkpoint_200000.pt
│   └── ...
├── final/
│   └── model.pt              # Final trained model
├── logs/                     # TensorBoard logs
└── config.json              # Training configuration
```

## Troubleshooting

**CUDA out of memory:**
```bash
python -m arena.train --num-envs 8 --batch-size 256
```

**Slow on CPU:**
```bash
python -m arena.train --device cpu --num-envs 4 --vec-backend Serial
```

**Module not found:**
```bash
uv pip install pufferlib
```

## Performance Tips

1. **Use CUDA**: 10-100x faster than CPU
   ```bash
   python -m arena.train --device cuda
   ```

2. **Increase parallel envs**: More envs = better GPU utilization
   ```bash
   python -m arena.train --num-envs 32
   ```

3. **Larger batches**: Better sample efficiency
   ```bash
   python -m arena.train --batch-size 1024
   ```

4. **Multiprocessing backend**: Faster for most cases
   ```bash
   python -m arena.train --vec-backend Multiprocessing
   ```

## Next Steps

- See `PUFFERLIB_MIGRATION.md` for complete migration details
- Check `README.md` for game mechanics and environment details
- Read PufferLib docs: https://puffer.ai/docs.html
