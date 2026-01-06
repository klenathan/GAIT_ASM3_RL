# Imitation Learning - Quick Reference

Quick commands for using the Imitation Learning system.

## Recording Demonstrations

### Option 1: Using the GUI
```bash
python arena/evaluate.py
```
Then select: **"Record Demos"** → Choose style → Play → **ESC** to save

### Option 2: Using the standalone script
```bash
# For control style 1 (rotation + thrust)
python record_demos.py --style 1

# For control style 2 (directional movement)
python record_demos.py --style 2

# Save to custom location
python record_demos.py --style 1 --output ./my_demos/expert.pkl
```

**Controls:**
- **Style 1:** W=thrust, A/D=rotate, Space=shoot
- **Style 2:** WASD=move, Space=shoot
- **ESC:** Stop and save

## Training with Demonstrations

### Basic usage
```bash
python arena/train.py \
  --algo ppo \
  --style 1 \
  --demo-path ./demonstrations/demo_style1_TIMESTAMP.pkl \
  --bc-pretrain \
  --steps 1000000
```

### With custom BC parameters
```bash
python arena/train.py \
  --algo ppo \
  --style 1 \
  --demo-path ./demonstrations/demo_style1_TIMESTAMP.pkl \
  --bc-pretrain \
  --bc-epochs 20 \
  --bc-lr 0.001 \
  --bc-batch-size 128 \
  --steps 1000000
```

### Different algorithms
```bash
# A2C with BC
python arena/train.py --algo a2c --style 1 --demo-path DEMO.pkl --bc-pretrain

# DQN with BC
python arena/train.py --algo dqn --style 2 --demo-path DEMO.pkl --bc-pretrain
```

## Checking Demonstration Quality

```bash
python -c "
from arena.training.imitation_learning import DemonstrationBuffer
buffer = DemonstrationBuffer.load('YOUR_DEMO_FILE.pkl')
stats = buffer.get_statistics()
for key, value in stats.items():
    print(f'{key}: {value}')
"
```

## Command-Line Arguments Reference

### Recording (`record_demos.py`)
- `--style {1,2}` - Control style (required)
- `--output PATH` - Custom save location (optional)

### Training (`arena/train.py`)
- `--demo-path PATH` - Path to demonstration file
- `--bc-pretrain` - Enable behavioral cloning (flag)
- `--bc-epochs N` - BC training epochs (default: 10)
- `--bc-lr FLOAT` - BC learning rate (default: 0.001)
- `--bc-batch-size N` - BC batch size (default: 64)

## Tips

1. **Record 10-20 episodes** with consistent strategy
2. **Aim for 70%+ win rate** in demonstrations
3. **BC accuracy >70%** indicates good learning
4. **Compare with/without BC** to measure improvement
5. **Style must match** between recording and training

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "No demonstrations found" | Record at least one complete episode |
| Low BC accuracy (<50%) | Record more consistent demonstrations |
| Control style mismatch | Use same `--style` for recording and training |
| Agent not improving | Increase `--bc-epochs` or check demo quality |

## Full Documentation

See `IMITATION_LEARNING.md` for detailed documentation and examples.
