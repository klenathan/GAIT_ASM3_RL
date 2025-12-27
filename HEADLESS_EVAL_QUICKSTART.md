# Headless Evaluation - Quick Start Guide

## What is This?

A fast, headless evaluation system that can run **thousands of games** without rendering to give you reliable statistics for comparing models. This is **10-100x faster** than watching games in the GUI.

## Why Use This?

- **Speed**: Evaluate 1000 episodes in 2-5 minutes (vs hours with GUI)
- **Statistics**: Get reliable win rates, rewards, and performance metrics
- **Comparison**: Compare multiple models side-by-side
- **Automation**: Batch evaluate all your models

## Quick Commands

### 1. Evaluate Your Latest Model (100 episodes)

```bash
python -m arena.eval_latest --episodes 100
```

This finds your most recent model and evaluates it. Perfect for quick checks during training.

### 2. Evaluate a Specific Model (1000 episodes)

```bash
python -m arena.eval_headless --model runs/ppo/style1/your_model.zip --episodes 1000
```

For more reliable statistics, use 1000+ episodes.

### 3. Compare Multiple Models

```bash
python -m arena.eval_headless --models model1.zip model2.zip model3.zip --episodes 500 --compare
```

See which model performs best.

### 4. Batch Evaluate All Models

```bash
python -m arena.eval_compare_all --algo ppo --style 1 --episodes 200
```

Evaluate all PPO style1 models and get a comparison table.

## Real Example

Let's say you just trained a PPO model and want to know how good it is:

```bash
# Quick check (30 seconds)
python -m arena.eval_latest --episodes 100

# Detailed evaluation (3 minutes)
python -m arena.eval_latest --episodes 1000 --output my_model_results.json
```

Output will show:
```
================================================================================
EVALUATION SUMMARY
================================================================================
Win Rate:         45.20% (452/1000)
Mean Reward:      1250.45 ± 345.67
Spawners/Episode: 2.34 ± 1.12
Avg Phase:        2.15
```

## Understanding the Results

### Key Metrics

- **Win Rate**: Percentage of games won (completed all phases)
  - 40%+ = Decent
  - 60%+ = Good
  - 80%+ = Excellent

- **Mean Reward**: Average total reward per episode
  - Higher is better
  - Compare relative performance between models

- **Spawners/Episode**: Average spawners destroyed
  - 2.0+ = Decent progress
  - 3.0+ = Good performance
  - 4.0+ = Excellent

- **Avg Phase**: How far the agent progresses
  - Phase 2+ = Learning basics
  - Phase 3+ = Competent
  - Phase 4+ = Advanced

### What to Look For

**Good Model:**
- Win rate > 40%
- Consistent rewards (low std dev)
- Progresses to later phases
- Destroys multiple spawners

**Needs More Training:**
- Win rate < 10%
- High reward variance
- Stuck in early phases
- Few spawner kills

## Common Workflows

### 1. During Training - Quick Checks

Check progress every few checkpoints:

```bash
python -m arena.eval_latest --checkpoint-only --episodes 100
```

### 2. After Training - Full Evaluation

Get comprehensive statistics:

```bash
python -m arena.eval_latest --final-only --episodes 1000 --output final_eval.json
```

### 3. Compare Checkpoints

See improvement over training:

```bash
python -m arena.eval_compare_all --run-dir runs/ppo/style1/your_run --episodes 500
```

### 4. Find Best Model

Evaluate all models and find the winner:

```bash
python -m arena.eval_compare_all --algo ppo --style 1 --final-only --episodes 500
```

## Tips for Speed

- **100 episodes**: ~10-30 seconds - quick sanity checks
- **500 episodes**: ~1-2 minutes - decent statistics
- **1000 episodes**: ~2-5 minutes - reliable statistics
- **10000 episodes**: ~15-45 minutes - publication quality

Use fewer episodes during development, more for final benchmarks.

## Complete Examples

### Example 1: Check if training is working

After starting training and getting some checkpoints:

```bash
python -m arena.eval_latest --checkpoint-only --episodes 100
```

If win rate is improving, training is working!

### Example 2: Compare hyperparameters

You trained 3 models with different learning rates:

```bash
python -m arena.eval_headless \
    --models runs/ppo/lr_0.0001/final/*.zip \
             runs/ppo/lr_0.0003/final/*.zip \
             runs/ppo/lr_0.001/final/*.zip \
    --episodes 1000 \
    --compare
```

Pick the one with highest win rate.

### Example 3: Final model selection

Training complete, find your best model:

```bash
# Evaluate all final models
python -m arena.eval_compare_all --final-only --episodes 1000 --output best_model.json

# Or evaluate specific run's checkpoints to find best checkpoint
python -m arena.eval_compare_all --run-dir runs/ppo/style1/ppo_style1_20251225_175203 --episodes 1000
```

### Example 4: Publication benchmarks

For a paper or final report:

```bash
python -m arena.eval_headless \
    --model runs/ppo/style1/best_model.zip \
    --episodes 10000 \
    --output publication_results.json
```

Run overnight for 10,000 episodes = rock-solid statistics.

## Troubleshooting

**"No models found"**
- Make sure you're in the project root directory
- Check that you have models in `runs/` directory
- Use `--run-dir` to point to a specific directory

**"Failed to load model"**
- Verify the model path is correct
- Try specifying `--algo ppo` explicitly
- Check the model file isn't corrupted

**Evaluation seems wrong**
- Make sure you're using the right `--style` (1 or 2)
- Check that VecNormalize stats are being loaded (warnings will show)
- Use `--deterministic` (default) for consistent results

**Too slow**
- Reduce `--episodes` for quick checks
- Make sure no GUI is running
- Close other programs

## Next Steps

1. **Read the full documentation**: `arena/EVAL_HEADLESS_README.md`
2. **Check examples**: `arena/eval_examples.bat` (Windows) or `.sh` (Linux)
3. **Customize**: Use the API directly for custom analysis

## Advanced: Using Results Programmatically

The JSON output can be loaded for custom analysis:

```python
import json

with open('results.json') as f:
    data = json.load(f)

for eval_result in data['evaluations']:
    print(f"Model: {eval_result['model_name']}")
    print(f"Win Rate: {eval_result['win_rate']:.1%}")
    print(f"Mean Reward: {eval_result['mean_reward']:.1f}")
    print()
```

## Summary

**For quick checks:**
```bash
python -m arena.eval_latest --episodes 100
```

**For reliable statistics:**
```bash
python -m arena.eval_latest --episodes 1000
```

**For comparisons:**
```bash
python -m arena.eval_compare_all --algo ppo --episodes 500
```

That's it! Start with 100 episodes for speed, use 1000+ for accuracy.

---

**Need help?** Check `arena/EVAL_HEADLESS_README.md` for complete documentation.

