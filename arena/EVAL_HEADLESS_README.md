# Headless Evaluation System

A high-performance, headless evaluation system for Deep RL Arena models that can efficiently run thousands of episodes and generate comprehensive statistics for model comparison.

## Features

- **Fast Headless Execution**: No rendering overhead, runs 10-100x faster than GUI evaluation
- **Comprehensive Statistics**: Win rate, rewards, episode length, spawner kills, phase progression, and more
- **Model Comparison**: Evaluate and compare multiple models side-by-side
- **Flexible Output**: Human-readable summaries and machine-readable JSON
- **Algorithm Support**: Automatic detection and support for PPO, PPO-LSTM, PPO-DICT, A2C, DQN
- **VecNormalize Compatible**: Automatically loads normalization stats when available
- **Deterministic & Stochastic**: Support for both evaluation modes

## Quick Start

### Evaluate a Single Model

```bash
python -m arena.eval_headless --model path/to/model.zip --episodes 1000
```

### Compare Multiple Models

```bash
python -m arena.eval_headless \
    --models model1.zip model2.zip model3.zip \
    --episodes 500 \
    --compare
```

### Evaluate All Models in a Directory

```bash
python -m arena.eval_headless \
    --directory runs/ppo/style1/ \
    --episodes 100 \
    --compare
```

## Usage

```bash
python -m arena.eval_headless [options]
```

### Required Arguments (choose one)

- `--model PATH`: Evaluate a single model file
- `--models PATH [PATH ...]`: Evaluate multiple model files
- `--directory PATH`: Evaluate all models in a directory

### Optional Arguments

**Evaluation Parameters:**
- `--episodes N`: Number of episodes per model (default: 100)
- `--style {1,2}`: Control style (default: 1)
- `--stochastic`: Use stochastic policy instead of deterministic
- `--algo ALGO`: Force algorithm type (auto-detected by default)

**Output Options:**
- `--output PATH`: Save results to JSON file
- `--save-episodes`: Include raw episode data in output
- `--quiet`: Suppress progress messages
- `--compare`: Print comparison table for multiple models

**Performance:**
- `--device {auto,cpu,cuda}`: Device to use (default: auto)
- `--workers N`: Parallel workers (default: 1, not yet fully implemented)

## Output Statistics

### Win Statistics
- Win rate (percentage)
- Number of wins
- Average step at which wins occur

### Reward Statistics
- Mean, standard deviation, median
- Min and max rewards

### Episode Length
- Mean, standard deviation, median
- Min and max length

### Performance Metrics
- Average spawners destroyed per episode
- Average enemies destroyed
- Average phase reached
- Average final health
- Average step of first spawner kill

### Distributions
- Phase distribution (how many episodes reached each phase)
- Spawner kill distribution

## Examples

### 1. Quick Check (100 episodes)

Good for quick model sanity checks during training:

```bash
python -m arena.eval_headless \
    --model runs/ppo/style1/ppo_style1_latest.zip \
    --episodes 100
```

**Speed**: ~10-30 seconds  
**Use case**: Quick validation during training

### 2. Reliable Evaluation (1000 episodes)

For more reliable statistics:

```bash
python -m arena.eval_headless \
    --model runs/ppo/style1/ppo_style1_final.zip \
    --episodes 1000 \
    --output results.json
```

**Speed**: ~2-5 minutes  
**Use case**: Model comparison and selection

### 3. Publication Quality (10000 episodes)

For research papers or final benchmarks:

```bash
python -m arena.eval_headless \
    --model runs/ppo/style1/ppo_style1_final.zip \
    --episodes 10000 \
    --output publication_results.json
```

**Speed**: ~15-45 minutes  
**Use case**: Final benchmarks, research papers

### 4. Compare Training Checkpoints

Track improvement across training:

```bash
python -m arena.eval_headless \
    --models \
        runs/ppo/style1/run/checkpoints/model_600000_steps.zip \
        runs/ppo/style1/run/checkpoints/model_1200000_steps.zip \
        runs/ppo/style1/run/final/model_final.zip \
    --episodes 500 \
    --compare \
    --output checkpoint_comparison.json
```

### 5. Batch Evaluate All Models

Find your best model across all runs:

```bash
python -m arena.eval_headless \
    --directory runs/ppo/style1/ \
    --episodes 200 \
    --compare \
    --output all_models_evaluation.json
```

### 6. Stochastic vs Deterministic

Compare policy behaviors:

```bash
# Deterministic
python -m arena.eval_headless \
    --model model.zip \
    --episodes 1000 \
    --output det_results.json

# Stochastic
python -m arena.eval_headless \
    --model model.zip \
    --episodes 1000 \
    --stochastic \
    --output stoch_results.json
```

## Output Format

### Console Output

The script provides a detailed summary including:

```
================================================================================
EVALUATION SUMMARY: ppo_style1_20251225_175203_final.zip
================================================================================
Algorithm: ppo | Style: 1 | Deterministic: True
Episodes: 1000 | Eval Time: 145.32s
Speed: 6.9 episodes/sec

--------------------------------------------------------------------------------
WIN STATISTICS
--------------------------------------------------------------------------------
Win Rate:         45.20% (452/1000)
Avg Win Step:       1234 steps

--------------------------------------------------------------------------------
REWARD STATISTICS
--------------------------------------------------------------------------------
Mean:             1250.45 ± 345.67
Median:           1280.32
Min/Max:           -50.00 / 2100.50

[... more statistics ...]
```

### JSON Output

The JSON file contains all statistics in a structured format:

```json
{
  "timestamp": "2025-12-26 10:30:45",
  "num_models": 1,
  "evaluations": [
    {
      "model_path": "path/to/model.zip",
      "model_name": "model.zip",
      "algorithm": "ppo",
      "style": 1,
      "deterministic": true,
      "total_episodes": 1000,
      "eval_time_seconds": 145.32,
      "win_rate": 0.452,
      "wins": 452,
      "mean_reward": 1250.45,
      "std_reward": 345.67,
      ...
    }
  ]
}
```

## Performance Tips

### Speed Benchmarks

On a typical machine (CPU-based evaluation):
- **Single episode**: ~0.1-0.3 seconds
- **100 episodes**: ~10-30 seconds
- **1000 episodes**: ~2-5 minutes
- **10000 episodes**: ~15-45 minutes

GPU acceleration provides minimal benefit for inference at these scales.

### Optimization Tips

1. **Use deterministic evaluation** (default): Faster and more reproducible
2. **Run without GUI**: This script is already headless, much faster than GUI eval
3. **Batch multiple models**: More efficient than running separately
4. **Save to JSON**: Analyze results later without re-running

## Integration with Training

You can easily integrate this into your training workflow:

```bash
# After training completes
python -m arena.train --algo ppo --style 1 --steps 1000000

# Evaluate the final model
python -m arena.eval_headless \
    --model runs/ppo/style1/latest_run/final/*.zip \
    --episodes 1000 \
    --output final_evaluation.json
```

## Common Use Cases

### 1. Model Selection

Evaluate all checkpoints to find the best one:

```bash
python -m arena.eval_headless \
    --directory runs/ppo/style1/run_name/checkpoints/ \
    --episodes 500 \
    --compare
```

### 2. Hyperparameter Comparison

Compare models trained with different hyperparameters:

```bash
python -m arena.eval_headless \
    --models \
        runs/ppo/style1/lr_0.0001/final/*.zip \
        runs/ppo/style1/lr_0.0003/final/*.zip \
        runs/ppo/style1/lr_0.001/final/*.zip \
    --episodes 1000 \
    --compare \
    --output hp_comparison.json
```

### 3. Algorithm Comparison

Compare different algorithms on the same task:

```bash
python -m arena.eval_headless \
    --models \
        runs/ppo/style1/final/*.zip \
        runs/a2c/style1/final/*.zip \
        runs/ppo_lstm/style1/final/*.zip \
    --episodes 1000 \
    --compare
```

### 4. Curriculum Learning Validation

Evaluate model at different training stages:

```bash
# Early training (should be weak)
python -m arena.eval_headless \
    --model runs/ppo/style1/checkpoints/model_200000_steps.zip \
    --episodes 500

# Mid training
python -m arena.eval_headless \
    --model runs/ppo/style1/checkpoints/model_600000_steps.zip \
    --episodes 500

# Final model
python -m arena.eval_headless \
    --model runs/ppo/style1/final/*.zip \
    --episodes 500
```

## Troubleshooting

### Model Load Errors

**Problem**: "Failed to load model"
- Ensure the model path is correct
- Verify the model file exists and is not corrupted
- Try specifying the algorithm explicitly with `--algo`

### VecNormalize Warnings

**Problem**: "No VecNormalize stats found"
- This is expected for older models
- Model will still run but may have degraded performance
- Newer models automatically save VecNormalize stats

### Slow Evaluation

**Problem**: Evaluation is slower than expected
- Ensure you're not rendering (use this script, not GUI eval)
- Check if other processes are using resources
- Consider using fewer episodes for quick checks

### Memory Issues

**Problem**: Out of memory errors
- Reduce the number of episodes
- Don't use `--save-episodes` flag
- Evaluate models one at a time instead of batch

## Advanced Usage

### Custom Analysis

Load the JSON output for custom analysis:

```python
import json
import numpy as np

with open('results.json', 'r') as f:
    data = json.load(f)

for eval in data['evaluations']:
    print(f"Model: {eval['model_name']}")
    print(f"Win Rate: {eval['win_rate']:.2%}")
    print(f"Mean Reward: {eval['mean_reward']:.2f}")
```

### Confidence Intervals

For statistical significance, use 1000+ episodes and calculate confidence intervals:

```python
import scipy.stats as stats

# 95% confidence interval for win rate
n = eval['total_episodes']
win_rate = eval['win_rate']
ci = stats.binom.interval(0.95, n, win_rate)
print(f"Win rate: {win_rate:.2%} (95% CI: {ci[0]/n:.2%} - {ci[1]/n:.2%})")
```

## Future Enhancements

Planned features:
- [ ] Multi-process parallel evaluation
- [ ] Real-time progress visualization
- [ ] Statistical significance testing
- [ ] Automated report generation (HTML/PDF)
- [ ] Integration with TensorBoard
- [ ] Curriculum-aware evaluation

## FAQ

**Q: How many episodes should I run?**
A: Depends on your goal:
- Quick check: 100 episodes
- Model comparison: 500-1000 episodes
- Final benchmarks: 1000-10000 episodes

**Q: Should I use deterministic or stochastic evaluation?**
A: Deterministic (default) is standard for RL evaluation as it measures learned policy directly. Use stochastic to measure exploration behavior.

**Q: Can I evaluate models trained on different control styles?**
A: Yes, but make sure to use the correct `--style` flag matching the training style.

**Q: How do I know if my model is good?**
A: Key metrics:
- Win rate > 40% is decent
- Win rate > 60% is good
- Win rate > 80% is excellent
- Mean spawners destroyed > 2.0 is decent
- Mean phase reached > 2.0 is good

**Q: Why is my evaluation different from training metrics?**
A: Training metrics are averaged over many environments and may include curriculum effects. This evaluation uses clean, unmodified environments.

## Citation

If you use this evaluation system in your research, please cite:

```bibtex
@software{arena_headless_eval,
  title = {Deep RL Arena Headless Evaluation System},
  year = {2025},
  url = {https://github.com/your-repo/arena}
}
```


