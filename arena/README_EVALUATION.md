# Model Evaluation Guide

Deep RL Arena provides two evaluation systems:

## 1. GUI Evaluation (Interactive)

**Use for:** Watching your agent play, debugging behavior, demonstrations

```bash
python -m arena.evaluate
```

Features:
- Visual rendering
- Real-time model output visualization
- Interactive menu
- Control toggles (health, vision, debug)

## 2. Headless Evaluation (Batch)

**Use for:** Fast statistics, model comparison, benchmarking

```bash
python -m arena.eval_headless --model model.zip --episodes 1000
```

Features:
- 10-100x faster than GUI
- Run thousands of episodes
- Comprehensive statistics
- Model comparison
- JSON output

## Quick Start - Headless Evaluation

### Evaluate Your Latest Model
```bash
python -m arena.eval_latest --episodes 100
```

### Compare Multiple Models
```bash
python -m arena.eval_headless --models model1.zip model2.zip model3.zip --compare
```

### Batch Evaluate All Models
```bash
python -m arena.eval_compare_all --algo ppo --style 1 --episodes 500
```

## Documentation

- **Quick Start:** `../HEADLESS_EVAL_QUICKSTART.md`
- **Full Guide:** `EVAL_HEADLESS_README.md`
- **Implementation:** `../HEADLESS_EVAL_SUMMARY.md`
- **Examples:** `eval_examples.bat` (Windows) or `eval_examples.sh` (Linux)

## When to Use Each

| Task | GUI | Headless |
|------|-----|----------|
| Watch agent play | ✓ | ✗ |
| Debug behavior | ✓ | ✗ |
| Quick visual check | ✓ | ✗ |
| Get win rate | ✗ | ✓ |
| Compare models | ✗ | ✓ |
| Reliable statistics | ✗ | ✓ |
| Batch evaluation | ✗ | ✓ |
| Automated testing | ✗ | ✓ |

## Typical Workflow

1. **During Training:** Use headless eval for quick checks
   ```bash
   python -m arena.eval_latest --checkpoint-only --episodes 100
   ```

2. **After Training:** Use headless eval for comprehensive stats
   ```bash
   python -m arena.eval_latest --final-only --episodes 1000
   ```

3. **For Debugging:** Use GUI eval to watch behavior
   ```bash
   python -m arena.evaluate
   ```

4. **For Comparison:** Use headless eval to compare models
   ```bash
   python -m arena.eval_compare_all --algo ppo --episodes 500
   ```

## Output Example

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
Win Rate:        45.20% (452/1000)
Avg Win Step:      1234 steps

--------------------------------------------------------------------------------
REWARD STATISTICS
--------------------------------------------------------------------------------
Mean:            1250.45 ± 345.67
Median:          1280.32
Min/Max:          -50.00 / 2100.50

--------------------------------------------------------------------------------
PERFORMANCE METRICS
--------------------------------------------------------------------------------
Spawners/Episode:  2.34 ± 1.12
Enemies/Episode:  15.6
Avg Phase:         2.15
Avg Final HP:       45
```

## Testing

Test the headless evaluation system:
```bash
python test_headless_eval.py
```

## Help

```bash
# Headless evaluation help
python -m arena.eval_headless --help

# Latest model evaluation help
python -m arena.eval_latest --help

# Batch comparison help
python -m arena.eval_compare_all --help
```

## More Information

- Main README: `../README.md`
- Training Guide: `../README.md` (Training section)
- Code Documentation: Inline docstrings in all modules


