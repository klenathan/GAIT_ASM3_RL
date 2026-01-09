# Train commands

Artifacts are written under repo-root `runs/gridworld/<run_name>/`.

- TensorBoard events: `runs/gridworld/<run_name>/logs/`
- Best/final model: `runs/gridworld/<run_name>/final/`
- Checkpoints: `runs/gridworld/<run_name>/checkpoints/`

```
python main.py --level 4 --episodes 100000 --save_model level4_q_sarsa.pkl --no_render --algo sarsa
```

# Evaluate commands

UI mode (pygame):

```
python -m gridworld.evaluate --level 4 --model runs/gridworld/<run_name>/final/level4_q_sarsa.pkl --mode ui --render_delay 0.3

# eg:
python -m gridworld.evaluate --level 0 --model runs/gridworld/q_learning_level0_1767852881/final/q_level0.pkl --mode ui --render_delay 0.3
```

Headless batch evaluation (no rendering):

```
python -m gridworld.evaluate --level 4 --model runs/gridworld/<run_name>/final/level4_q_sarsa.pkl --mode headless --episodes 100
```
