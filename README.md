**COSC3145 | Games and Artificial Intelligence Techniques | Assignment 3 – Deep RL Arena & Gridworld**

> **Group:** 10
>
> **Student Name - Student ID:** Pham Vo Dong - s3891968
>
> **Student Name - Student ID:** Tran Nam Thai - s3891890
>
> **Student Name - Student ID:** Truong Nhat Anh - s3878231

---

## GAIT Assignment 3 – Deep RL Arena & Gridworld

This repository contains two main reinforcement learning environments used for the assignment:

- **`arena/`**: A Pygame-based Deep RL combat arena using SB3 (DQN, PPO, PPO-LSTM).
- **`gridworld/`**: A tabular RL Gridworld (Q-learning / SARSA) with headless and UI-based evaluation.

The root `makefile` is configured for the **arena** environment to simplify common training and evaluation workflows.

---

## 1. Project Layout (High Level)

- **`arena/`** – Deep RL arena package

  - `core/` – configs, device selection, environment wrappers
  - `game/` – game entities and physics
  - `training/` – algorithms (DQN, PPO, PPO-LSTM), runner, callbacks
  - `evaluation/` – evaluation utilities
  - `ui/` – renderer and menus
  - `train.py` – CLI entry for training (`python -m arena.train`)
  - `evaluate.py`, `eval_headless.py`, `eval_latest.py` – evaluation scripts

- **`gridworld/`** – Tabular RL Gridworld

  - `main.py` – training entrypoint
  - `evaluate.py` – evaluation (UI / headless)
  - See `gridworld/README.md` for detailed commands.

- **`tools/`** – Utility scripts (e.g. upload/evaluation helpers)
- **`tests/`** – Unit tests (e.g. `test_spawner_multiplier.py`)
- **Notebooks** – `rl_training.ipynb`, `model_comparison.ipynb` for analysis & plots

---

## 2. Environment & Installation

This project is configured to work well with **uv** (recommended) but can also be run with plain `pip`.  
All examples below **do not use the makefile** – they call Python modules directly.

- **Python version**: use the version compatible with `pyproject.toml` / `requirements.txt` (typically 3.10+).

### 2.1. Using `uv` (recommended)

From the repository root:

```bash
# Example: show training CLI help
uv run python -m arena.train --help
```

You do **not** need to manually create or activate a virtual environment when using `uv`.

### 2.2. Using `pip`

If you prefer a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # on macOS/Linux
# .venv\Scripts\activate   # on Windows (PowerShell/CMD)

pip install -r requirements.txt
```

---

## 3. Running the Arena - Starwars Game (Direct Commands)

All of the following commands are run from the **repository root** and do **not** use `make`.

### 3.1. Training

```bash
# DQN, Style 1
uv run python -m arena.train --algo dqn --style 1 --steps 1000000

# PPO, Style 2
uv run python -m arena.train --algo ppo --style 2 --steps 20000000 --no-render --device cpu

# PPO-LSTM, Style 1
uv run python -m arena.train --algo ppo_lstm --style 1 --steps 10000000 --no-render --device cpu
```

To resume from a checkpoint:

```bash
uv run python -m arena.train \
  --algo ppo \
  --style 1 \
  --load-model runs/ppo/style1/<your_run>/checkpoints/<checkpoint>.zip
```

### 3.2. Evaluation

Interactive UI evaluation:

```bash
uv run python -m arena.evaluate --device cpu
```

Headless evaluation of a single model:

```bash
uv run python -m arena.eval_headless \
  --model runs/ppo/style1/<your_checkpoint>.zip \
  --episodes 100 \
  --device cpu \
  --workers 10 \
  --stochastic
```

Headless evaluation of multiple models:

```bash
uv run python -m arena.eval_headless \
  --models <model1.zip> <model2.zip> ... \
  --workers 10 \
  --episodes 100 \
  --stochastic \
  --csv comparison_s1_curriculum.csv
```

---

## 4. Gridworld (Brief)

The **Gridworld** environment lives in `gridworld/`. Key example commands (see `gridworld/README.md` for more):

```bash
# Train (example)
uv run python -m gridworld.main --level 4 --episodes 100000 --save_model level4_q_sarsa.pkl --no_render --algo sarsa

# Evaluate in UI mode
uv run python -m gridworld.evaluate --level 4 --model runs/gridworld/<run_name>/final/level4_q_sarsa.pkl --mode ui --render_delay 0.3

# Evaluate headless
uv run python -m gridworld.evaluate --level 4 --model runs/gridworld/<run_name>/final/level4_q_sarsa.pkl --mode headless --episodes 100
```

---

## 5. Plots & Analysis

### 5.1 Notebook

The root-level notebooks (`rl_training.ipynb`, `model_comparison.ipynb`) read from generated CSVs (e.g. `comparison_s1_curriculum.csv`, `comparison_s2.csv`) and produce visualizations saved under `model_performance/` and `outcomes/`.  
Ensure the CSVs are regenerated via the `eval_headless_*` make targets before re-running the notebooks if you want up-to-date charts.

### 5.2 TensorBoard

```bash
uv run tensorboard --logdir runs
```

Run the above command and navigate using left panel to see logs in TensorBoard

