# GridWorld Report

This report documents the `gridworld/` tabular RL implementation, focusing on:

I. Description of environments  
II. Observation design  
III. Reward design  
IV. Hyperparameter exploration

The descriptions below reflect the **current code** in `gridworld/`.

## I. Description of Environments

### Core environment rules

The environment is implemented by `GridWorld` in `gridworld/environment.py`.

- **Grid size**: fixed at **10×10** (`GRID_WIDTH = GRID_HEIGHT = 10` in `gridworld/config.py`).
- **Agent start**: `(row=0, col=0)` on every `reset()`.
- **Actions**: 4 discrete moves
  - `0`: Up, `1`: Down, `2`: Left, `3`: Right
- **Collisions / movement constraints**:
  - Off-grid moves are ignored.
  - Moves into **rocks** (`R`) are ignored.
- **Termination conditions**:
  - **Fire** (`F`) is terminal with death.
  - **Monster** (`M`) collision is terminal with death (checked both before and after monsters move).
  - **Win** occurs when all objectives are completed (all apples collected and all chests opened).

### Entities and their mechanics

Levels are defined as ASCII layouts in `gridworld/config.py` and parsed during `reset()`.

- `.` empty cell
- `R` rock (impassable)
- `F` fire (hazard; terminal)
- `A` apple (collectible)
- `K` key (enables opening chests)
- `C` chest (collectible; only opens if the agent has a key)
- `M` monster (hazard; stochastic movement)

### Objective structure

The environment supports multiple objectives:

- **Apples**: each apple gives reward once and then is considered collected.
- **Chests**: each chest gives reward once but only after the key is obtained.

Win condition (checked every step):

- `len(collected_apples) == len(apples)` AND `len(opened_chests) == len(chests)`.

This means a level with *no chests* is winnable by collecting all apples only.

### Monster dynamics (stochastic transitions)

Monsters move after the agent acts:

- Each monster has a **40% chance** to move each step.
- A moving monster chooses uniformly among the four cardinal directions.
- Monsters cannot leave the grid and cannot enter rocks.
- If a monster ends up on the agent after moving, the agent dies immediately.

This makes the environment **stochastic**, and the transition distribution depends partly on random monster actions.

### Updated level overview (0–6)

The code defines seven distinct levels in `LEVELS`.

- **Level 0 (basic shortest path)**: empty grid with a single apple near the bottom-right (last row contains `"........A."`).
- **Level 1 (maze-like rocks + fire strip)**: denser rock layout than before plus a fire band near the bottom, with an apple at the goal region. A backup layout exists as `LEVEL_1_BACKUP`.
- **Level 2 (multi-apple + key + chest)**: multiple apples, one key, and one chest; includes rock patterns creating corridors.
- **Level 3 (harder key/chest variant)**: custom layout with repeated wall blocks (`RR.RR.RR.` style), with key, apples, and a chest.
- **Level 4 (monsters introduced)**: two monsters placed in open space; one apple goal.
- **Level 5 (monsters + rocks)**: multiple monsters embedded inside a more constrained rock pattern; one apple goal.
- **Level 6 (exploration / intrinsic-reward setting)**: alternating rock rows create repeated barriers; one apple goal. This level is designed to illustrate exploration effects when intrinsic reward is enabled.

## II. Observation Design

### State returned to the agent

The agent receives a **tabular** state (hashable Python tuples) via `GridWorld.get_state()`.

A key code change is that the state is now **constructed from components depending on which entities exist in the level**, rather than always returning a fixed schema.

#### Case A: simple levels (0–1 without monsters)

If `level_idx <= 1` and there are no monsters, the state is just the agent position:

- `state = (agent_row, agent_col)`

This yields at most 100 states, which is ideal for demonstrating shortest-path learning with a small Q-table.

#### Case B: levels with objectives and/or monsters

For more complex levels, the state is a tuple of components:

1. Agent position: `(agent_row, agent_col)`
2. `has_key` (only if keys exist or a key has been collected)
3. Remaining apples: `tuple(sorted(apples not yet collected))`
4. Remaining chests: `tuple(sorted(chests not yet opened))`
5. Monster positions (if monsters exist): `tuple(sorted(monster_positions))`

So conceptually:

- `state = (agent_pos, [has_key], remaining_apples, remaining_chests, [monster_positions])`

### Design rationale

- **Markov property**: remaining-apples/remaining-chests + key status make it possible to infer what rewards remain available and whether the episode can terminate.
- **Compactness vs. expressiveness trade-off**:
  - Tracking *remaining* items (rather than collected items) keeps the representation aligned with “what is left to do”.
  - Including monsters in the state allows the agent to condition behavior on monster proximity, but can increase state space significantly.

### Implications for tabular learning

The agents in `gridworld/agent.py` implement tabular Q-learning and SARSA using a Python dict `q_table[state] -> np.ndarray(|A|)`.

- Levels 0–1: very compact and stable.
- Levels 2–3: state space grows with the power set of collectibles (apples/chests), but remains manageable because the layouts contain only a few items.
- Levels 4–5: state space can grow substantially due to monster positions (even though movement is stochastic).

## III. Reward Design

Rewards are defined in `gridworld/config.py` and applied in `GridWorld.step()`.

### Extrinsic reward components

- **Step penalty**: `REWARD_STEP = -0.01` (applied every step, encouraging shorter solutions)
- **Apple**: `REWARD_APPLE = +1.0` when collected first time
- **Chest**: `REWARD_CHEST = +2.0` when opened (requires `has_key=True`)
- **Death**: `REWARD_DEATH = -10.0` on fire/monster collision, terminates immediately
- **Win bonus**: `REWARD_WIN = +10.0` added when all objectives are complete

Typical non-terminal transition reward:

- `reward = REWARD_STEP + [optional apple/chest] + [optional intrinsic]`

Terminal win transition reward:

- `reward = REWARD_STEP + [optional apple/chest] + REWARD_WIN + [optional intrinsic]`

Terminal death transition reward:

- `reward = REWARD_DEATH` (returns immediately; no step penalty or intrinsic term is added on that transition).

### Key reward shaping

Picking up the key grants **no direct reward**; it only changes `has_key` so that chests can be opened later. This keeps the reward signal focused on task completion rather than intermediate milestones.

### Intrinsic reward (now implemented)

The code now supports optional intrinsic reward via:

- `GridWorld(..., use_intrinsic_reward=...)`
- In `step()`, when enabled: `intrinsic_reward = 1 / sqrt(N(s))`, where `N(s)` is the visit count of the agent’s current position.

The intrinsic term is **added on top of the environment step reward**:

- `reward = REWARD_STEP + intrinsic_reward + (optional extrinsic events)`

This design encourages exploration by making novel states more rewarding early in training.

## IV. Hyperparameter Exploration

### Training entry point and new training controls

Training is driven by `gridworld/main.py` (not `gridworld/train.py`, which is now an experiment launcher).

Important new training-related controls in `gridworld/main.py`:

- `--max_steps`: caps episode length (default 100)
- `--intrinsic`: enables intrinsic reward shaping
- `--save_model`, `--load_model`: save/load Q-tables (pickled) under `gridworld/models/`
- `--test`: disables learning (`alpha=0`) and exploration (`epsilon=0`) for evaluation
- `--checkpoint_interval`: periodic checkpoint saves when `--save_model` is used
- Optional TensorBoard logging if `torch.utils.tensorboard` is available:
  - `rollout/ep_rew_mean`, `train/epsilon`, `rollout/ep_len_mean`

### Algorithmic hyperparameters (tabular RL)

`gridworld/config.py` defines base defaults:

- `ALPHA = 0.1`
- `GAMMA = 0.99`
- `EPSILON_START = 1.0`
- `EPSILON_END = 0.01`

#### Key change: epsilon decay schedule is now linear

In `gridworld/main.py`, epsilon decay is computed as:

- `linear_decay = (EPSILON_START - EPSILON_END) / episodes`

And `BaseAgent.decay_epsilon()` updates epsilon via:

- `epsilon = max(epsilon_end, epsilon - epsilon_decay)`

So `epsilon_decay` is acting as a **per-episode linear decrement**, not a multiplicative factor.

### What to explore (recommended sweeps)

Even without changing code, you can explore the following axes (by editing `gridworld/config.py` and rerunning):

- **Learning rate (`ALPHA`)**: e.g., `{0.05, 0.1, 0.2, 0.5}`
  - Higher can learn faster but can destabilize in stochastic monster levels.
- **Discount (`GAMMA`)**: e.g., `{0.90, 0.95, 0.99}`
  - Higher helps long-horizon goals (key→chest + multi-apple collection).
- **Episode budget (`--episodes`)**: trade off convergence vs. runtime.
- **Episode cap (`--max_steps`)**: affects exploration pressure and return scale.
- **Intrinsic reward (`--intrinsic`)**:
  - Compare Level 6 baseline vs intrinsic shaping to quantify exploration benefits.

### Experiment orchestration (`gridworld/train.py`)

`gridworld/train.py` now contains a list of long-running experiment commands (multi-threaded runner) that execute `python3 main.py ...` across tasks/levels and track status in the terminal.

It encodes a curriculum-like set of runs, including:

- Level 0/1 (Q-learning)
- Level 1 (SARSA)
- Level 2/3 (Q-learning/SARSA)
- Level 4/5 (Q-learning/SARSA)
- Level 6 (Q-learning with and without `--intrinsic`)

### Suggested reporting metrics

For consistent hyperparameter comparison, track:

- Mean return over a sliding window (e.g., last 50 episodes)
- Win rate (% episodes terminating in win)
- Mean episode length
- Best-model moving average (the code uses an average over the last 10 episodes to decide model saving)

---

**Files referenced**: `gridworld/environment.py`, `gridworld/config.py`, `gridworld/agent.py`, `gridworld/main.py`, `gridworld/train.py`.
