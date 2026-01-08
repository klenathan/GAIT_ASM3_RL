# GridWorld Report

This report documents the `gridworld/` tabular reinforcement learning (RL) implementation, focusing on (I) environment descriptions, (II) observation/state design, (III) reward design, and (IV) hyperparameter exploration.

## I. Description of Environments

### Core environment rules

The environment is implemented in `gridworld/environment.py` as the `GridWorld` class.

- **Grid size**: fixed at **10×10** (`GRID_WIDTH = GRID_HEIGHT = 10` in `gridworld/config.py`).
- **Agent start**: resets to the top-left corner at `(row=0, col=0)`.
- **Actions**: 4 discrete moves (no-op is not present):
  - `0`: Up, `1`: Down, `2`: Left, `3`: Right
- **Movement constraints**:
  - The agent cannot leave the grid.
  - The agent cannot enter **rocks** (`'R'`), so moves into rocks are ignored.
- **Terminal conditions**:
  - Entering **fire** (`'F'`) ends the episode (death).
  - Sharing a cell with a **monster** (`'M'`) ends the episode (death).
  - Completing all objectives ends the episode (win).

### Entities / tiles

The level layouts are ASCII maps in `gridworld/config.py` (`LEVEL_0 ... LEVEL_6`). Tiles are parsed on reset.

- `.` empty space
- `R` rock / wall (impassable)
- `F` fire (hazard; terminal)
- `A` apple (collectible)
- `C` chest (collectible but requires key)
- `K` key (enables opening any chest)
- `M` monster (hazard that moves stochastically)

### Objectives and success criteria

The win condition is checked each step:

- **Win** if **all apples are collected** AND **all chests (if any) are opened**.

This is implemented by comparing:

- `len(collected_apples) == len(apples)`
- `len(opened_chests) == len(chests)`

### Monster dynamics

Monsters are represented as mutable `[r, c]` lists and move after the agent acts.

- For each monster, there is a **40% chance** of moving on a given step.
- If moving, the monster picks a random direction uniformly from {up, down, left, right}.
- Monsters obey bounds and cannot enter rocks.
- After monsters move, collisions with the agent are checked again.

### Level overview (0–6)

The code currently defines 7 levels in `LEVELS`:

- **Level 0 (basic navigation)**: mostly empty grid with a single apple near the bottom-right (`"........A."` on the last row).
- **Level 1 (obstacles + hazard strip)**:
  - Two horizontal rock barriers.
  - A horizontal fire strip near the bottom.
  - One apple near the bottom-right.
- **Level 2 (key + chest + obstacles)**:
  - Alternating rock columns near the top.
  - One key (`K`) placed roughly mid-map.
  - One chest (`C`) at the right edge.
  - One apple near the bottom-right.
- **Level 3**: currently set equal to Level 2 (`LEVEL_3 = LEVEL_2` placeholder).
- **Level 4 (monsters)**:
  - Two monsters placed in the open area.
  - One apple near the bottom-right.
- **Level 5**: currently set equal to Level 4 (`LEVEL_5 = LEVEL_4` placeholder).
- **Level 6 (intrinsic reward level)**: currently set equal to Level 0 (`LEVEL_6 = LEVEL_0`), but includes bookkeeping for visitation counts.

## II. Observation Design

### What the agent receives (state)

The environment exposes state via `GridWorld.get_state()`.

A key design decision is that the state representation *changes by level index*:

1. **Levels 0–1: minimal state**

For level indices `<= 1`, the state is just the agent position:

- `state = (agent_row, agent_col)`

This creates a small, fixed state space of up to 100 states; ideal for demonstrating shortest-path style learning with tabular methods.

2. **Levels >= 2: extended state with objectives**

For more complex levels, the state includes both position and progress flags:

- `state = ( (agent_row, agent_col), has_key, collected_apples, opened_chests )`

Where:

- `has_key` is a boolean
- `collected_apples` is a sorted tuple of apple coordinates visited/collected
- `opened_chests` is a sorted tuple of chest coordinates opened

This design makes the state **Markov** with respect to the environment’s objectives (i.e., whether rewards remain available and whether the episode can terminate).

### Tabular compatibility and state-space growth

The Q-learning/SARSA agents are tabular (`q_table: Dict[state, np.ndarray]` in `gridworld/agent.py`).

- With `(r,c)` only, tabular learning is compact.
- Adding item progress makes learning tractable only if the number of items is small.
  - The code attempts to control this by encoding objective progress as tuples of coordinates, which is general but can grow combinatorially if many items were added.

### Observation vs. rendering

The renderer (`gridworld/renderer.py`) visualizes the full grid (agent, rocks, hazards, items, monsters), but the observation passed to the agent is only the `get_state()` representation above.

This is a key separation:

- **Renderer**: for human understanding/debugging.
- **Agent state**: minimal information necessary for tabular learning.

## III. Reward Design

Rewards are defined in `gridworld/config.py`.

### Extrinsic rewards

Each step returns a scalar reward, accumulated as:

- **Step penalty**: `REWARD_STEP = -0.01`
  - Encourages shorter paths and prevents wandering.
- **Apple reward**: `REWARD_APPLE = +1.0`
  - Awarded once per apple when first collected.
- **Chest reward**: `REWARD_CHEST = +2.0`
  - Awarded once per chest when opened (requires key).
- **Death penalty**: `REWARD_DEATH = -10.0`
  - Applied when stepping into fire or colliding with a monster.
- **Win bonus**: `REWARD_WIN = +10.0`
  - Applied when all objectives are completed.

Total reward per transition is typically:

- `reward = REWARD_STEP + (optional item reward) + (optional win bonus)`

Note: death returns immediately with `REWARD_DEATH` (not also adding `REWARD_STEP`).

### Key mechanics and reward shaping

When the agent picks up a key (`K`), the environment sets `has_key = True` but **does not provide a reward** (comment indicates the intended behavior: “no reward but allow opening chests”).

This is a deliberate shaping choice:

- It avoids giving reward for an intermediate subgoal unless the task explicitly wants it.
- It makes the key’s value purely instrumental (it unlocks access to chest reward).

### Intrinsic reward (planned / partial)

The environment maintains `visit_counts` for “intrinsic reward (Level 6)” and increments it at every step using the agent’s `(r,c)` position.

However, in the current `step()` implementation, **no intrinsic bonus is actually added to `reward`**. In other words:

- `visit_counts` is recorded,
- but it is not used to shape reward.

If intrinsic exploration is desired for Level 6, a common approach would be something like:

- `reward += beta / sqrt(visit_counts[state])` (count-based exploration)

…but as written, Level 6 behaves like Level 0 in terms of reward.

## IV. Hyperparameter Exploration

Training is driven by `gridworld/main.py`, which supports Q-learning and SARSA with the following hyperparameters from `gridworld/config.py`:

- `ALPHA = 0.1` (learning rate)
- `GAMMA = 0.99` (discount)
- Epsilon-greedy schedule:
  - `EPSILON_START = 1.0`
  - `EPSILON_END = 0.01`
  - `EPSILON_DECAY = 0.995` (multiplicative decay per episode)

The repository currently does not include a dedicated sweep runner (and `gridworld/train.py` is a placeholder), but the components are suitable for systematic exploration.

### Recommended exploration methodology

A practical sweep plan for this codebase:

1. Choose a representative set of levels
   - **Level 0/1**: shortest-path + obstacles + fire
   - **Level 2**: sparse reward with key→chest dependency
   - **Level 4**: stochastic dynamics due to monster movement

2. Evaluate both algorithms under controlled settings
   - Compare Q-learning vs SARSA in stochastic levels (Level 4), where SARSA’s on-policy nature can be more conservative.

3. Track metrics
   - Average return over last N episodes (e.g., N=50)
   - Success rate (fraction of episodes ending with win)
   - Steps-to-complete for successful episodes

### Hyperparameters to vary (and expected effects)

#### Learning rate (`ALPHA`)
Suggested values: `{0.05, 0.1, 0.2, 0.5}`

- Too low: slow learning, especially in sparse reward levels.
- Too high: unstable Q-values and oscillation, worse with stochastic monsters.

#### Discount factor (`GAMMA`)
Suggested values: `{0.90, 0.95, 0.99}`

- Lower gamma can prioritize immediate apple reward and avoid long planning.
- Higher gamma helps planning for key→chest dependencies (Level 2).

#### Exploration schedule (`EPSILON_*`)
Suggested exploration knobs:

- `EPSILON_DECAY` in `{0.99, 0.995, 0.999}`
- `EPSILON_END` in `{0.01, 0.05, 0.1}`

Expected behavior:

- Faster decay can work on simple maps (Level 0), but risks premature convergence on harder maps.
- Higher ε_end can help in non-stationary/stochastic settings (Level 4 monsters) but may reduce asymptotic performance.

### Notes about implementation constraints

- The agent uses a Python dict Q-table keyed by the state tuple; this makes sweeps easy, but memory can grow if the state is large (Levels ≥ 2).
- Because episodes end on death, hazards can dominate returns; for stable comparisons, report both return and win-rate.

---

**Files referenced**: `gridworld/environment.py`, `gridworld/config.py`, `gridworld/agent.py`, `gridworld/main.py`, `gridworld/renderer.py`.
