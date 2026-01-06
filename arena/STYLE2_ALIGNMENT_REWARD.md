# Control Style 2 - Firing Line Alignment Reward Shaping

## Overview

This document describes the reward shaping implementation for **Control Style 2** in the Arena environment. Control Style 2 features a fixed shooting angle that is randomized at the start of each episode, creating a unique challenge where the agent must learn to position itself strategically rather than simply rotating to face targets.

## The Challenge

In Control Style 2:
- The agent has a **fixed rotation** (set randomly between 0-360° at episode start)
- It can only shoot in **one direction** (along the fixed angle)
- Movement is directional: Up, Down, Left, Right (4 directions) + Shoot action
- The agent must position itself so spawners align with its firing line

## The Solution: Firing Line Alignment Reward

The reward shaping encourages the agent to position itself optimally by considering three key factors:

### 1. Angular Alignment Score

Measures how well the nearest spawner aligns with the agent's fixed shooting direction.

```python
alignment_score = max(0, cos(angle_difference))
```

- **Perfect alignment (0°)**: score = 1.0
- **Moderate misalignment (45°)**: score ≈ 0.7
- **Perpendicular (90°)**: score = 0.0
- **Behind (180°)**: score = 0.0 (negative values clipped to 0)

### 2. Optimal Range Score

Encourages maintaining an effective distance from spawners.

- **Too close (< 200px)**: Linear penalty (dangerous, enemies spawn nearby)
- **Optimal range (200-600px)**: Full score = 1.0
- **Too far (> 600px)**: Gradual penalty (harder to hit)

```python
if distance < 200:
    range_score = distance / 200
elif distance > 600:
    range_score = max(0.3, 1.0 - (distance - 600) / max_distance)
else:
    range_score = 1.0
```

### 3. Clear Shot Bonus

Penalizes positioning when enemies block the line of fire.

- Uses geometric line-intersection detection
- Counts enemies within 50px perpendicular distance from firing line
- Applies penalty: `blocking_penalty = min(0.5, num_blocking * 0.15)`

## Reward Calculation

```python
base_reward = alignment_score * range_score
final_reward = base_reward * (1.0 - blocking_penalty)
scaled_reward = final_reward * STYLE2_ALIGNMENT_SCALE
```

Where `STYLE2_ALIGNMENT_SCALE = 0.02` (configured in `config.py`)

## Integration

The alignment reward is automatically added to the overall reward during each step when `control_style=2`:

```python
# In environment.py, _calculate_shaping_reward()
if self.control_style == 2:
    style2_alignment_reward = self._calculate_style2_alignment_reward()
    reward += style2_alignment_reward
```

## Test Results

From `test_style2_alignment.py`:

| Scenario | Alignment Reward |
|----------|------------------|
| Perfect Alignment (0°) | 0.0200 |
| Slight Misalignment (15°) | 0.0193 |
| Moderate Misalignment (45°) | 0.0141 |
| Perpendicular (90°) | 0.0000 |
| Behind (180°) | 0.0000 |

## Key Design Decisions

### 1. Why Fixed Shooting Angle?
Makes the task fundamentally different from Style 1, requiring spatial reasoning and positioning strategy rather than target tracking.

### 2. Why Random Angle Each Episode?
Prevents the agent from learning episode-specific strategies. It must learn general positioning principles that work regardless of shooting direction.

### 3. Why These Range Values?
- **200px minimum**: Prevents clustering near spawners where enemy spawns are dangerous
- **600px maximum**: Maintains reasonable shooting accuracy
- These values were tuned based on the arena size (1600x1280) and projectile behavior

### 4. Why Include Blocking Penalty?
Encourages the agent to find clear lines of fire, adding tactical depth and preventing naive "just align and shoot" strategies.

## Configuration

The reward scale can be adjusted in `arena/core/config.py`:

```python
# Control Style 2 specific: Firing line alignment reward scaling
STYLE2_ALIGNMENT_SCALE = 0.02  # Tune this to balance with other rewards
```

**Recommended values:**
- Start: 0.02 (subtle guidance)
- Increase if agent struggles to learn positioning: 0.03-0.05
- Decrease if agent over-optimizes alignment vs combat: 0.01

## Files Modified

1. **`arena/core/environment.py`**:
   - Modified `_calculate_shaping_reward()` to add Style 2 alignment reward
   - Added `_calculate_style2_alignment_reward()` method (lines 619-721)

2. **`arena/core/config.py`**:
   - Added `STYLE2_ALIGNMENT_SCALE = 0.02` parameter

3. **`test_style2_alignment.py`** (new):
   - Comprehensive test suite for the reward shaping
   - Tests various alignment scenarios
   - Validates reward calculation

## Usage Example

```python
from arena.core.environment import ArenaEnv

# Create Style 2 environment (alignment reward automatically included)
env = ArenaEnv(control_style=2, render_mode=None)

obs, info = env.reset()
# Player rotation is now fixed (random) for this episode

for step in range(1000):
    action = model.predict(obs)  # Agent learns to position optimally
    obs, reward, done, truncated, info = env.step(action)
    # reward includes alignment shaping automatically
    if done:
        break

env.close()
```

## Future Improvements

Potential enhancements:
1. **Dynamic scaling**: Adjust alignment reward based on curriculum stage
2. **Multi-spawner support**: Reward aligning with multiple spawners
3. **Predictive positioning**: Reward positioning that anticipates enemy movement
4. **Adaptive ranges**: Adjust optimal range based on current threat level

## Summary

The Firing Line Alignment reward shaping provides intelligent guidance for Control Style 2 agents, helping them learn to:
- Position themselves strategically given their fixed shooting constraint
- Maintain optimal engagement distances
- Find clear lines of fire
- Generalize across random shooting angles

This creates a unique gameplay pattern distinct from Style 1, emphasizing spatial awareness and tactical positioning over twitch reflexes.
