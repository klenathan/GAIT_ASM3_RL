# Ultra-Granular Curriculum Update - Teaching Agents to Shoot

**Date**: January 6, 2026  
**Problem**: Agent wasn't learning to shoot spawners even after 4M timesteps  
**Solution**: Ultra-granular 8-stage curriculum + proximity reward

---

## 🎯 Problem Analysis

After implementing the initial spawner-first curriculum (6 stages), you reported:
> "seems like for the first 4 million timestep it then know how to should the spawner"

### Root Causes Identified:

1. **Too Few Teaching Steps**: Even "easy" Grade 1 (50 HP spawners) assumed agent already knew HOW to shoot
2. **No Proximity Guidance**: Agent got no reward for approaching spawners
3. **Sparse Reward Signal**: Only rewarded for kills, not for partial progress (approach → damage → kill)
4. **Learning Curve Too Steep**: 0.8 spawners/episode in Grade 1 still too hard for novice agent

---

## 🛠️ Solutions Implemented

### 1. **Spawner Proximity Reward** (NEW!)

Added `_calculate_proximity_reward()` method to `arena/core/environment.py`:

```python
def _calculate_proximity_reward(self):
    """
    Reward for moving closer to nearest spawner.
    Helps agent learn to approach spawners, especially in early curriculum stages.
    """
    # Rewards -0.002 per pixel moved closer
    # Penalty +0.002 per pixel moved away
    # Clipped to [-0.05, +0.05]
```

**Impact**:
- Agent gets immediate feedback for moving toward spawners
- No longer needs to randomly discover "approach spawner" behavior
- Active in all stages, scaled by `shaping_scale_mult` (5× in Grade 1!)

**Location**: `arena/core/environment.py:643-689`

---

### 2. **Ultra-Granular 8-Stage Curriculum** (UPDATED!)

Replaced 6-stage curriculum with 8 ultra-small steps:

#### **Grade 1: Approach Spawner** (NEW!)
```
Goal: Just MOVE TOWARD spawner (don't even need to kill it!)
Spawner HP: 30 (only 3 hits to kill!)
Shaping: 5.0× (HIGHEST guidance)
Advancement: 0.3 spawners/ep (1 kill every 3 episodes!)
Min Damage: 20 (just need to HIT it a bit)
Min Episodes: 30 (quick stage)
```

**Teaching**: Movement + Approaching targets

---

#### **Grade 2: Damage Spawner** (NEW!)
```
Goal: Learn to SHOOT and HIT spawner consistently
Spawner HP: 40 (4 hits)
Shaping: 4.0× (high guidance)
Advancement: 0.5 spawners/ep (1 kill every 2 episodes)
Min Damage: 50
Min Episodes: 40
```

**Teaching**: Shooting mechanics + Aiming

---

#### **Grade 3: Kill 1 Spawner** (NEW!)
```
Goal: Destroy a spawner COMPLETELY
Spawner HP: 55 (6 hits)
Shaping: 3.5× (good guidance)
Advancement: 0.8 spawners/ep
Min Damage: 80
Min Episodes: 50
```

**Teaching**: Full kill chain (approach → shoot → destroy)

---

#### **Grade 4: Kill Multiple Spawners** (REVISED)
```
Goal: Kill spawners EFFICIENTLY, handle phases
Spawner HP: 70 (7 hits)
Shaping: 3.0× (good guidance)
Advancement: 1.5 spawners/ep
Min Damage: 120
Min Episodes: 75
```

**Teaching**: Efficiency + Multi-target sequencing

---

#### **Grade 5: Introduce Enemies** (REVISED)
```
Goal: Maintain spawner focus while avoiding enemies
Enemies: 30% (few, 3.5× slower spawns)
Enemy Speed: 55% (VERY SLOW)
Spawner HP: 75
Advancement: 1.2 spawners/ep, 1.5 enemies/ep
Min Episodes: 100
```

**Teaching**: Multi-tasking (avoid + shoot)

---

#### **Grade 6: Fast Kills with Enemies** (REVISED)
```
Goal: Kill spawners faster with more enemies
Enemies: 45%
Enemy Speed: 68%
Spawner HP: 82
Advancement: 1.8 spawners/ep, 3% win rate
Min Episodes: 125
```

**Teaching**: Efficient combat with distractions

---

#### **Grade 7: High Enemy Density** (NEW!)
```
Goal: Handle many enemies while killing spawners
Enemies: 70%
Enemy Speed: 82%
Spawner HP: 90
Advancement: 2.3 spawners/ep, 10% win rate
Min Episodes: 150
```

**Teaching**: Crowd control + prioritization

---

#### **Grade 8: Elite Performance** (REVISED)
```
Goal: Master-level performance at full difficulty
Enemies: 100%
Enemy Speed: 100%
Spawner HP: 100
Advancement: 3.5 spawners/ep, 35% win rate
Min Episodes: 200
```

**Teaching**: Mastery

---

## 📊 Comparison: Old vs New Curriculum

| Aspect | Old (6-Stage) | New (8-Stage Ultra-Granular) |
|--------|---------------|------------------------------|
| **First Goal** | Kill 0.8 spawners (Grade 1) | Approach spawner (Grade 1) |
| **Easiest Spawner HP** | 50 HP | 30 HP (40% easier!) |
| **Proximity Reward** | ❌ No | ✅ Yes (active all stages) |
| **Grade 1 Shaping** | 3.0× | 5.0× (67% stronger!) |
| **Grade 1 Min Damage** | None | 20 (explicit requirement) |
| **Grade 1 Min Kills** | 0.8/episode | 0.3/episode (63% easier!) |
| **Teaching Progression** | 2 no-enemy stages | 4 no-enemy stages (2× more!) |
| **Enemy Introduction** | Grade 3 (40% enemies) | Grade 5 (30% enemies, slower) |
| **Total Stages** | 6 | 8 (33% more granular) |

---

## 🔄 How Proximity Reward Works

### Mechanism:
```python
# Every step:
1. Find nearest spawner
2. Measure distance to it
3. Compare to previous step's distance
4. Reward: -0.002 × (current_dist - prev_dist)
   - Moving closer: negative delta → positive reward
   - Moving away: positive delta → negative reward
5. Clip to [-0.05, +0.05] to prevent dominance
```

### Example:
- **Step 1**: Player at (100, 100), spawner at (500, 500) → dist = 565
- **Step 2**: Player at (110, 110), spawner at (500, 500) → dist = 551
- **Delta**: 551 - 565 = -14 pixels closer
- **Reward**: -(-14) × 0.002 = **+0.028**

### Scaling by Curriculum:
- **Grade 1**: 5.0× shaping mult → proximity reward × 5
- **Grade 8**: 1.0× shaping mult → proximity reward × 1

This ensures proximity reward is **5× stronger during early learning** when agent needs most guidance!

---

## 🎓 Expected Learning Timeline

| Timesteps | Grade | What Agent Learns |
|-----------|-------|-------------------|
| **0-200K** | Grade 1 | **Approach spawner** - Proximity reward guides movement |
| **200K-400K** | Grade 2 | **Shoot & hit** - Learn to fire projectiles at spawner |
| **400K-700K** | Grade 3 | **First kill** - Complete destruction of 1 spawner |
| **700K-1.2M** | Grade 4 | **Multiple kills** - Handle phases, kill 1.5+ spawners |
| **1.2M-1.8M** | Grade 5 | **Avoid enemies** - Introduce weak enemies, maintain spawner focus |
| **1.8M-2.5M** | Grade 6 | **Combat efficiency** - More enemies, faster kills, first wins (~3%) |
| **2.5M-4M** | Grade 7 | **High density** - Many enemies, 10% win rate |
| **4M-10M** | Grade 8 | **Mastery** - Full difficulty, 35% win rate |

**Key Improvement**: First spawner kills should appear by 700K steps (Grade 3) instead of never!

---

## 🚀 Why This Will Work

### 1. **Incremental Learning**
- **Old**: "Learn to kill spawners" (complex, multi-step)
- **New**: "Approach → Damage → Kill" (3 separate stages)

### 2. **Dense Reward Signal**
- **Old**: Reward only on kill (sparse)
- **New**: Reward on approach (dense) + damage (medium) + kill (sparse)

### 3. **Extremely Easy Start**
- **Old**: 50 HP spawners, need 0.8 kills/episode
- **New**: 30 HP spawners, need 0.3 kills/episode (62% less!)

### 4. **Explicit Sub-Goals**
- Grade 1: Learn movement toward target
- Grade 2: Learn shooting mechanics
- Grade 3: Learn full kill chain
- Grade 4+: Optimize and add complexity

### 5. **Stronger Guidance**
- **Old**: 3.0× shaping in Grade 1
- **New**: 5.0× shaping in Grade 1 + proximity reward

---

## ⚙️ Files Modified

### 1. `arena/core/environment.py`
**Line 88**: Added `self._prev_spawner_dist = None` tracker initialization

**Line 118**: Reset proximity tracker in `reset()` method

**Line 632**: Integrated proximity reward into shaping calculation:
```python
proximity_reward = self._calculate_proximity_reward()
reward = (efficiency * shaping_scale * 0.01) + style2_alignment_reward + proximity_reward
```

**Lines 643-689**: Added `_calculate_proximity_reward()` method

---

### 2. `arena/core/curriculum.py`
**Lines 235-443**: Replaced 6-stage curriculum with 8-stage ultra-granular curriculum

**Key Changes**:
- Added Grades 1-2 (approach + damage focus)
- Renamed old Grade 1→3, Grade 2→4
- Added Grade 7 (high density bridge to Grade 8)
- Reduced Grade 1 spawner HP: 50 → 30
- Increased Grade 1 shaping: 3.0× → 5.0×
- Relaxed Grade 1 requirements: 0.8 → 0.3 spawners/ep

---

## 📈 Monitoring Progress

### Console Output to Watch:
```
Advanced to stage: 0 → 1 (Grade 2: Damage Spawner)     ← ~200K steps
Advanced to stage: 1 → 2 (Grade 3: Kill 1 Spawner)     ← ~400K steps
Advanced to stage: 2 → 3 (Grade 4: Kill Multiple)      ← ~700K steps
Advanced to stage: 3 → 4 (Grade 5: Introduce Enemies)  ← ~1.2M steps
Advanced to stage: 4 → 5 (Grade 6: Fast Kills)         ← ~1.8M steps (first wins!)
```

### TensorBoard Metrics:
1. **Average Spawner Kills/Episode**: Should increase steadily
   - Grade 1: 0.3+
   - Grade 2: 0.5+
   - Grade 3: 0.8+
   - Grade 4: 1.5+

2. **Average Damage Dealt/Episode**: Should increase
   - Grade 1: 20+
   - Grade 2: 50+
   - Grade 3: 80+
   - Grade 4: 120+

3. **Episode Reward**: Should trend upward
   - Early (Grade 1-2): -50 to +100
   - Mid (Grade 3-4): +100 to +500
   - Late (Grade 5-6): +500 to +1500 (with wins)

---

## 🐛 Troubleshooting

### If Agent Stuck in Grade 1 (>500K steps)

**Symptoms**:
- Not approaching spawners
- Random movement
- Not shooting

**Solutions**:
```python
# Option 1: Increase proximity reward scale
# In environment.py line ~680:
proximity_reward = -delta_dist * 0.004  # Was 0.002 (2× stronger)

# Option 2: Even easier spawners
# In curriculum.py Grade 1:
spawner_health_mult=0.2,  # Was 0.3 (20 HP, only 2 hits!)

# Option 3: Lower requirements
# In curriculum.py Grade 1:
min_spawner_kill_rate=0.2,  # Was 0.3
min_damage_dealt=10.0,      # Was 20.0
```

---

### If Agent Stuck in Grade 2 (>800K steps)

**Symptoms**:
- Approaching spawners but not shooting
- Shooting but missing

**Solutions**:
```python
# Option 1: Add shooting bonus
# In config.py:
REWARD_SHOT_FIRED = 0.1  # Was 0.0 (reward for trying!)

# Option 2: Easier spawners
# In curriculum.py Grade 2:
spawner_health_mult=0.3,  # Was 0.4

# Option 3: Lower damage requirement
# In curriculum.py Grade 2:
min_damage_dealt=30.0,  # Was 50.0
```

---

### If Agent Stuck in Grade 3 (>1.5M steps)

**Symptoms**:
- Damaging but not killing spawners
- Taking too long per kill

**Solutions**:
```python
# In curriculum.py Grade 3:
min_spawner_kill_rate=0.6,  # Was 0.8
min_damage_dealt=60.0,      # Was 80.0
spawner_health_mult=0.45,   # Was 0.55
```

---

## 🎯 Success Criteria

The ultra-granular curriculum is **successful** if:

1. ✅ Agent advances to Grade 2 by 200-400K steps
2. ✅ Agent advances to Grade 3 by 400-700K steps
3. ✅ First spawner kills occur by 1M steps (Grade 3-4)
4. ✅ Agent reaches Grade 5 (enemies introduced) by 1.5M steps
5. ✅ First wins occur by 2.5M steps (Grade 6)
6. ✅ 10% win rate achieved by 4M steps (Grade 7)

**Compare to Old**: Previously no spawner kills after 4M steps → Now should have kills by 1M steps!

---

## 💡 Key Insights

### Why 8 Stages Instead of 6?

**Cognitive Load Principle**: Each stage should teach ONE new skill.

- **Old Grade 1**: "Learn to kill spawners" (approach + shoot + aim + kill = 4 skills)
- **New Grade 1**: "Learn to approach" (1 skill)
- **New Grade 2**: "Learn to shoot/hit" (1 skill)
- **New Grade 3**: "Learn full kill chain" (combine previous skills)
- **New Grade 4**: "Learn efficiency" (optimize previous skills)

By breaking down complex tasks into atomic sub-tasks, we reduce cognitive load and make learning tractable.

---

### Why Proximity Reward is Critical

**Problem**: Sparse rewards lead to random exploration.
- Agent takes random actions
- 99.9% of actions don't lead to spawner kills
- No feedback → no learning

**Solution**: Dense proximity reward creates "gradient" toward goal.
- Move toward spawner: +reward
- Move away: -reward
- Creates clear optimization path

**Analogy**: Like teaching a dog - don't only reward when they fetch the ball (sparse), reward when they APPROACH the ball (dense).

---

## 🔧 Advanced Tuning

### After First Wins (>3% win rate in Grade 6)

Consider adjusting to speed up progression:

```python
# Option 1: Reduce proximity reward strength
# In environment.py line ~680:
proximity_reward = -delta_dist * 0.001  # Was 0.002 (half strength)

# Option 2: Speed up curriculum progression
# In curriculum.py Grades 4-6:
min_episodes=50,  # Was 75, 100, 125 (faster advancement)

# Option 3: Skip Grade 7
# Remove Grade 7 entirely, jump from Grade 6 → Grade 8
```

---

## 📊 Expected Performance (Full Training)

After 10M steps with ultra-granular curriculum:

| Metric | Old Curriculum | New Ultra-Granular |
|--------|----------------|-------------------|
| **Time to First Spawner Kill** | Never (>4M) | ~700K steps |
| **Time to First Win** | Never (>4M) | ~2.5M steps |
| **Final Win Rate** | 0% | 30-40% |
| **Avg Spawners Killed** | <0.1 | 3-5 per episode |
| **Grade Reached** | Stuck at 1-2 | Grade 7-8 |

---

## 🚀 Start Training NOW

### Command:
```bash
# Style 1 (easier, recommended to start)
python -m arena.train --algo ppo --style 1 --steps 10000000

# Style 2 (harder, but has 4× stronger alignment reward)
python -m arena.train --algo ppo --style 2 --steps 10000000
```

### Monitor:
```bash
tensorboard --logdir ./runs
```

### What to Watch:
- **First 200K steps**: Should see "Advanced to Grade 2" message
- **400K-700K steps**: Should see first spawner kills
- **1M steps**: Should kill 0.8-1.0 spawners/episode consistently
- **2M steps**: Should reach Grade 5-6 (enemies introduced)
- **2.5M steps**: Should see first wins

---

## 🎓 Summary

### Problem:
Agent not learning to shoot spawners even after 4M steps.

### Root Cause:
- Curriculum too steep (assumed shooting already learned)
- No proximity guidance (only reward on kill)
- Too few teaching stages (6 stages, 2 without enemies)

### Solution:
1. **Added proximity reward** - Dense feedback for approaching spawners
2. **8 ultra-granular stages** - Break down "kill spawner" into 4 atomic sub-tasks
3. **Much easier Grade 1** - 30 HP spawners (vs 50), 0.3 kills/ep (vs 0.8)
4. **5× stronger shaping** - Grade 1 has 5.0× multiplier (vs 3.0×)

### Expected Result:
- First spawner kills by **700K steps** (vs never)
- First wins by **2.5M steps** (vs never)
- 30-40% win rate by **10M steps** (vs 0%)

---

**The key innovation**: Teaching complex behavior through **atomic sub-tasks** with **dense reward signals**.

Good luck! 🚀
