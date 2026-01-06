# 🚀 Quick Start: New Spawner-First Curriculum

**Last Updated**: January 6, 2026

---

## ⚡ TL;DR - What Changed?

We implemented a **spawner-first curriculum** to accelerate first wins:

1. **Grades 1-2**: NO enemies - learn spawner destruction without distractions
2. **Grade 3+**: Gradually introduce enemies
3. **2× more exploration** (PPO entropy: 0.05 → 0.10)
4. **4× stronger Style 2 alignment reward** (0.02 → 0.08)

**Expected Result**: First wins in 2-3M steps instead of 5-10M+ steps (2-5× faster)

---

## 🏃 Start Training NOW

### Style 1 (Rotation + Thrust - Easier)
```bash
python -m arena.train --algo ppo --style 1 --steps 10000000
```

### Style 2 (Fixed Nozzle - Harder)
```bash
python -m arena.train --algo ppo --style 2 --steps 10000000
```

### Monitor Progress
```bash
# In a separate terminal
tensorboard --logdir ./runs
```

Then open: http://localhost:6006

---

## 📊 What to Watch For

### Console Output
Look for these messages:
```
Advanced to stage: 1 → 2 (Grade 2: Fast Spawner Kills)
Advanced to stage: 2 → 3 (Grade 3: Enemy Introduction)
Advanced to stage: 3 → 4 (Grade 4: Fast Kills with Enemies)  ← FIRST WINS HERE!
```

### TensorBoard Metrics
- **Spawner kills per episode**: Should increase steadily
- **Win rate**: Should appear in Grade 4 (around 2-3M steps)
- **Episode reward**: Watch for spikes (wins = ~1,500-2,000 points)

---

## ⏱️ Expected Timeline

| Steps | Grade | Milestone |
|-------|-------|-----------|
| 0-500K | Grade 1 | Learning to approach & shoot spawners (no enemies) |
| 500K-1M | Grade 2 | Getting faster at killing spawners (still no enemies) |
| 1M-2M | Grade 3 | Learning to dodge slow enemies while shooting spawners |
| 2M-3M | Grade 4 | **🎉 FIRST WINS** (5% win rate target) |
| 3M-5M | Grade 5 | Optimizing with more enemies (15% win rate target) |
| 5M-10M | Grade 6 | Elite performance at full difficulty (40% win rate) |

---

## 🎯 Success Checkpoints

### ✅ Checkpoint 1: Grade 1 Advancement (~500K steps)
**Expected**: Agent kills 0.8+ spawners per episode

**If stuck**:
- Agent should be moving toward spawners
- Should be shooting projectiles
- Spawner health should be decreasing

**Fix if needed**: See troubleshooting section in CURRICULUM_UPDATE_SUMMARY.md

---

### ✅ Checkpoint 2: Grade 2 Advancement (~1M steps)
**Expected**: Agent kills 1.5+ spawners per episode

**If stuck**:
- Agent should be killing spawners efficiently
- Episode length should be reasonable (~1000-1500 steps)

---

### ✅ Checkpoint 3: First Win (~2-3M steps)
**Expected**: At least one complete victory (all 5 phases cleared)

**If not happening by 5M steps**:
- Check if agent reached Grade 4
- Check spawner kill rate (should be 2.0+)
- See troubleshooting guide for adjustments

---

## 🔧 Quick Adjustments

### If Progress Too Slow in Grade 1
Edit `arena/core/curriculum.py` line ~242:
```python
spawner_health_mult=0.3,  # Even easier (was 0.5)
```

### If Progress Too Slow in Grade 3 (Enemy Introduction)
Edit `arena/core/curriculum.py` line ~293:
```python
max_enemies_mult=0.2,  # Fewer enemies (was 0.4)
```

### If No Wins in Grade 4
Edit `arena/core/curriculum.py` line ~330:
```python
min_win_rate=0.02,  # Lower requirement (was 0.05)
```

Then restart training from the last checkpoint.

---

## 📁 Key Files

- **Training script**: `arena/train.py`
- **Curriculum config**: `arena/core/curriculum.py` (lines 235-366)
- **Reward/hyperparam config**: `arena/core/config.py`
- **Full documentation**: `CURRICULUM_UPDATE_SUMMARY.md`

---

## 🎮 Training Commands Cheat Sheet

### Basic Training
```bash
# Default (PPO, Style 1, 10M steps)
python -m arena.train --algo ppo --style 1 --steps 10000000

# Different algorithm
python -m arena.train --algo dqn --style 1 --steps 5000000

# More parallel environments (faster training if you have compute)
python -m arena.train --algo ppo --style 1 --steps 10000000 --num-envs 32
```

### Resume Training
```bash
# Find your checkpoint
ls -lt runs/ppo_style1_*/checkpoints/*.zip | head -1

# Resume from checkpoint
python -m arena.train \
  --algo ppo \
  --style 1 \
  --steps 10000000 \
  --load-model runs/ppo_style1_YYYYMMDD_HHMMSS/checkpoints/rl_model_XXXXX_steps.zip
```

### Evaluate After Training
```bash
# Run 100 episodes with visualization
python -m arena.evaluate \
  --model-path runs/ppo_style1_YYYYMMDD_HHMMSS/checkpoints/rl_model_XXXXX_steps.zip \
  --episodes 100
```

---

## 🐛 Common Issues

### Issue: "Curriculum not advancing"
**Symptom**: Stuck in Grade 1 for >1M steps

**Fix**: Agent may not be learning basics. Check:
1. Is agent moving? (velocity > 0)
2. Is agent shooting? (projectiles visible)
3. Is spawner taking damage?

If not, reduce spawner HP further (see Quick Adjustments above).

---

### Issue: "Training very slow"
**Symptom**: <1000 steps/second

**Fix**:
```bash
# Reduce parallel environments
python -m arena.train --algo ppo --style 1 --num-envs 8
```

---

### Issue: "Out of memory"
**Symptom**: Training crashes with memory error

**Fix**:
```bash
# Reduce batch size and parallel envs
python -m arena.train --algo ppo --style 1 --num-envs 8 --batch 32
```

---

## 📈 Interpreting Results

### Good Signs ✅
- Spawner kill rate increasing over time
- Episode reward trending upward
- Curriculum advancing every 50-200 episodes
- First wins appearing in Grade 4

### Warning Signs ⚠️
- Stuck in same grade for >500K steps
- Spawner kill rate not improving
- Episode reward flat or decreasing
- No wins by 5M steps

If you see warning signs, consult the troubleshooting section in `CURRICULUM_UPDATE_SUMMARY.md`.

---

## 🎓 After First Wins

Once you achieve 10%+ win rate:

1. **Continue training** to Grade 6 for optimal performance
2. **Optional**: Reduce Style 2 alignment reward (0.08 → 0.04) in `config.py`
3. **Optional**: Reduce PPO entropy (0.10 → 0.07) for more exploitation
4. **Celebrate!** 🎉 You've successfully trained an agent

---

## 💡 Pro Tips

1. **Use TensorBoard**: It's your best friend for monitoring progress
2. **Check logs regularly**: Look for curriculum advancement messages
3. **Be patient in Grade 1**: It may take 500K-1M steps to learn basics
4. **Don't restart too early**: Give it at least 3M steps before major changes
5. **Save checkpoints**: You can always resume if something goes wrong

---

## 🆘 Need Help?

1. **Read the full summary**: `CURRICULUM_UPDATE_SUMMARY.md`
2. **Check console logs**: Look for error messages
3. **Inspect TensorBoard**: Compare your curves to expected progression
4. **Adjust curriculum**: Use Quick Adjustments section above

---

## 📊 Target Performance (Full Training)

After 10M steps with this curriculum, you should see:

| Metric | Target |
|--------|--------|
| **Grade Reached** | 5 or 6 |
| **Win Rate** | 15-40% |
| **Avg Spawners Killed** | 3-5 per episode |
| **Avg Episode Reward** | 500-1500 |
| **Time to First Win** | 2-3M steps |

---

**Good luck!** 🚀

Remember: The key is **learning spawner destruction first (Grades 1-2), then adding enemy complexity (Grades 3-6)**.
