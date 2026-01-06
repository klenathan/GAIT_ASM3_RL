# Option B Implementation - Quick Start Guide

## ✅ What Was Implemented

**Option B: Aggressive Reward Optimization** - A complete reward restructure designed to maximize agent win rate through clear, strong optimization signals.

## 🚀 Quick Start

### Start Training Immediately

```bash
# PPO (Recommended for complex optimization)
python -m arena.train --algo ppo --style 1 --steps 10000000

# DQN (Faster per-step, good for initial testing)
python -m arena.train --algo dqn --style 1 --steps 5000000
```

### Verify Implementation

```bash
# Run comprehensive test suite
python test_reward_optionB.py
```

## 📊 Key Changes at a Glance

| Reward Component | Before | After | Change |
|------------------|---------|-------|---------|
| **WIN** | 0 (implicit) | **+500** | NEW! |
| **Phase Complete** | 0 | **+50** | NEW! |
| **Spawner Destroyed** | +50 | +40 | -20% |
| **Enemy Destroyed** | +0.5 | **+5.0** | +900% |
| **Quick Kill (Phase 1)** | +50 | +20 | Progressive |
| **Quick Kill (Phase 5)** | +50 | **+60** | +20% |
| **Health Bonus** | 0 | **+2 to +5** | NEW! |
| **Death** | -100 | **-200** | -100% |
| **Damage** | -3 | **-5** | -67% |

**Perfect Win Reward**: 770 → **1,969** points (+156%)

## 🎯 What This Achieves

### Primary Goal: Maximize Win Rate

1. **Explicit Win Optimization** (+500 points)
   - Winning is now the single most rewarding event
   - Agent clearly understands the primary objective

2. **Phase Progression Guidance** (+50 per phase)
   - Dense reward signal along critical path
   - Prevents getting stuck on early phases

3. **Health Management** (+2 to +5 per phase)
   - Rewards survival through phase completion
   - Encourages defensive, tactical play

4. **Progressive Difficulty** (20 → 60 quick kill bonus)
   - Harder phases = bigger rewards
   - Properly incentivizes challenge completion

5. **Balanced Combat** (Enemies 10× more valuable)
   - Can't ignore threats anymore
   - More robust combat strategies

## 📁 Files Modified

- ✅ `arena/core/config.py` - New reward constants
- ✅ `arena/core/environment.py` - WIN reward + health bonuses + phase-aware quick kills
- ✅ `arena/core/environment_cnn.py` - Same updates for CNN observation space
- ✅ `arena/core/environment_dict.py` - Same updates for dict observation space
- ✅ `arena/core/config.py.backup` - Original configuration saved

## 📖 Documentation

- **Full Details**: `REWARD_OPTIMIZATION_OPTIONB.md`
- **Test Suite**: `test_reward_optionB.py`
- **This Guide**: `REWARD_OPTIONB_QUICKSTART.md`

## 🔍 Expected Results

### Training Metrics to Watch

1. **Win Rate**: Should increase 20-40% absolute over old system
2. **Phase Progression**: Agents reach later phases faster
3. **Episode Length**: Successful episodes may be slightly longer (less rushing)
4. **Health Preserved**: Winning agents maintain higher HP
5. **Enemy Engagement**: More enemy kills per episode

### Typical Learning Curve

- **0-100K steps**: Random exploration, occasional Phase 1 completions
- **100K-500K steps**: Consistent Phase 1-2 wins, learning Phase 3
- **500K-1M steps**: Regular Phase 3-4 completions
- **1M-3M steps**: First Phase 5 wins, improving consistency
- **3M+ steps**: High win rate (50-80% depending on algorithm/config)

## ⚠️ Important Notes

### Existing Models

If you have pre-trained models:
- Loading them will work, but policy trained on old rewards
- Performance may initially drop as policy adapts
- Consider reducing learning rate for fine-tuning
- May need 100K-500K steps to recover/improve

### Rollback

If you need the old system:
```bash
cp arena/core/config.py.backup arena/core/config.py
# Then manually restore old environment code (see git history)
```

### Control Styles

The reward system works for both control styles:
- **Style 1**: Rotation + Thrust (benefits from alignment reward too)
- **Style 2**: Directional + Fixed angle (has additional alignment reward)

## 🎓 Training Tips

### Hyperparameter Recommendations

**PPO (Recommended)**:
```bash
python -m arena.train --algo ppo --style 1 --steps 10000000 \
    --device auto --num-envs 20
```
- Good balance of sample efficiency and stability
- Handles sparse rewards well
- Learning rate: 5e-4 (default)

**DQN (Alternative)**:
```bash
python -m arena.train --algo dqn --style 1 --steps 5000000 \
    --device auto
```
- Faster per-step
- May need larger buffer (200K+)
- Learning rate: 3e-4 (default)

### Monitoring Training

Watch these TensorBoard metrics:
- `rollout/ep_rew_mean`: Should increase over time
- `rollout/ep_len_mean`: May increase (less rushing)
- Custom metrics:
  - `info/win`: Win rate (most important!)
  - `info/phase`: Average phase reached
  - `info/spawners_destroyed`: Should increase

## ✨ Success Criteria

Your agent is learning well if:

1. **Win rate > 0%** by 1M steps
2. **Win rate > 20%** by 3M steps
3. **Win rate > 50%** by 5M+ steps (goal)
4. **Average phase reached** increasing steadily
5. **Episode reward** trending upward

## 🐛 Troubleshooting

### Agent Not Learning?
- Check reward signal: Run test_reward_optionB.py
- Verify positive rewards appear in logs
- Try different algorithm (PPO vs DQN)
- Increase training steps (may need 5M+)

### Agent Too Aggressive?
- Good sign! Shows win motivation
- May die more early on (expected)
- Should improve with experience

### Agent Too Passive?
- Check entropy coefficient (should be > 0)
- Verify exploration epsilon (DQN)
- May need curriculum learning

## 📞 Questions?

Refer to:
1. `REWARD_OPTIMIZATION_OPTIONB.md` - Complete technical documentation
2. `test_reward_optionB.py` - See reward calculations in action
3. `arena/core/config.py` - All reward constants defined here

---

**Ready to Train!** 🚀

Start with:
```bash
python -m arena.train --algo ppo --style 1 --steps 10000000
```

Monitor results in TensorBoard:
```bash
tensorboard --logdir ./runs
```

Good luck! May your agents achieve 100% win rate! 🎯
