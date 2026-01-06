# Curriculum Redesign: 50-Stage Ultra-Granular Learning

## Summary

This update completely redesigns the curriculum system to fix two critical issues:
1. **Anti-camping exploit** in reward shaping
2. **Stuck progression** at early stages

## Key Changes

### 1. Anti-Camping Reward Fix (`arena/core/environment.py`)

**Problem**: Agent exploited proximity reward by camping near spawners without shooting them.

**Solution**: Modified `_calculate_proximity_reward()` to:
- Add penalty for being too close (<150 pixels) without dealing damage
- Reduce rewards for being too far (>600 pixels) 
- Optimal engagement range: 150-600 pixels
- Camping penalty: -0.01 scaled by proximity

```python
if current_dist < optimal_distance_min:
    current_spawner_health = sum(s.health for s in self.spawners if s.alive)
    spawner_damage_this_step = max(0, self._prev_spawner_total_health - current_spawner_health) if self._prev_spawner_total_health else 0
    
    if spawner_damage_this_step == 0:  # Camping without dealing damage
        camping_penalty = -0.01 * (1.0 - current_dist / optimal_distance_min)
        proximity_reward += camping_penalty
```

### 2. 50-Stage Granular Curriculum (`arena/core/curriculum.py`)

**Old System**: 8 stages, too aggressive jumps, stuck at stage 0
**New System**: 50 stages, ultra-fine progression, ~8.59M timesteps total

#### Stage Breakdown

**Part 1: Single Spawner Mastery (Stages 0-19)**
- **Stages 0-4**: Ultra-basic positioning (8-16 HP spawners, 25x→21x guidance)
  - Fixed center position
  - Learn: Move → Aim → Shoot basics
  - 100 episodes/stage, ~100K timesteps/stage

- **Stages 5-9**: Position transfer (15-27 HP spawners, 20x→12x guidance)
  - Fixed positions: 1 → 2 → 3 → 4 corners → Random
  - Learn: Generalize positioning to different locations
  - 120 episodes/stage, ~130K timesteps/stage

- **Stages 10-14**: Tougher spawners (30-70 HP, 12x→6x guidance)
  - Random spawner positions
  - Learn: Sustained shooting, tracking
  - 120 episodes/stage, ~150K timesteps/stage

- **Stages 15-19**: Efficiency mastery (75-95 HP, 6x→2x guidance)
  - Near full-health spawners
  - Learn: Fast, accurate kills
  - 120 episodes/stage, ~160K timesteps/stage

**Part 2: Multiple Spawners (Stages 20-39)**
- **Stages 20-24**: Two spawners intro (80-96 HP, still no enemies)
  - Learn: Phase transitions, target prioritization
  - 120 episodes/stage, ~210K timesteps/stage

- **Stages 25-29**: Multi-spawner progress (70-90 HP)
  - Learn: Efficient clearing, movement patterns
  - 120 episodes/stage, ~205K timesteps/stage

- **Stages 30-34**: Full HP multiple spawners (90-98 HP)
  - Learn: Consistent multi-target elimination
  - 120 episodes/stage, ~200K timesteps/stage

- **Stages 35-39**: Efficient multi-clear (100 HP)
  - Learn: Speed optimization, 1.7-2.1 spawners/episode
  - 120 episodes/stage, ~195K timesteps/stage

**Part 3: Enemies + Spawners (Stages 40-49)**
- **Stages 40-42**: Slow enemies (15-35%, 40-60% speed)
  - Learn: Dodging while maintaining offensive focus
  - 120 episodes/stage, ~185K timesteps/stage

- **Stages 43-45**: Moderate enemies (40-70%, 60-80% speed)
  - Learn: Combat multitasking, survival
  - 120 episodes/stage, ~195K timesteps/stage

- **Stages 46-49**: Full difficulty mastery (75-99%, 85-100% speed)
  - Learn: Elite performance, 20% win rate target
  - 120 episodes/stage, ~185K timesteps/stage

### 3. Training Speed Optimizations

**Config Changes** (`arena/core/config.py`):
- Reduced `MAX_STEPS` from 3000 → 2500
- Optimized PPO hyperparameters:
  - `n_steps`: 4096 → 2048 (faster updates)
  - `batch_size`: 64 → 128 (better gradients)
  - `ent_coef`: 0.10 → 0.08 (faster convergence)

**Curriculum Optimizations**:
- Reduced `min_episodes` from 150 → 120 for stages 20-49
- Shorter episode lengths via `max_episode_steps`
- Tighter advancement criteria for faster progression

## Training Time Estimate

```
Part 1 (S0-19):   ~2.94M timesteps
Part 2 (S20-39):  ~4.20M timesteps
Part 3 (S40-49):  ~1.91M timesteps
──────────────────────────────────
Total:            ~8.59M timesteps
```

With 20 parallel environments: **~430K wall-clock timesteps**

**Expected training time** (at 2000 FPS): ~3.5 hours for full curriculum

## How to Train

```bash
# Style 1 (Rotation + Thrust)
python arena/train.py --algo ppo --style 1 --steps 10000000 --num-envs 20

# Style 2 (Fixed Nozzle)
python arena/train.py --algo ppo --style 2 --steps 10000000 --num-envs 20
```

## Testing

Run the test script to verify curriculum structure:

```bash
python test_new_curriculum.py
```

Expected output:
- ✓ 50 stages loaded correctly
- ✓ Part 1: Single spawner mastery (no enemies)
- ✓ Part 2: Multiple spawners (no enemies)
- ✓ Part 3: Full difficulty with enemies
- ✓ Training estimate: ~8.59M timesteps

## Key Design Principles

1. **Ultra-small steps**: Each stage only increases difficulty by 5-10%
2. **Fast advancement**: 100-120 episodes per stage (vs 200-300 before)
3. **No enemy clutter early**: Focus on positioning/shooting first (S0-39)
4. **Smooth transitions**: Health increases gradually, no sudden jumps
5. **Clear objectives**: Each stage has specific, achievable goals

## Expected Outcomes

1. **No more camping**: Anti-camping penalty prevents exploit
2. **Consistent progression**: Agent should advance every 150-250K timesteps
3. **Learn positioning first**: Master single-target before multi-target
4. **Fast training**: Complete under 10M timesteps (<4 hours wall-clock)
5. **Better generalization**: Gradual difficulty → robust policies

## Monitoring Progress

Watch TensorBoard for:
- `curriculum/stage`: Should advance steadily (0 → 49)
- `arena/ep_spawners_mean`: Should increase (0.5 → 2.5)
- `arena/win_rate_100ep`: Should reach 20%+ by stage 49
- `arena/ep_len_mean`: Should decrease over time (faster wins)

```bash
tensorboard --logdir runs/
```

## Troubleshooting

**If stuck at a stage**:
1. Check `arena/ep_spawners_mean` - should be above threshold
2. Verify agent is dealing damage (not camping)
3. Consider reducing `min_episodes` for that stage
4. Increase `shaping_scale_mult` for more guidance

**If advancing too fast**:
1. Agent might be lucky - verify over 100+ episodes
2. Increase `min_episodes` requirement
3. Tighten advancement criteria (higher kill rate)

## Comparison: Old vs New

| Metric | Old (8 stages) | New (50 stages) |
|--------|---------------|-----------------|
| Total stages | 8 | 50 |
| Timesteps/stage | ~500K-1M | ~150-250K |
| Health progression | 12 HP → 100 HP (8x jump) | 8 HP → 100 HP (50 gradual steps) |
| Enemy introduction | Stage 5 | Stage 40 |
| Min episodes/stage | 200-300 | 100-120 |
| Estimated total time | 10M+ (stuck) | 8.59M (smooth) |
| Guidance reduction | Abrupt | Gradual (25x → 1x) |

## Files Modified

1. `arena/core/environment.py`
   - Modified `_calculate_proximity_reward()` (anti-camping)

2. `arena/core/curriculum.py`
   - Rewrote `get_default_stages()` (50 stages)

3. `arena/core/config.py`
   - Reduced `MAX_STEPS` to 2500
   - Optimized `PPOHyperparams`

4. `test_new_curriculum.py` (new)
   - Comprehensive test suite for curriculum

## Next Steps

1. **Start training**:
   ```bash
   python arena/train.py --algo ppo --style 2 --steps 10000000
   ```

2. **Monitor progress**:
   ```bash
   tensorboard --logdir runs/
   ```

3. **Evaluate performance**:
   ```bash
   python arena/evaluate.py --model runs/[your-run]/checkpoints/rl_model_8000000_steps.zip
   ```

4. **Fine-tune if needed**:
   - Adjust individual stage thresholds
   - Modify reward scaling
   - Tune hyperparameters

---

**Author**: OpenCode AI  
**Date**: 2026-01-06  
**Version**: 1.0  
**Tested**: ✓ Curriculum loads correctly, ~8.59M timestep estimate
