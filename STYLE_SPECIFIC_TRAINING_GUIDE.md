# Style-Specific Training Quick Reference

## Overview
The arena now has **separate configurations, curriculum, and reward shaping** for each control style.

---

## Control Styles

### Style 1: Rotation + Thrust + Shoot
- **Mechanics**: Physics-based movement with momentum and rotation
- **Complexity**: High (rotation, inertia, friction)
- **Actions**: 5 (None, Thrust, Rotate Left, Rotate Right, Shoot)
- **Best for**: Testing complex continuous-like control in discrete action space

### Style 2: Directional + Fixed Angle Shoot  
- **Mechanics**: Direct 4-way movement with fixed shooting angle
- **Complexity**: Lower (no rotation, no momentum)
- **Actions**: 6 (None, Up, Down, Left, Right, Shoot)
- **Best for**: Simpler positional play and kiting tactics

---

## Quick Training Commands

```bash
# Train Style 1 (rotation/thrust)
python arena/train.py --algo ppo --style 1 --steps 1000000

# Train Style 2 (directional)
python arena/train.py --algo ppo --style 2 --steps 1000000

# Resume training from checkpoint
python arena/train.py --algo ppo --style 1 --load-model runs/ppo/style1/.../checkpoint.zip

# Train with custom settings
python arena/train.py --algo ppo --style 1 --steps 2000000 --num-envs 24
```

---

## What's Different Per Style?

### Configuration (`arena/core/config.py`)

| Parameter | Style 1 | Style 2 | Why Different? |
|-----------|---------|---------|----------------|
| `shaping_scale` | 1.2 | 0.8 | Style 1 needs more reward guidance |
| `max_steps` | 3500 | 3000 | Style 1 needs more time to rotate |
| `penalty_inactivity` | -0.05 | 0.0 | Style 1 penalizes passive rotation |

### Curriculum Stages (`arena/core/curriculum.py`)

**Style 1** (4 stages):
1. Grade 3: Spawner Targeting (learn rotation + aiming)
2. Grade 4: Multi-Target Management (momentum control)
3. Grade 5: Aggressive Combat (efficient rotation)
4. Grade 6: Elite Performance (perfect execution)

**Style 2** (5 stages):
1. Grade 1: Positioning Basics (exploit fixed angle)
2. Grade 2: Kiting and Spacing (movement tactics)
3. Grade 3: Map Control (space control)
4. Grade 4: Advanced Tactics (near-full difficulty)
5. Grade 5: Perfect Execution (full difficulty)

**Note**: Style 2 has more stages but progresses faster since movement is simpler.

### Reward Shaping

- **Style 1**: Higher shaping scale (1.2) for complex rotation mechanics
- **Style 2**: Lower shaping scale (0.8) for simpler direct movement

---

## Expected Training Times

### Style 1 (Rotation + Thrust)
- **Early progress**: 200K - 500K steps (basic rotation + shooting)
- **Spawner kills**: 500K - 1M steps (consistent stage completion)
- **Elite play**: 2M - 5M steps (high win rate, fast clears)
- **Difficulty**: Harder to train due to rotation complexity

### Style 2 (Directional)
- **Early progress**: 100K - 300K steps (basic positioning)
- **Spawner kills**: 300K - 700K steps (consistent stage completion)
- **Elite play**: 1M - 3M steps (high win rate, fast clears)
- **Difficulty**: Easier to train due to simpler mechanics

---

## Customizing Style-Specific Parameters

### Option 1: Edit Config Classes

Edit `arena/core/config.py`:
```python
@dataclass
class Style1Config(ControlStyleConfig):
    shaping_scale: float = 1.5        # Increase guidance
    max_steps: int = 4000             # Allow more time
    reward_enemy_destroyed: float = 7.0  # Higher enemy kill reward
```

### Option 2: Edit Curriculum Stages

Edit `arena/core/curriculum.py`:
```python
def get_style1_stages() -> List[CurriculumStage]:
    return [
        CurriculumStage(
            name="Custom Stage",
            spawn_cooldown_mult=2.5,      # Slower spawns
            enemy_speed_mult=0.7,         # Slower enemies
            shaping_scale_mult=3.0,       # More guidance
            min_spawner_kill_rate=0.5,    # Easier advancement
            min_episodes=50,
        ),
        # ... more stages
    ]
```

---

## Monitoring Training

### TensorBoard
```bash
# View training progress
tensorboard --logdir runs/ppo/style1/
tensorboard --logdir runs/ppo/style2/

# Compare both styles
tensorboard --logdir runs/
```

### Key Metrics to Watch

**For Both Styles:**
- `rollout/ep_rew_mean` - Average episode reward
- `rollout/ep_len_mean` - Average episode length
- `train/learning_rate` - Learning rate schedule
- `curriculum/stage_index` - Current curriculum stage

**Style 1 Specific:**
- Watch for longer episode lengths initially (rotation learning)
- Expect slower initial progress
- Look for smooth reward curves once rotation is learned

**Style 2 Specific:**
- Expect faster initial progress
- Watch for plateau around positioning mastery
- Look for rapid curriculum advancement

---

## Evaluation

```bash
# Run interactive evaluation
python arena/evaluate.py

# Headless evaluation (batch testing)
python arena/eval_headless.py --model path/to/model.zip --episodes 100
```

The evaluator automatically:
- Detects the model's control style
- Loads appropriate style-specific config
- Uses correct curriculum settings

---

## Troubleshooting

### Style 1 Not Learning Rotation
- Increase `shaping_scale` to 1.5 or higher
- Increase `max_steps` to 4000+
- Start with easier curriculum stage (higher `spawn_cooldown_mult`)

### Style 2 Too Passive
- Increase `penalty_corner` (e.g., -0.2)
- Decrease `spawn_cooldown_mult` for more pressure
- Adjust curriculum `max_survival_steps` to encourage faster clears

### Both Styles Struggling
- Check curriculum advancement criteria (might be too strict)
- Verify reward balance (spawner vs enemy rewards)
- Consider disabling curriculum temporarily (`CURRICULUM_ENABLED = False`)

---

## Best Practices

1. **Start with default configs** - They're tuned for typical training
2. **Monitor curriculum progression** - Should advance every 100-200 episodes
3. **Compare styles separately** - They have different learning curves
4. **Save checkpoints frequently** - Use `--checkpoint-freq 50000`
5. **Test on multiple seeds** - Verify consistency across runs

---

## Architecture Recommendations

### Style 1 (Rotation + Thrust)
- **Policy network**: [256, 128, 64] or deeper [384, 256, 128]
- **Learning rate**: 3e-4 (standard) or 5e-4 (faster)
- **Entropy coefficient**: 0.05 (encourage exploration)

### Style 2 (Directional)
- **Policy network**: [256, 128, 64] (standard works well)
- **Learning rate**: 5e-4 (can train faster)
- **Entropy coefficient**: 0.03 (less exploration needed)

---

## Advanced: Creating Custom Styles

To add a new control style (e.g., Style 3: Mouse-aim):

1. **Create config class**:
```python
# In arena/core/config.py
@dataclass
class Style3Config(ControlStyleConfig):
    shaping_scale: float = 1.0
    max_steps: int = 3000
    # ... custom parameters
```

2. **Create curriculum**:
```python
# In arena/core/curriculum.py
def get_style3_stages() -> List[CurriculumStage]:
    return [
        # ... custom stages
    ]
```

3. **Update factory functions**:
```python
def get_style_config(style: int):
    if style == 3:
        return Style3Config()
    # ...

def get_default_stages(control_style: int):
    if control_style == 3:
        return get_style3_stages()
    # ...
```

4. **Add entity movement logic**:
```python
# In arena/game/entities.py (Player class)
def update_style_3(self, action):
    # Custom movement logic
    pass
```

---

## Summary

- ✅ Each style has optimized config, curriculum, and rewards
- ✅ Training automatically uses correct style-specific settings
- ✅ Easy to customize per-style parameters
- ✅ Compatible with existing models and workflows
- ✅ Extensible for future control schemes

Happy training! 🚀
