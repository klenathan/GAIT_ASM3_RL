# TensorBoard Style Configuration Logging

## Overview

The training system now automatically logs **control style configuration, reward structure, and curriculum settings** to TensorBoard as formatted text. This makes it easy to:
- Identify which control style a training run uses
- Compare reward structures across different training runs
- Verify curriculum configuration
- Understand style-specific design decisions

---

## What's Logged

### 1. Control Style Information
- **Control Style Number**: 1 or 2
- **Style Name**: 
  - Style 1: "Rotation + Thrust + Shoot"
  - Style 2: "Directional (4-way) + Fixed Angle Shoot"
- **Complexity Level**: High vs Medium

### 2. Reward Structure
Complete reward configuration including:
- Enemy Destroyed
- Spawner Destroyed
- Quick Spawner Kill
- Hit Enemy/Spawner
- Damage Taken
- Death
- Step Survival
- Shot Fired
- Phase Complete

### 3. Episode Configuration
- Max Steps
- Shaping Mode
- Shaping Scale
- Shaping Clip

### 4. Activity Penalties
- Inactivity Penalty
- Corner Penalty
- Corner Margin
- Inactivity Velocity Threshold

### 5. Curriculum Configuration
- Whether curriculum is enabled
- Total number of stages
- Starting stage
- **Detailed stage breakdown** for each curriculum stage:
  - Spawn Cooldown Multiplier
  - Max Enemies Multiplier
  - Spawner Health Multiplier
  - Enemy Speed Multiplier
  - Shaping Scale Multiplier
  - Damage Penalty Multiplier
  - Advancement criteria (min spawner kill rate, min win rate, min episodes)

### 6. Design Notes
Style-specific explanations of configuration choices:
- Why certain parameters are set differently
- Focus areas for the curriculum
- Key learning objectives

---

## How to View in TensorBoard

### Step 1: Start Training
```bash
# Train with any style
python arena/train.py --algo ppo --style 1 --steps 100000
```

### Step 2: Launch TensorBoard
```bash
tensorboard --logdir runs/
```

### Step 3: View Configuration
1. Open TensorBoard in your browser (usually `http://localhost:6006`)
2. Click on the **TEXT** tab
3. Look for **"configuration/style_and_rewards"**
4. You'll see a formatted document with all configuration details

---

## Example Output

### Style 1 Configuration in TensorBoard

```markdown
# Control Style Configuration

**Control Style:** 1 - Rotation + Thrust + Shoot
**Style Complexity:** High (rotation, momentum, inertia)

---

## Reward Structure
| Reward Type | Value |
|-------------|-------|
| Enemy Destroyed | 5.00 |
| Spawner Destroyed | 75.00 |
| Quick Spawner Kill | 50.00 |
| Hit Enemy | 2.00 |
| Hit Spawner | 2.00 |
| Damage Taken | -2.00 |
| Death | -100.00 |
| Step Survival | -0.0100 |
| Shot Fired | 0.00 |
| Phase Complete | 0.00 |

## Episode Configuration
| Parameter | Value |
|-----------|-------|
| Max Steps | 3500 |
| Shaping Mode | delta |
| Shaping Scale | 1.20 |
| Shaping Clip | 0.20 |

## Activity Penalties
| Penalty Type | Value |
|--------------|-------|
| Inactivity Penalty | -0.0500 |
| Corner Penalty | -0.1000 |
| Corner Margin | 80 |
| Inactivity Velocity Threshold | 0.50 |

## Curriculum Configuration
| Parameter | Value |
|-----------|-------|
| Curriculum Enabled | Yes |
| Total Stages | 4 |
| Starting Stage | 0 - Grade 3: Spawner Targeting |

### Curriculum Stages

**Stage 0: Grade 3: Spawner Targeting**
| Modifier | Value |
|----------|-------|
| Spawn Cooldown Mult | 1.80 |
| Max Enemies Mult | 0.60 |
| Spawner Health Mult | 0.65 |
| Enemy Speed Mult | 0.90 |
| Shaping Scale Mult | 2.50 |
| Damage Penalty Mult | 0.90 |
| Min Spawner Kill Rate | 0.80 |
| Min Win Rate | 0.00 |
| Min Episodes | 100 |

... (additional stages)

---

## Style 1 Design Notes
- **Higher shaping scale** (1.20) provides more reward guidance for complex rotation mechanics
- **Longer episodes** (3500) to account for rotation learning curve
- **Inactivity penalty** (-0.0500) encourages active rotation and movement
- **Curriculum focused on:** rotation mastery, momentum management, aiming while moving
```

---

## Comparing Training Runs

### Use Case: Compare Style 1 vs Style 2

1. **Train both styles:**
```bash
python arena/train.py --algo ppo --style 1 --steps 500000
python arena/train.py --algo ppo --style 2 --steps 500000
```

2. **Launch TensorBoard with both runs:**
```bash
tensorboard --logdir runs/
```

3. **In TensorBoard:**
   - Switch between runs in the dropdown
   - Check TEXT tab → "configuration/style_and_rewards"
   - Compare:
     - Reward structures
     - Max steps
     - Shaping scales
     - Curriculum stages
     - Design rationale

4. **Cross-reference with SCALARS tab:**
   - Look at `arena/ep_rew_mean` to see reward differences
   - Compare `curriculum/stage` progression
   - Analyze `arena/win_rate_100ep` for learning efficiency

---

## Benefits

### 1. **Reproducibility**
Every training run documents its exact configuration in TensorBoard, making it easy to:
- Reproduce successful runs
- Understand why certain runs performed differently
- Share configurations with team members

### 2. **Debugging**
Quickly verify:
- Whether the correct style configuration was loaded
- If curriculum is enabled and configured properly
- That reward values match expectations
- Why learning might be slower/faster than expected

### 3. **Experiment Tracking**
When tuning hyperparameters:
- Compare different reward balances
- Test curriculum variations
- Identify which settings work best for each style

### 4. **Documentation**
- Configurations are automatically documented
- No need to manually track which settings were used
- Design rationale is included for future reference

---

## Technical Details

### Implementation

The logging is handled by the `StyleConfigCallback` in `arena/training/callbacks.py`:

```python
callback = StyleConfigCallback(
    control_style=1,  # or 2
    style_config=get_style_config(1),
    curriculum_manager=curriculum_manager
)
```

### When Logging Occurs

- Logs are written **once at the start of training** (timestep 0)
- Appears in TensorBoard's TEXT tab under "configuration/style_and_rewards"
- Includes both hyperparameters tab (table format) and style configuration (markdown format)

### Customization

To add more information to the logs, edit `StyleConfigCallback._on_training_start()` in `arena/training/callbacks.py`.

---

## Troubleshooting

### Issue: Configuration not showing in TensorBoard

**Solution:**
1. Make sure TensorBoard is pointing to the correct log directory
2. Check that training has started (configuration logs at timestep 0)
3. Refresh TensorBoard browser tab
4. Verify the TEXT tab is selected

### Issue: Curriculum stages not showing

**Solution:**
- Check if `CURRICULUM_ENABLED = True` in config
- Verify curriculum_manager is passed to the callback
- Ensure curriculum is initialized before training starts

### Issue: Wrong style configuration shown

**Solution:**
- Verify `--style` argument matches intended control style
- Check that `get_style_config(style)` is called correctly
- Restart training run to regenerate logs

---

## Example Use Cases

### Use Case 1: Verify Training Setup
Before a long training run, quickly verify configuration:
1. Start training for a few seconds
2. Open TensorBoard
3. Check TEXT tab to confirm:
   - Correct style is loaded
   - Reward values are as expected
   - Curriculum is configured properly
4. Stop training if anything looks wrong

### Use Case 2: Compare Reward Tuning
When experimenting with reward values:
1. Create multiple runs with different reward configs
2. View each in TensorBoard TEXT tab
3. Compare learning curves in SCALARS tab
4. Identify which reward structure works best

### Use Case 3: Debug Curriculum Issues
If agent isn't advancing through curriculum:
1. Check TEXT tab for curriculum stage requirements
2. Verify advancement criteria are achievable
3. Compare with actual performance in SCALARS tab
4. Adjust criteria if needed and retrain

---

## Summary

The automatic TensorBoard logging provides:
- ✅ **Complete visibility** into training configuration
- ✅ **Easy comparison** between different runs
- ✅ **Better debugging** when issues arise
- ✅ **Automatic documentation** of experimental settings
- ✅ **Style-specific context** for understanding design decisions

Every training run now includes comprehensive configuration documentation in TensorBoard! 🎯
