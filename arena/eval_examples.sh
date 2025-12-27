#!/bin/bash
# Example usage scripts for headless evaluation

# Example 1: Quick evaluation of a single model (100 episodes)
echo "Example 1: Quick evaluation"
python -m arena.eval_headless \
    --model runs/ppo/style1/latest_model.zip \
    --episodes 100

# Example 2: Deep evaluation with 1000 episodes for more reliable statistics
echo "Example 2: Deep evaluation (1000 episodes)"
python -m arena.eval_headless \
    --model runs/ppo/style1/latest_model.zip \
    --episodes 1000 \
    --output results_1000eps.json

# Example 3: Compare multiple models
echo "Example 3: Compare multiple models"
python -m arena.eval_headless \
    --models \
        runs/ppo/style1/checkpoint1.zip \
        runs/ppo/style1/checkpoint2.zip \
        runs/ppo/style1/final.zip \
    --episodes 500 \
    --compare \
    --output comparison.json

# Example 4: Evaluate all models in a directory
echo "Example 4: Evaluate all models in directory"
python -m arena.eval_headless \
    --directory runs/ppo/style1/ \
    --episodes 100 \
    --compare

# Example 5: Stochastic evaluation (for models with exploration)
echo "Example 5: Stochastic evaluation"
python -m arena.eval_headless \
    --model runs/ppo/style1/latest_model.zip \
    --episodes 1000 \
    --stochastic \
    --output stochastic_results.json

# Example 6: Evaluate on different control style
echo "Example 6: Different control style"
python -m arena.eval_headless \
    --model runs/ppo/style2/model.zip \
    --style 2 \
    --episodes 500

# Example 7: Quiet mode with JSON output only
echo "Example 7: Quiet mode"
python -m arena.eval_headless \
    --model runs/ppo/style1/latest_model.zip \
    --episodes 1000 \
    --quiet \
    --output results.json

# Example 8: Save detailed episode data for analysis
echo "Example 8: Save detailed episode data"
python -m arena.eval_headless \
    --model runs/ppo/style1/latest_model.zip \
    --episodes 500 \
    --save-episodes \
    --output detailed_results.json

# Example 9: Force CPU evaluation
echo "Example 9: CPU evaluation"
python -m arena.eval_headless \
    --model runs/ppo/style1/latest_model.zip \
    --episodes 100 \
    --device cpu

# Example 10: Massive evaluation for publication-quality statistics
echo "Example 10: Massive evaluation"
python -m arena.eval_headless \
    --model runs/ppo/style1/final_model.zip \
    --episodes 10000 \
    --output publication_results.json


