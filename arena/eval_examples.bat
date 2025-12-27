@echo off
REM Example usage scripts for headless evaluation (Windows)

echo.
echo Example 1: Quick evaluation of a single model (100 episodes)
echo ============================================================
uv run python -m arena.eval_headless --model runs\ppo\style1\ppo_style1_20251225_175203\final\ppo_style1_20251225_175203_final.zip --episodes 100
pause

echo.
echo Example 2: Deep evaluation with 1000 episodes
echo ============================================
uv run python -m arena.eval_headless --model runs\ppo\style1\ppo_style1_20251225_175203\final\ppo_style1_20251225_175203_final.zip --episodes 1000 --output results_1000eps.json
pause

echo.
echo Example 3: Compare multiple checkpoints
echo =======================================
uv run python -m arena.eval_headless --models runs\ppo\style1\ppo_style1_20251224_175544\checkpoints\ppo_style1_20251224_175544_600000_steps.zip runs\ppo\style1\ppo_style1_20251224_175544\checkpoints\ppo_style1_20251224_175544_1200000_steps.zip --episodes 500 --compare
pause

echo.
echo Example 4: Evaluate all models in a directory
echo =============================================
uv run python -m arena.eval_headless --directory runs\ppo\style1 --episodes 100 --compare
pause

echo.
echo Example 5: Stochastic evaluation
echo ================================
uv run python -m arena.eval_headless --model runs\ppo\style1\ppo_style1_20251225_175203\final\ppo_style1_20251225_175203_final.zip --episodes 1000 --stochastic --output stochastic_results.json
pause

echo.
echo Example 6: Quiet mode with JSON output
echo ======================================
uv run python -m arena.eval_headless --model runs\ppo\style1\ppo_style1_20251225_175203\final\ppo_style1_20251225_175203_final.zip --episodes 1000 --quiet --output results.json
pause

echo.
echo Example 7: Massive evaluation (10000 episodes)
echo ==============================================
uv run python -m arena.eval_headless --model runs\ppo\style1\ppo_style1_20251225_175203\final\ppo_style1_20251225_175203_final.zip --episodes 10000 --output publication_results.json
pause
