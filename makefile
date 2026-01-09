eval_headless:
	uv run python -m arena.eval_headless \
		--model runs/dqn/style2/dqn_style2_20260109_145231/final/dqn_style2_20260109_145231_final.zip \
		--episodes 100 \
		--style 2 \
		--algo dqn \
		--device cpu \
		--workers 10