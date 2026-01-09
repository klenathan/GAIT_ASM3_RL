eval_headless:
	uv run python -m arena.eval_headless \
		--model runs/ppo/style2/ppo_style2_20260109_151548/final/ppo_style2_20260109_151548_final.zip \
		--episodes 100 \
		--style 2 \
		--algo ppo \
		--device cpu \
		--workers 10

eval_headless_multiple_s2:
	uv run python -m arena.eval_headless \
	--models \
	runs/dqn/style2/dqn_style2_20260109_145231/final/dqn_style2_20260109_145231_final.zip \
	runs/ppo/style2/ppo_style2_20260107_160605/final/ppo_style2_20260107_160605_final.zip \
	runs/ppo/style2/ppo_style2_20260109_151548/final/ppo_style2_20260109_151548_final.zip \
	runs/ppo_lstm/style2/ppo_lstm_style2_20260107_160654/final/ppo_lstm_style2_20260107_160654_final.zip \
	runs/ppo_lstm/style2/ppo_lstm_style2_20260109_151613/final/ppo_lstm_style2_20260109_151613_final.zip \
	--workers 10 \
	--episodes 100 \
	--csv comparison_s2.csv

eval_headless_multiple_s1:
	uv run python -m arena.eval_headless \
	--models \
	runs/ppo_lstm/style1/ppo_lstm_style1_20260109_183659/checkpoints/ppo_lstm_style1_20260109_183659_2000000_steps.zip \
	--workers 10 \
	--episodes 100 \
	--csv comparison_s1.csv

train_ppo_lstm_s1:
	uv run python -m arena.train --algo ppo_lstm --style 1 --steps 10000000 --no-render --device cpu

train_ppo_ppo_s1:
	uv run python -m arena.train --algo ppo --style 1 --steps 10000000 --no-render --device cpu