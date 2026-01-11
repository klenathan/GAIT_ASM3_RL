evaluate:
	uv run python -m arena.evaluate --device cpu

eval_headless:
	uv run python -m arena.eval_headless \
		--model runs/ppo_lstm/style1/ppo_lstm_style1_20260109_203229/checkpoints/ppo_lstm_style1_20260109_203229_7000000_steps.zip \
		--episodes 100 \
		--device cpu \
		--workers 10 \
		--stochastic

eval_headless_multiple_s2:
	uv run python -m arena.eval_headless \
	--models \
	runs/dqn/style2/dqn_style2_20260107_164443/final/dqn_style2_20260107_164443_final.zip \
	runs/dqn/style2/dqn_style2_20260109_145231/final/dqn_style2_20260109_145231_final.zip \
	runs/ppo/style2/ppo_style2_20260107_160605/final/ppo_style2_20260107_160605_final.zip \
	runs/ppo/style2/ppo_style2_20260109_151548/final/ppo_style2_20260109_151548_final.zip \
	runs/ppo_lstm/style2/ppo_lstm_style2_20260107_160654/final/ppo_lstm_style2_20260107_160654_final.zip \
	runs/ppo_lstm/style2/ppo_lstm_style2_20260109_151613/final/ppo_lstm_style2_20260109_151613_final.zip \
	--workers 10 \
	--episodes 100 \
	--stochastic \
	--csv comparison_s2.csv 

eval_headless_multiple_s1:
	uv run python -m arena.eval_headless \
	--models \
	runs/ppo_lstm/style1/ppo_lstm_style1_20260109_203229/checkpoints/ppo_lstm_style1_20260109_203229_7000000_steps.zip \
	runs/dqn/style1/dqn_style1_20260109_190750/final/dqn_style1_20260109_190750_final.zip \
	runs/ppo/style1/ppo_style1_20260109_204327/checkpoints/ppo_style1_20260109_204327_1600000_steps.zip \
	--workers 10 \
	--stochastic \
	--csv comparison_s1.csv \
	--episodes 100

train_ppo_lstm_s1:
	uv run python -m arena.train --algo ppo_lstm --style 1 --steps 10000000 --no-render --device cpu

train_ppo_ppo_s1:
	uv run python -m arena.train --algo ppo --style 1 --steps 15000000 --no-render --device cpu
