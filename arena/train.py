"""
Main entry point for training RL agents in the Arena environment.
Uses the modular TrainerRunner and AlgorithmRegistry.
"""

import argparse
from arena.core.config import TrainerConfig
from arena.training.runner import TrainerRunner

def parse_args():
    parser = argparse.ArgumentParser(description="Arena RL Trainer")
    parser.add_argument("--algo", type=str, default="ppo", choices=["dqn", "ppo", "ppo_lstm"])
    parser.add_argument("--style", type=int, default=1, choices=[1, 2])
    parser.add_argument("--steps", type=int, help="Override total timesteps")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--batch", type=int, help="Override batch size")
    parser.add_argument("--tensorboard", action="store_true", default=True)
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create config from args
    config = TrainerConfig(
        algo=args.algo,
        control_style=args.style,
        device=args.device,
        render=not args.no_render
    )
    
    # Overrides
    if args.steps: config.total_timesteps = args.steps
    if args.lr: config.learning_rate = args.lr
    if args.batch: config.batch_size = args.batch
    
    print(f"Starting training: {config.algo} (Style {config.control_style}) on {config.device}")
    
    runner = TrainerRunner(config)
    runner.run()

if __name__ == "__main__":
    main()
