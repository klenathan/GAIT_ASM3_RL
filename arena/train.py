"""
Main entry point for training RL agents in the Arena environment.
Uses the modular TrainerRunner and AlgorithmRegistry.
"""

import argparse
from arena.core.config import TrainerConfig
from arena.training.runner import TrainerRunner
from arena.training.registry import AlgorithmRegistry
# Import algorithms to register them
import arena.training.algorithms 

def parse_args():
    parser = argparse.ArgumentParser(description="Arena RL Trainer")
    available_algos = AlgorithmRegistry.list_algorithms()
    parser.add_argument("--algo", type=str, default="ppo", choices=available_algos)
    parser.add_argument("--style", type=int, default=1, choices=[1, 2])
    parser.add_argument("--steps", type=int, help="Override total timesteps")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--batch", type=int, help="Override batch size")
    parser.add_argument("--num-envs", type=int, help="Number of parallel environments")
    parser.add_argument("--tensorboard", action="store_true", default=True)
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument(
        "--load-model",
        type=str,
        default=None,
        help="Path to an SB3/SB3-Contrib .zip model to resume training from (same algo + style).",
    )
    parser.add_argument(
        "--reset-timesteps",
        action="store_true",
        help="If set with --load-model, reset timesteps to 0 for the new run (otherwise continues).",
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # By default, new training resets timesteps; resume training continues timesteps unless overridden.
    reset_num_timesteps = True if not args.load_model else args.reset_timesteps
    
    # Create config from args
    config = TrainerConfig(
        algo=args.algo,
        style=args.style,
        device=args.device,
        render=not args.no_render,
        num_envs=args.num_envs,
        pretrained_model_path=args.load_model,
        reset_num_timesteps=reset_num_timesteps,
    )
    
    # Overrides
    if args.steps: config.total_timesteps = args.steps
    if args.lr: config.learning_rate = args.lr
    if args.batch: config.batch_size = args.batch
    
    print(f"Starting training: {config.algo} (Style {config.style}) on {config.device}")
    
    runner = TrainerRunner(config)
    runner.run()

if __name__ == "__main__":
    main()
