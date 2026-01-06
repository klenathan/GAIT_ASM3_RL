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
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"]
    )
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
    # Imitation Learning arguments
    parser.add_argument(
        "--demo-path",
        type=str,
        default=None,
        help="Path to demonstration file (.pkl) for behavioral cloning pretraining.",
    )
    parser.add_argument(
        "--bc-pretrain",
        action="store_true",
        help="Enable behavioral cloning pretraining from demonstrations before RL.",
    )
    parser.add_argument(
        "--bc-epochs",
        type=int,
        default=10,
        help="Number of behavioral cloning pretraining epochs (default: 10).",
    )
    parser.add_argument(
        "--bc-lr",
        type=float,
        default=1e-3,
        help="Learning rate for behavioral cloning (default: 1e-3).",
    )
    parser.add_argument(
        "--bc-batch-size",
        type=int,
        default=64,
        help="Batch size for behavioral cloning (default: 64).",
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
        # Imitation Learning
        demo_path=args.demo_path,
        bc_pretrain=args.bc_pretrain,
        bc_epochs=args.bc_epochs,
        bc_learning_rate=args.bc_lr,
        bc_batch_size=args.bc_batch_size,
    )

    # Overrides
    if args.steps:
        config.total_timesteps = args.steps

    print(f"Starting training: {config.algo} (Style {config.style}) on {config.device}")
    if config.bc_pretrain:
        print(f"Behavioral Cloning: Enabled (demos: {config.demo_path})")

    runner = TrainerRunner(config)
    runner.run()


if __name__ == "__main__":
    main()
