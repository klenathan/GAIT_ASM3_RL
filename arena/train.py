"""
Main entry point for training RL agents with PufferLib.
Replaces old SB3-based training script.
"""

import argparse
from arena.training.pufferl_trainer import PufferTrainer, PufferTrainConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Arena RL Trainer - PufferLib Edition",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Basic settings
    parser.add_argument("--algo", type=str, default="ppo", help="Algorithm (only PPO supported)")
    parser.add_argument("--style", type=int, default=1, choices=[1, 2], help="Control style")
    parser.add_argument("--env-type", type=str, default="standard", 
                       choices=["standard", "dict", "cnn"], help="Environment observation type")
    parser.add_argument("--policy", type=str, default="mlp",
                       choices=["mlp", "lstm", "cnn"], help="Policy architecture")
    
    # Training settings
    parser.add_argument("--steps", type=int, default=1_000_000, help="Total training timesteps")
    parser.add_argument("--device", type=str, default="cuda", 
                       choices=["cuda", "cpu", "mps"], help="Device to use")
    parser.add_argument("--num-envs", type=int, default=16, help="Number of parallel environments")
    parser.add_argument("--vec-backend", type=str, default="Multiprocessing",
                       choices=["Serial", "Multiprocessing"], help="Vectorization backend")
    
    # Hyperparameters
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size (steps per update)")
    parser.add_argument("--minibatch-size", type=int, default=128, help="Minibatch size for PPO updates")
    parser.add_argument("--epochs", type=int, default=4, help="Number of epochs per batch")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip-eps", type=float, default=0.2, help="PPO clip epsilon")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--value-coef", type=float, default=0.5, help="Value loss coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Max gradient norm")
    
    # Policy architecture
    parser.add_argument("--hidden-size", type=int, default=256, help="Hidden layer size")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of hidden layers")
    parser.add_argument("--activation", type=str, default="ReLU", help="Activation function")
    
    # Normalization
    parser.add_argument("--no-normalize-obs", action="store_true", help="Disable observation normalization")
    parser.add_argument("--no-normalize-rewards", action="store_true", help="Disable reward normalization")
    
    # Curriculum
    parser.add_argument("--no-curriculum", action="store_true", help="Disable curriculum learning")
    
    # Logging & checkpoints
    parser.add_argument("--log-interval", type=int, default=10, help="Log every N batches")
    parser.add_argument("--checkpoint-freq", type=int, default=100_000, help="Checkpoint frequency (steps)")
    parser.add_argument("--runs-dir", type=str, default="./runs", help="Base directory for runs")
    parser.add_argument("--name", type=str, default=None, help="Experiment name (auto-generated if not set)")
    
    # LR schedule
    parser.add_argument("--lr-schedule", type=str, default="constant",
                       choices=["constant", "linear", "cosine"], help="Learning rate schedule")
    parser.add_argument("--lr-warmup", type=int, default=0, help="LR warmup steps")
    parser.add_argument("--lr-min", type=float, default=1e-6, help="Minimum LR for schedules")
    
    # Logging backends
    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="arena-rl", help="W&B project name")
    parser.add_argument("--neptune", action="store_true", help="Use Neptune logging")
    parser.add_argument("--neptune-project", type=str, help="Neptune project name")
    parser.add_argument("--tags", nargs="+", help="Tags for logging")
    
    # Resume training
    parser.add_argument("--load-checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create config from args
    config = PufferTrainConfig(
        algo=args.algo,
        env_type=args.env_type,
        style=args.style,
        total_timesteps=args.steps,
        device=args.device,
        num_envs=args.num_envs,
        vec_backend=args.vec_backend,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        minibatch_size=args.minibatch_size,
        num_epochs=args.epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_eps=args.clip_eps,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        policy_type=args.policy,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        activation=args.activation,
        normalize_obs=not args.no_normalize_obs,
        normalize_rewards=not args.no_normalize_rewards,
        curriculum_enabled=not args.no_curriculum,
        log_interval=args.log_interval,
        checkpoint_freq=args.checkpoint_freq,
        runs_dir=args.runs_dir,
        experiment_name=args.name,
        lr_schedule=args.lr_schedule,
        lr_warmup_steps=args.lr_warmup,
        lr_min=args.lr_min,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project if args.wandb else None,
        use_neptune=args.neptune,
        neptune_project=args.neptune_project if args.neptune else None,
        tags=args.tags,
        load_checkpoint=args.load_checkpoint,
    )
    
    print("=" * 60)
    print("Arena RL Training - PufferLib Edition")
    print("=" * 60)
    print(f"Algorithm: {config.algo}")
    print(f"Environment: {config.env_type}")
    print(f"Policy: {config.policy_type}")
    print(f"Control Style: {config.style}")
    print(f"Device: {config.device}")
    print(f"Total Steps: {config.total_timesteps:,}")
    print(f"Num Envs: {config.num_envs}")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Batch Size: {config.batch_size}")
    print("=" * 60)
    
    # Create and run trainer
    trainer = PufferTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
