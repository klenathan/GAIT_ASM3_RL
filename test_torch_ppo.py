"""
Test script for PyTorch-based PPO training.
Validates GPU acceleration and training loop.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arena.core.config import TrainerConfig
from arena.training.algorithms.ppo_torch_trainer import PPOTorchTrainer

def main():
    """Test PyTorch PPO training for a few steps."""
    print("=" * 60)
    print("PyTorch PPO Test - GPU Acceleration Validation")
    print("=" * 60)
    
    # Create minimal config for testing
    config = TrainerConfig(
        algo="ppo_torch",
        style=2,  # Directional movement
        total_timesteps=10000,  # Just 10k steps for testing
        num_envs=4,  # Small number for testing
        checkpoint_freq=5000,
        render=False,
        device="cuda",  # Force CUDA if available
        progress_bar=True,
    )
    
    print(f"\nConfiguration:")
    print(f"  Algorithm: {config.algo}")
    print(f"  Control Style: {config.style}")
    print(f"  Total Timesteps: {config.total_timesteps}")
    print(f"  Num Environments: {config.num_envs}")
    print(f"  Device: {config.device}")
    print()
    
    # Create trainer
    trainer = PPOTorchTrainer(config)
    
    print(f"Trainer created successfully")
    print(f"Run directory: {trainer.run_dir}")
    print()
    
    # Run training
    try:
        model = trainer.train()
        print("\n" + "=" * 60)
        print("✓ Training completed successfully!")
        print("=" * 60)
        print(f"\nFinal model saved to: {trainer.final_dir}")
        print(f"Logs available at: {trainer.log_dir}")
        print("\nTo view training progress:")
        print(f"  tensorboard --logdir {trainer.log_dir}")
        
        return 0
    
    except Exception as e:
        print("\n" + "=" * 60)
        print("✗ Training failed!")
        print("=" * 60)
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
