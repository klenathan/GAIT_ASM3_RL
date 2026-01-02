"""
Performance comparison between SB3 (CPU-bottleneck) and PyTorch (GPU-optimized).
"""

import sys
import os
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_gpu_usage():
    """Check if CUDA is available and report GPU memory."""
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"  Current allocated: {torch.cuda.memory_allocated() / 1e9:.3f} GB")
        return True
    else:
        print("✗ CUDA not available")
        return False

def benchmark_torch_ppo():
    """Benchmark PyTorch-based PPO."""
    from arena.core.config import TrainerConfig
    from arena.training.algorithms.ppo_torch_trainer import PPOTorchTrainer
    
    print("\n" + "="*70)
    print("BENCHMARKING: PyTorch-Native PPO (GPU-Optimized)")
    print("="*70)
    
    config = TrainerConfig(
        algo="ppo_torch",
        style=2,
        total_timesteps=50000,
        num_envs=8,
        checkpoint_freq=0,  # Disable checkpoints for benchmarking
        render=False,
        device="cuda",
        progress_bar=False,
    )
    
    print(f"\nConfiguration:")
    print(f"  Environments: {config.num_envs}")
    print(f"  Total timesteps: {config.total_timesteps}")
    print(f"  Device: {config.device}")
    
    check_gpu_usage()
    
    trainer = PPOTorchTrainer(config)
    
    print(f"\nStarting training...")
    start_time = time.time()
    
    try:
        model = trainer.train()
        elapsed = time.time() - start_time
        
        print(f"\n✓ Training completed!")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Steps/sec: {config.total_timesteps / elapsed:.0f}")
        print(f"  FPS (env interactions): {config.total_timesteps / elapsed:.0f}")
        
        if torch.cuda.is_available():
            print(f"  Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.3f} GB")
            torch.cuda.reset_peak_memory_stats()
        
        return {
            'success': True,
            'time': elapsed,
            'steps_per_sec': config.total_timesteps / elapsed,
            'timesteps': config.total_timesteps,
        }
    
    except Exception as e:
        print(f"\n✗ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'success': False}

def benchmark_sb3_ppo():
    """Benchmark SB3 PPO for comparison."""
    from arena.core.config import TrainerConfig
    from arena.training.algorithms.ppo import PPOTrainer
    
    print("\n" + "="*70)
    print("BENCHMARKING: SB3 PPO (CPU Multiprocessing)")
    print("="*70)
    
    config = TrainerConfig(
        algo="ppo",
        style=2,
        total_timesteps=50000,
        num_envs=8,
        checkpoint_freq=0,
        render=False,
        device="cuda",
        progress_bar=False,
    )
    
    print(f"\nConfiguration:")
    print(f"  Environments: {config.num_envs}")
    print(f"  Total timesteps: {config.total_timesteps}")
    print(f"  Device: {config.device}")
    
    check_gpu_usage()
    
    trainer = PPOTrainer(config)
    
    print(f"\nStarting training...")
    start_time = time.time()
    
    try:
        model = trainer.train()
        elapsed = time.time() - start_time
        
        print(f"\n✓ Training completed!")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Steps/sec: {config.total_timesteps / elapsed:.0f}")
        print(f"  FPS (env interactions): {config.total_timesteps / elapsed:.0f}")
        
        if torch.cuda.is_available():
            print(f"  Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.3f} GB")
            torch.cuda.reset_peak_memory_stats()
        
        return {
            'success': True,
            'time': elapsed,
            'steps_per_sec': config.total_timesteps / elapsed,
            'timesteps': config.total_timesteps,
        }
    
    except Exception as e:
        print(f"\n✗ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'success': False}

def main():
    print("="*70)
    print("GPU ACCELERATION BENCHMARK")
    print("Comparing SB3 (CPU bottleneck) vs PyTorch (GPU-optimized)")
    print("="*70)
    
    if not check_gpu_usage():
        print("\n⚠ Warning: CUDA not available. Benchmark will run on CPU.")
    
    # Benchmark PyTorch implementation
    torch_results = benchmark_torch_ppo()
    
    # Benchmark SB3 implementation
    print("\n" + "="*70)
    print("Now benchmarking SB3 for comparison...")
    print("="*70)
    sb3_results = benchmark_sb3_ppo()
    
    # Compare results
    if torch_results['success'] and sb3_results['success']:
        print("\n" + "="*70)
        print("PERFORMANCE COMPARISON")
        print("="*70)
        
        speedup = torch_results['steps_per_sec'] / sb3_results['steps_per_sec']
        time_saved = sb3_results['time'] - torch_results['time']
        time_saved_pct = (time_saved / sb3_results['time']) * 100
        
        print(f"\nSB3 PPO (CPU multiprocessing):")
        print(f"  Time: {sb3_results['time']:.2f}s")
        print(f"  Throughput: {sb3_results['steps_per_sec']:.0f} steps/sec")
        
        print(f"\nPyTorch PPO (GPU-optimized):")
        print(f"  Time: {torch_results['time']:.2f}s")
        print(f"  Throughput: {torch_results['steps_per_sec']:.0f} steps/sec")
        
        print(f"\n🚀 SPEEDUP: {speedup:.2f}x faster")
        print(f"⏱️  TIME SAVED: {time_saved:.1f}s ({time_saved_pct:.1f}%)")
        
        print("\nFor 1M timesteps training:")
        print(f"  SB3: ~{(1_000_000 / sb3_results['steps_per_sec']) / 60:.1f} minutes")
        print(f"  PyTorch: ~{(1_000_000 / torch_results['steps_per_sec']) / 60:.1f} minutes")
        print(f"  Time saved: ~{((1_000_000 / sb3_results['steps_per_sec']) - (1_000_000 / torch_results['steps_per_sec'])) / 60:.1f} minutes")

if __name__ == "__main__":
    main()
