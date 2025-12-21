"""
Training script for Deep RL Arena
Supports DQN and PPO with both control styles
"""

import argparse
import os
import time
from datetime import datetime

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
import torch
import numpy as np

from arena.environment import ArenaEnv
from arena.callbacks import ArenaCallback, PerformanceCallback
from arena import config
from arena.sb3_utils import resolve_activation_fn


def get_device():
    """Auto-detect best available device (MPS > CUDA > CPU)"""
    if torch.backends.mps.is_available():
        device = "mps"
        print("✓ Using Mac Silicon GPU (MPS) for training")
    elif torch.cuda.is_available():
        device = "cuda"
        print(f"✓ Using CUDA GPU ({torch.cuda.get_device_name(0)}) for training")
    else:
        device = "cpu"
        print("⚠ Using CPU for training (slower)")
    return device


def setup_torch_optimizations(device):
    """Configure PyTorch for optimal performance"""
    # Set number of threads for CPU operations
    if device == "cpu":
        # Default to a sensible fraction of available CPU cores (override via TORCH_NUM_THREADS)
        # Using all cores can sometimes hurt due to oversubscription with env workers.
        try:
            env_threads = os.environ.get("TORCH_NUM_THREADS")
            if env_threads is not None:
                torch_threads = max(1, int(env_threads))
            else:
                cpu_count = os.cpu_count() or 4
                torch_threads = max(1, min(8, cpu_count // 2))
            torch.set_num_threads(torch_threads)
        except Exception:
            # Fall back to PyTorch defaults if anything goes wrong
            pass
    
    # Enable cuDNN benchmarking for faster convolutions (if using CUDA)
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        # Prefer faster matmul kernels (works well with TF32 on Ampere+)
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            # Older torch versions may not support this API
            pass
    
    # Enable TF32 for faster matrix multiplications on Ampere GPUs
    if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    print(f"✓ PyTorch optimizations configured for {device}")

def _recommended_cuda_num_envs() -> int:
    """
    Pick a reasonable default for SubprocVecEnv on this machine.
    On Windows, too many processes can hurt more than it helps; this picks a CPU-aware cap.
    """
    cpu_count = os.cpu_count() or 4
    # Keep at least 2 to benefit from parallel stepping, but avoid oversubscribing heavily.
    return int(min(config.NUM_ENVS_DEFAULT_CUDA, max(2, cpu_count // 2)))

def _limit_torch_threads_for_vecenv(num_envs: int, device: str) -> None:
    """
    Avoid CPU thread oversubscription when using many SubprocVecEnv workers.
    This commonly improves throughput on Windows + CUDA setups.
    """
    # Only clamp threads when the model runs on an accelerator (CUDA/MPS).
    # If the model itself runs on CPU, clamping to 1 often *reduces* throughput.
    if num_envs <= 1 or device == "cpu":
        return
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

def _pick_ppo_batch_size(n_steps: int, num_envs: int, preferred: int) -> int:
    """
    PPO requirement: batch_size must divide (n_steps * num_envs).
    Choose a GPU-friendly batch close to 'preferred' when possible.
    """
    rollout_size = int(n_steps) * int(num_envs)
    preferred = int(min(preferred, rollout_size))
    for bs in (1024, 512, 256, 2048, 128, 64):
        if bs <= rollout_size and rollout_size % bs == 0:
            return bs
    # Fallback: find a divisor <= preferred (bounded loop, preferred is <= 1024 in our usage)
    for bs in range(max(preferred, 64), 31, -1):
        if rollout_size % bs == 0:
            return bs
    return min(64, rollout_size)

def make_env(control_style, render_mode=None):
    """Create environment factory for vectorization"""
    def _init():
        return ArenaEnv(control_style=control_style, render_mode=render_mode)
    return _init


def train(args):
    """Main training function"""
    
    # Create directories
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(config.TENSORBOARD_LOG_DIR, exist_ok=True)
    
    # Setup device
    if args.device == "auto":
        device = get_device()
    else:
        device = args.device
        print(f"Using manually specified device: {device}")

    # Validate manually selected accelerators (avoid crashing on CPU-only torch builds)
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠ Requested CUDA device, but this PyTorch build has no CUDA support or no GPU is visible.")
        print("  Falling back to CPU. Install a CUDA-enabled PyTorch to train on GPU.")
        device = "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        print("⚠ Requested MPS device, but MPS is not available on this machine.")
        print("  Falling back to CPU.")
        device = "cpu"
    
    # Setup PyTorch optimizations
    setup_torch_optimizations(device)
    
    # Determine number of environments
    if args.num_envs is None:
        if device == "mps":
            num_envs = config.NUM_ENVS_DEFAULT_MPS
        elif device == "cuda":
            num_envs = _recommended_cuda_num_envs()
        else:
            num_envs = config.NUM_ENVS_DEFAULT_CPU
    else:
        num_envs = args.num_envs
    
    # IMPORTANT: MPS doesn't work with SubprocVecEnv due to Metal/fork incompatibility
    # If using MPS with multiple environments, switch to CPU for environments
    env_device = device
    if device == "mps" and num_envs > 1:
        print("\n⚠ Note: Mac Silicon (MPS) doesn't support multiprocessing with SubprocVecEnv")
        print("  Using CPU for parallel environments, but MPS for neural network training")
        print("  This still provides significant speedup from parallelization.\n")
        env_device = "cpu"
        # Keep MPS for the actual model training
    
    # Timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.algo}_style{args.style}_{timestamp}"
    
    print("=" * 60)
    print(f"Deep RL Arena Training")
    print("=" * 60)
    print(f"Algorithm: {args.algo.upper()}")
    print(f"Control Style: {args.style}")
    print(f"Total Timesteps: {args.steps:,}")
    print(f"Parallel Environments: {num_envs}")
    print(f"Run Name: {run_name}")
    print(f"Model Device: {device}")
    if env_device != device:
        print(f"Env Device: {env_device} (workaround for MPS multiprocessing)")
    print("=" * 60)
    
    # Create vectorized environments
    render_mode = "human" if args.render else None
    
    # Use SubprocVecEnv for true parallelization (much faster)
    # Falls back to DummyVecEnv if there's only 1 environment or rendering is enabled
    if num_envs > 1 and not args.render:
        try:
            print(f"Creating {num_envs} parallel environments with SubprocVecEnv...")
            env = SubprocVecEnv([make_env(args.style, render_mode) for _ in range(num_envs)])
            print("✓ Parallel environments created successfully")
            # Reduce CPU oversubscription when we have multiple env worker processes.
            # This is especially important on Windows.
            _limit_torch_threads_for_vecenv(num_envs, device)
        except Exception as e:
            print(f"⚠ Failed to create SubprocVecEnv: {e}")
            print("  Falling back to DummyVecEnv (slower)")
            env = DummyVecEnv([make_env(args.style, render_mode) for _ in range(num_envs)])
    else:
        if args.render:
            print("Using single environment (rendering enabled)")
        env = DummyVecEnv([make_env(args.style, render_mode)])
    
    # Select hyperparameters based on device (GPU allows larger batches)
    use_gpu_params = device in ["cuda", "mps"]
    
    # Create model based on algorithm
    if args.algo == "dqn":
        # Use GPU-optimized hyperparameters if available
        hyperparams = config.DQN_HYPERPARAMS_GPU if use_gpu_params else config.DQN_HYPERPARAMS
        
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=hyperparams['learning_rate'],
            buffer_size=hyperparams['buffer_size'],
            batch_size=hyperparams['batch_size'],
            gamma=hyperparams['gamma'],
            exploration_fraction=hyperparams['exploration_fraction'],
            exploration_initial_eps=hyperparams['exploration_initial_eps'],
            exploration_final_eps=hyperparams['exploration_final_eps'],
            target_update_interval=hyperparams['target_update_interval'],
            train_freq=hyperparams['train_freq'],
            gradient_steps=hyperparams['gradient_steps'],
            learning_starts=hyperparams['learning_starts'],
            policy_kwargs=dict(
                net_arch=config.DQN_HIDDEN_LAYERS,
                activation_fn=resolve_activation_fn(config.DQN_ACTIVATION),
            ),
            tensorboard_log=config.TENSORBOARD_LOG_DIR,
            device=device,
            verbose=1,
        )
    
    elif args.algo == "ppo":
        # Use GPU-optimized hyperparameters if available
        hyperparams = config.PPO_HYPERPARAMS_GPU if use_gpu_params else config.PPO_HYPERPARAMS
        # Ensure PPO batch_size divides the rollout size (n_steps * num_envs), for both speed and SB3 constraints.
        rollout_size = hyperparams["n_steps"] * env.num_envs
        preferred_bs = int(hyperparams["batch_size"])
        tuned_bs = _pick_ppo_batch_size(hyperparams["n_steps"], env.num_envs, preferred_bs)
        if tuned_bs != preferred_bs:
            print(f"✓ Adjusted PPO batch_size {preferred_bs} -> {tuned_bs} to divide rollout_size={rollout_size}")
        
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=hyperparams['learning_rate'],
            n_steps=hyperparams['n_steps'],
            batch_size=tuned_bs,
            n_epochs=hyperparams['n_epochs'],
            gamma=hyperparams['gamma'],
            gae_lambda=hyperparams['gae_lambda'],
            clip_range=hyperparams['clip_range'],
            ent_coef=hyperparams['ent_coef'],
            vf_coef=hyperparams['vf_coef'],
            max_grad_norm=hyperparams['max_grad_norm'],
            policy_kwargs=dict(
                net_arch=config.PPO_NET_ARCH,
                ortho_init=True,  # Orthogonal initialization
                activation_fn=resolve_activation_fn(config.PPO_ACTIVATION),
            ),
            tensorboard_log=config.TENSORBOARD_LOG_DIR,
            device=device,
            verbose=1,
        )
    
    else:
        raise ValueError(f"Unknown algorithm: {args.algo}")
    
    # Setup checkpoint directory with algo/style structure
    checkpoint_dir = os.path.join(config.MODEL_SAVE_DIR, args.algo, f"style{args.style}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=checkpoint_dir,
        name_prefix=run_name,
        save_replay_buffer=args.save_replay_buffer,
        save_vecnormalize=args.save_vecnormalize,
    )
    
    arena_callback = ArenaCallback(verbose=0)
    performance_callback = PerformanceCallback(verbose=0)
    
    callbacks = CallbackList([checkpoint_callback, arena_callback, performance_callback])
    
    print("\nStarting training...")
    print(f"TensorBoard: tensorboard --logdir {config.TENSORBOARD_LOG_DIR}")
    print("=" * 60)
    
    # Train the model
    model.learn(
        total_timesteps=args.steps,
        callback=callbacks,
        tb_log_name=run_name,
        progress_bar=args.progress,
    )
    
    # Save final model to the same algo/style directory
    final_model_path = os.path.join(checkpoint_dir, f"{run_name}_final")
    model.save(final_model_path)
    
    print("=" * 60)
    print(f"Training complete!")
    print(f"Final model saved to: {final_model_path}.zip")
    print("=" * 60)
    
    env.close()
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Train Deep RL Agent in Arena")
    
    parser.add_argument(
        "--algo",
        type=str,
        choices=["dqn", "ppo"],
        required=True,
        help="RL algorithm to use"
    )
    
    parser.add_argument(
        "--style",
        type=int,
        choices=[1, 2],
        required=True,
        help="Control style: 1=rotation/thrust, 2=directional"
    )
    
    parser.add_argument(
        "--steps",
        type=int,
        default=100_000,
        help="Total training timesteps"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use for training (auto=detect best available)"
    )
    
    parser.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help="Number of parallel environments (auto if not specified)"
    )
    
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render environment during training (slows down training)"
    )

    parser.add_argument(
        "--progress",
        action="store_true",
        default=True,
        help="Enable SB3 progress bar (can reduce FPS on fast runs)"
    )

    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=50_000,
        help="Checkpoint save frequency in timesteps (larger is faster)"
    )

    parser.add_argument(
        "--save-replay-buffer",
        action="store_true",
        help="Also save replay buffer in checkpoints (DQN only, slow)"
    )

    parser.add_argument(
        "--save-vecnormalize",
        action="store_true",
        help="Also save VecNormalize stats in checkpoints (usually unnecessary here)"
    )
    
    args = parser.parse_args()
    
    # Train
    train(args)


if __name__ == "__main__":
    main()
