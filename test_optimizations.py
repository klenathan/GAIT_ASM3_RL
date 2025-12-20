#!/usr/bin/env python
"""
Quick test to verify training optimizations
"""
import torch
import sys

print("=" * 60)
print("Training Optimization Verification")
print("=" * 60)

# Test 1: MPS Backend
print("\n1. Testing MPS Backend Detection...")
if torch.backends.mps.is_available():
    print("   ✓ MPS (Mac Silicon GPU) is AVAILABLE")
    print(f"   ✓ MPS Built: {torch.backends.mps.is_built()}")
    device = "mps"
else:
    print("   ⚠ MPS not available, checking CUDA...")
    if torch.cuda.is_available():
        print(f"   ✓ CUDA GPU available: {torch.cuda.get_device_name(0)}")
        device = "cuda"
    else:
        print("   Using CPU (slower)")
        device = "cpu"

print(f"   Selected device: {device}")

# Test 2: Simple tensor operation on device
print("\n2. Testing Tensor Operations on Device...")
try:
    x = torch.randn(100, 100, device=device)
    y = torch.randn(100, 100, device=device)
    z = torch.matmul(x, y)
    print(f"   ✓ Successfully performed matrix multiplication on {device}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 3: Vectorized environments
print("\n3. Testing Vectorized Environments...")
try:
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from arena.train import make_env
    
    print("   Creating 4 parallel environments...")
    env = SubprocVecEnv([make_env(2) for _ in range(4)])
    print(f"   ✓ Created {env.num_envs} parallel environments")
    
    # Test reset
    obs = env.reset()
    print(f"   ✓ Reset successful, observation shape: {obs[0].shape if isinstance(obs, tuple) else obs.shape}")
    
    env.close()
    print("   ✓ Environments closed successfully")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("All tests passed! ✓")
print("=" * 60)
