"""
Device management and PyTorch optimizations for Deep RL Arena.
"""

import os
import torch
import numpy as np

class DeviceManager:
    """Handles auto-detection of compute devices and runtime optimizations."""
    
    @staticmethod
    def get_device(requested_device: str = "auto") -> str:
        """
        Auto-detect best available device (MPS > CUDA > CPU) or use specified.
        """
        if requested_device != "auto":
            device = requested_device
        elif torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
            
        return device

    @staticmethod
    def setup_optimizations(device: str) -> None:
        """Configure PyTorch for optimal performance based on device."""
        # Set number of threads for CPU operations
        if device == "cpu":
            try:
                env_threads = os.environ.get("TORCH_NUM_THREADS")
                if env_threads is not None:
                    torch_threads = max(1, int(env_threads))
                else:
                    cpu_count = os.cpu_count() or 4
                    torch_threads = max(1, min(8, cpu_count // 2))
                torch.set_num_threads(torch_threads)
            except Exception:
                pass
        
        # Enable cuDNN benchmarking for faster convolutions
        if device == "cuda":
            torch.backends.cudnn.benchmark = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
        
        # Enable TF32 for faster matrix multiplications on Ampere GPUs
        if device == "cuda" and torch.cuda.is_available():
            try:
                if torch.cuda.get_device_capability()[0] >= 8:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
            except Exception:
                pass
                
    @staticmethod
    def limit_threads_for_vecenv(num_envs: int, device: str) -> None:
        """Avoid CPU thread oversubscription when using many parallel workers."""
        if num_envs <= 1 or device == "cpu":
            return
        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception:
            pass

    @staticmethod
    def get_recommended_num_envs(device: str) -> int:
        """Pick a reasonable default for parallel environments."""
        from arena.core.config import NUM_ENVS_DEFAULT_MPS, NUM_ENVS_DEFAULT_CUDA, NUM_ENVS_DEFAULT_CPU
        
        if device == "mps":
            cpu_count = os.cpu_count() or 4
            return int(min(NUM_ENVS_DEFAULT_MPS, cpu_count))
        elif device == "cuda":
            cpu_count = os.cpu_count() or 4
            return  cpu_count or  int(min(NUM_ENVS_DEFAULT_CUDA, max(2, cpu_count // 2)))
        else:
            return os.cpu_count() or NUM_ENVS_DEFAULT_CPU
