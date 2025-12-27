"""
Base extractor class defining the interface for model output extraction.
"""

from abc import ABC, abstractmethod
from typing import Union, Dict, Any
import numpy as np
import torch

from .model_output import ModelOutput


class BaseExtractor(ABC):
    """Abstract base class for model output extractors."""
    
    @abstractmethod
    def supports(self, model) -> bool:
        """
        Check if this extractor can handle the given model.
        
        Args:
            model: The SB3 model instance
            
        Returns:
            True if this extractor supports the model, False otherwise
        """
        pass
    
    @abstractmethod
    def extract_internal(
        self, 
        model, 
        obs: Union[np.ndarray, Dict[str, np.ndarray]], 
        action: int,
        **kwargs
    ) -> ModelOutput:
        """
        Extract model output for visualization.
        
        Args:
            model: The SB3 model instance
            obs: Observation (array or dict)
            action: Action taken by the model
            **kwargs: Additional extractor-specific parameters
            
        Returns:
            ModelOutput containing extracted information
        """
        pass
    
    def _process_observation(
        self, 
        obs: Union[np.ndarray, Dict[str, np.ndarray]], 
        device: torch.device
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Convert observation to torch tensor with proper shape.
        
        Handles:
        - Array observations: adds batch dimension if needed
        - Dict observations: converts each value to tensor
        
        Args:
            obs: Observation (array or dict)
            device: Target device for tensors
            
        Returns:
            Tensor or dict of tensors with batch dimension
        """
        if isinstance(obs, dict):
            # Convert dict observation to tensors
            obs_tensor = {}
            for key, value in obs.items():
                # Convert to numpy array if not already
                if not isinstance(value, np.ndarray):
                    value = np.array(value)
                
                tensor = torch.as_tensor(value).to(device)
                # Ensure proper shape: add batch dim if needed
                if tensor.dim() == 0:
                    # Scalar - add two dimensions
                    tensor = tensor.unsqueeze(0).unsqueeze(0)
                elif tensor.dim() == 1:
                    # 1D array - add batch dimension: (N,) -> (1, N)
                    tensor = tensor.unsqueeze(0)
                # If already 2D with shape (1, N), keep it as is
                obs_tensor[key] = tensor
            return obs_tensor
        else:
            # Convert array observation to tensor
            tensor = torch.as_tensor(obs).to(device)
            if tensor.dim() == 0:
                # Scalar
                tensor = tensor.unsqueeze(0).unsqueeze(0)
            elif tensor.dim() == 1:
                # Add batch dimension
                tensor = tensor.unsqueeze(0)
            return tensor
    
    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Safely convert tensor to numpy array, handling various shapes.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Numpy array
        """
        return tensor.cpu().numpy()
    
    def _extract_scalar(self, tensor: torch.Tensor) -> float:
        """
        Extract scalar value from tensor, handling various shapes.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Scalar float value
        """
        arr = self._tensor_to_numpy(tensor)
        if arr.ndim == 0:
            return float(arr.item())
        elif arr.ndim == 1:
            return float(arr[0])
        elif arr.ndim == 2:
            return float(arr[0, 0])
        else:
            # Fallback: flatten and take first
            return float(arr.flat[0])
    
    def _extract_vector(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Extract vector from tensor, removing batch dimension if present.
        
        Args:
            tensor: Input tensor
            
        Returns:
            1D numpy array
        """
        arr = self._tensor_to_numpy(tensor)
        if arr.ndim > 1:
            # Remove batch dimension
            return arr[0]
        return arr

