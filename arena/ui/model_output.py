"""
Extraction utilities for model output visualization.
Supports action probabilities (PPO/A2C), Q-values (DQN), and value estimates.
"""

from typing import Union, Dict
import numpy as np

# Import ModelOutput and registry from extractors
from arena.ui.extractors import ModelOutput, ExtractorRegistry

# Re-export ModelOutput for backwards compatibility
__all__ = ['ModelOutput', 'ModelOutputExtractor']


class ModelOutputExtractor:
    """
    Facade for extracting model output using registered extractors.
    
    This class delegates to algorithm-specific extractors registered
    in the ExtractorRegistry. To add support for new algorithms,
    create a new extractor subclassing BaseExtractor and register it
    in arena.ui.extractors.__init__.py.
    """
    
    def __init__(self):
        """Initialize the extractor facade."""
        self.registry = ExtractorRegistry
        
    def extract(
        self, 
        model, 
        obs: Union[np.ndarray, Dict[str, np.ndarray]], 
        action: int, 
        lstm_states=None, 
        episode_start=None
    ) -> ModelOutput:
        """
        Extract detailed model info for a single observation.
        
        Delegates to the appropriate extractor based on model type.
        
        Args:
            model: The loaded SB3 model (PPO, DQN, etc.)
            obs: The observation vector (numpy array) or dict observation
            action: The action chosen by model.predict()
            lstm_states: Recurrent states if applicable (for LSTM models)
            episode_start: Episode start flag if applicable (for LSTM models)
            
        Returns:
            ModelOutput dataclass with extracted information
            
        Raises:
            ValueError: If no extractor supports the model type
        """
        # Get the appropriate extractor for this model
        extractor = self.registry.get_extractor(model)
        
        # Delegate extraction to the specific extractor
        return extractor.extract_internal(
            model=model,
            obs=obs,
            action=action,
            lstm_states=lstm_states,
            episode_start=episode_start
        )
