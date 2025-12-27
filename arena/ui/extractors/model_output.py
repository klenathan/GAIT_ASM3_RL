"""
ModelOutput dataclass for storing extracted model information.
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class ModelOutput:
    """Container for model prediction metadata."""
    action_probs: Optional[np.ndarray] = None
    q_values: Optional[np.ndarray] = None
    value: Optional[float] = None
    entropy: Optional[float] = None
    action_taken: Optional[int] = None
    is_q_value: bool = False
    
    @property
    def labels(self) -> List[str]:
        """Names of actions; to be filled by the renderer based on style."""
        return []


