"""
Registry for model output extractors.
"""

from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BaseExtractor


class ExtractorRegistry:
    """
    Registry for model output extractors with priority-based selection.
    
    Higher priority extractors are checked first, allowing specialized
    extractors to override generic ones.
    """
    
    _extractors: List[Tuple['BaseExtractor', int]] = []
    
    @classmethod
    def register(cls, extractor: 'BaseExtractor', priority: int = 0):
        """
        Register an extractor with a given priority.
        
        Args:
            extractor: Extractor instance to register
            priority: Priority level (higher = checked first)
        """
        cls._extractors.append((extractor, priority))
    
    @classmethod
    def get_extractor(cls, model) -> 'BaseExtractor':
        """
        Get the appropriate extractor for a given model.
        
        Args:
            model: SB3 model instance
            
        Returns:
            Extractor that supports the model
            
        Raises:
            ValueError: If no extractor supports the model
        """
        # Sort by priority (descending) and check each extractor
        for extractor, _ in sorted(cls._extractors, key=lambda x: -x[1]):
            if extractor.supports(model):
                return extractor
        
        raise ValueError(
            f"No extractor found for model type: {model.__class__.__name__}. "
            f"Available extractors: {[e.__class__.__name__ for e, _ in cls._extractors]}"
        )
    
    @classmethod
    def clear(cls):
        """Clear all registered extractors (useful for testing)."""
        cls._extractors = []


