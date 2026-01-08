"""
Algorithm registry for Deep RL Arena.
Allows dynamic registration and lookup of trainer classes.
"""

from typing import Dict, Type, TYPE_CHECKING, List

if TYPE_CHECKING:
    from arena.training.base import BaseTrainer

class AlgorithmRegistry:
    """Registry for RL algorithm trainers."""
    
    _trainers: Dict[str, Type["BaseTrainer"]] = {}
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register a trainer class."""
        def decorator(trainer_class: Type["BaseTrainer"]):
            cls._trainers[name.lower()] = trainer_class
            return trainer_class
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Type["BaseTrainer"]:
        """Get trainer class by algorithm name."""
        name_lower = name.lower()
        if name_lower not in cls._trainers:
            available = ", ".join(cls._trainers.keys())
            raise ValueError(
                f"Unknown algorithm: '{name}'. Available algorithms: {available}\n"
                "Make sure the algorithm module is imported so it can register itself."
            )
        return cls._trainers[name_lower]
    
    @classmethod
    def list_algorithms(cls) -> List[str]:
        """List all registered algorithm names sorted alphabetically."""
        return sorted(list(cls._trainers.keys()))
