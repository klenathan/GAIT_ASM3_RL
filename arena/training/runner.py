"""
Trainer runner to orchestrate the training process.
"""

from arena.core.config import TrainerConfig
from arena.training.registry import AlgorithmRegistry

# Import algorithms to ensure registration
import arena.training.algorithms.dqn
import arena.training.algorithms.ppo
import arena.training.algorithms.ppo_lstm

class TrainerRunner:
    """Orchestrates the training process based on configuration."""
    
    def __init__(self, config: TrainerConfig):
        self.config = config
        
    def run(self):
        """Execute training with the configured algorithm."""
        print(f"Loading trainer for: {self.config.algo}")
        trainer_class = AlgorithmRegistry.get(self.config.algo)
        trainer = trainer_class(self.config)
        
        return trainer.train()
