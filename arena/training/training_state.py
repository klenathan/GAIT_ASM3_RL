"""
Training state management for checkpointing and transfer learning.
Provides unified serialization for all training-related state.
"""

import os
import json
import glob
from dataclasses import dataclass, asdict
from typing import Optional
from datetime import datetime


@dataclass
class TrainingState:
    """Complete training state for checkpointing."""
    # Model info
    model_path: str
    algo: str
    style: int
    
    # Training progress
    total_timesteps_completed: int
    total_episodes: int
    
    # Curriculum state
    curriculum_stage_index: int
    curriculum_metrics: dict  # serialized CurriculumMetrics
    
    # Paths to related files
    vecnormalize_path: Optional[str] = None
    replay_buffer_path: Optional[str] = None
    
    # Metadata
    created_at: str = ""
    last_updated: str = ""
    
    def __post_init__(self):
        """Ensure timestamps are set."""
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.last_updated:
            self.last_updated = datetime.now().isoformat()


def save_training_state(path: str, state: TrainingState) -> None:
    """
    Save training state to JSON file.
    
    Args:
        path: Output path (e.g., /path/to/model_training_state.json)
        state: TrainingState instance to save
    """
    state.last_updated = datetime.now().isoformat()
    
    with open(path, 'w') as f:
        json.dump(asdict(state), f, indent=2)
    
    print(f"Training state saved: {path}")


def load_training_state(path: str) -> Optional[TrainingState]:
    """
    Load training state from JSON file.
    
    Args:
        path: Path to training state JSON
        
    Returns:
        TrainingState instance or None if file doesn't exist
    """
    if not os.path.exists(path):
        return None
    
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        return TrainingState(**data)
    except Exception as e:
        print(f"Warning: Failed to load training state from {path}: {e}")
        return None


def find_training_state(model_path: str) -> Optional[TrainingState]:
    """
    Auto-discover training state file for a given model checkpoint.
    
    Looks for a file matching the pattern:
    - Same directory as model
    - Same base name with _training_state.json suffix
    
    Args:
        model_path: Path to model .zip file
        
    Returns:
        TrainingState instance or None if not found
    """
    model_dir = os.path.dirname(model_path)
    model_name = os.path.basename(model_path).replace('.zip', '')
    
    # Try exact match first
    state_path = os.path.join(model_dir, f"{model_name}_training_state.json")
    state = load_training_state(state_path)
    if state:
        return state
    
    # Try pattern match (for checkpoints with step numbers)
    # Extract run prefix (e.g., ppo_style2_20251222_154509)
    parts = model_name.split('_')
    if len(parts) >= 4:
        run_prefix = '_'.join(parts[:4])  # algo_styleX_YYYYMMDD_HHMMSS
        pattern = os.path.join(model_dir, f"{run_prefix}_*_training_state.json")
        matches = sorted(glob.glob(pattern), reverse=True)  # Latest first
        if matches:
            return load_training_state(matches[0])
    
    return None


def get_training_state_path(model_path: str) -> str:
    """
    Get the training state path for a given model path.
    
    Args:
        model_path: Path to model .zip file
        
    Returns:
        Path where training state should be saved
    """
    return model_path.replace('.zip', '_training_state.json')
