"""
PufferLib Environment Wrappers for Arena.

Provides factory functions to create PufferLib-compatible environments.
"""

import pufferlib
import pufferlib.emulation
from arena.core.environment import ArenaEnv
from arena.core.environment_dict import ArenaDictEnv
from arena.core.environment_cnn import ArenaCNNEnv
from arena.core.curriculum import CurriculumManager


def make_arena_env(control_style=1, render_mode=None, curriculum_manager=None):
    """
    Create standard Arena environment wrapped for PufferLib.
    
    Args:
        control_style: 1 (rotation+thrust) or 2 (directional)
        render_mode: None, 'human', or 'rgb_array'
        curriculum_manager: Optional CurriculumManager instance
    
    Returns:
        PufferLib-wrapped environment
    """
    def env_creator():
        return ArenaEnv(
            control_style=control_style,
            render_mode=render_mode,
            curriculum_manager=curriculum_manager,
        )
    
    return pufferlib.emulation.GymnasiumPufferEnv(env_creator=env_creator)


def make_arena_dict_env(control_style=1, render_mode=None, curriculum_manager=None):
    """
    Create Dict observation Arena environment wrapped for PufferLib.
    
    Args:
        control_style: 1 (rotation+thrust) or 2 (directional)
        render_mode: None, 'human', or 'rgb_array'
        curriculum_manager: Optional CurriculumManager instance
    
    Returns:
        PufferLib-wrapped environment
    """
    def env_creator():
        return ArenaDictEnv(
            control_style=control_style,
            render_mode=render_mode,
            curriculum_manager=curriculum_manager,
        )
    
    return pufferlib.emulation.GymnasiumPufferEnv(env_creator=env_creator)


def make_arena_cnn_env(control_style=1, render_mode=None, curriculum_manager=None):
    """
    Create CNN observation Arena environment wrapped for PufferLib.
    
    Args:
        control_style: 1 (rotation+thrust) or 2 (directional)
        render_mode: None, 'human', or 'rgb_array'
        curriculum_manager: Optional CurriculumManager instance
    
    Returns:
        PufferLib-wrapped environment
    """
    def env_creator():
        return ArenaCNNEnv(
            control_style=control_style,
            render_mode=render_mode,
            curriculum_manager=curriculum_manager,
        )
    
    return pufferlib.emulation.GymnasiumPufferEnv(env_creator=env_creator)


# Registry for easy lookup
ENV_REGISTRY = {
    'standard': make_arena_env,
    'dict': make_arena_dict_env,
    'cnn': make_arena_cnn_env,
}


def make_env(env_type='standard', **kwargs):
    """
    Generic environment factory.
    
    Args:
        env_type: 'standard', 'dict', or 'cnn'
        **kwargs: Passed to specific environment creator
    
    Returns:
        PufferLib-wrapped environment
    """
    if env_type not in ENV_REGISTRY:
        raise ValueError(f"Unknown env_type: {env_type}. Choose from {list(ENV_REGISTRY.keys())}")
    
    return ENV_REGISTRY[env_type](**kwargs)
