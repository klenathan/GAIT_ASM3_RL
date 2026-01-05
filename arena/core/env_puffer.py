import gymnasium as gym
import pufferlib.emulation
from arena.core.environment import ArenaEnv
from arena.core.config import TrainerConfig, OBS_DIM


def make_puffer_env(
    config: TrainerConfig = None, render_mode=None, buf=None, seed=0, **kwargs
):
    """
    Creates and wraps the ArenaEnv for PufferLib compatibility.

    Args:
        config: Trainer configuration object
        render_mode: Render mode for the environment
        buf: PufferLib shared memory buffer
        seed: Random seed

    Returns:
        A GymnasiumPufferEnv instance
    """

    # Create the base environment
    def env_creator():
        style = config.style if config else 1
        return ArenaEnv(control_style=style, render_mode=render_mode)

    return pufferlib.emulation.GymnasiumPufferEnv(
        env_creator=env_creator,
        buf=buf,
    )
