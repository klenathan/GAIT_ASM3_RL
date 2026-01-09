import pytest

from arena.core.curriculum import CurriculumConfig, CurriculumStage, CurriculumManager
from arena.core.environment import ArenaEnv
from arena.core import config


def test_spawner_multiplier_applied():
    # Base phase has 1 spawner in phase 0
    base_spawners = config.PHASE_CONFIG[0]["spawners"]

    # Create a custom curriculum with stage multiplier 3x
    stage = CurriculumStage(name="test", spawner_multiplier=3.0)
    cfg = CurriculumConfig(enabled=True, stages=[stage])
    cm = CurriculumManager(cfg)

    env = ArenaEnv(curriculum_manager=cm)
    obs, info = env.reset()

    assert len(env.spawners) == max(
        1, int(round(base_spawners * stage.spawner_multiplier)))
