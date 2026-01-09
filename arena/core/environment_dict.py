"""
Deep RL Arena - Dict Observation Environment.
Variant of ArenaEnv that uses dictionary observation space for better feature grouping.
"""

import warnings

# Suppress pkg_resources deprecation warning from pygame
warnings.filterwarnings(
    "ignore", message=".*pkg_resources is deprecated.*", category=UserWarning
)

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import pygame

from arena.core import config
from arena.core.curriculum import CurriculumManager, CurriculumStage
from arena.game import utils
from arena.game.entities import Player, Enemy, Spawner, Projectile
from arena.ui.renderer import ArenaRenderer


class ArenaDictEnv(gym.Env):
    """
    Dict-observation variant of ArenaEnv.

    Uses dictionary observation space with semantic grouping:
    - player_state: Agent's own state (position, velocity, health, etc.)
    - combat_targets: Enemy and spawner information
    - mission_progress: Phase, objectives, time remaining
    - spatial_awareness: Wall distances and positioning
    - enemy_count: Number of active enemies

    Supports two control schemes:
    - Style 1: Rotation + Thrust (5 actions)
    - Style 2: Directional Movement (6 actions)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": config.FPS}

    def __init__(
        self,
        control_style=1,
        render_mode=None,
        curriculum_manager: CurriculumManager = None,
    ):
        super().__init__()

        self.control_style = control_style
        self.render_mode = render_mode
        self.curriculum_manager = curriculum_manager

        # Action space (same as original)
        if control_style == 1:
            self.action_space = spaces.Discrete(config.ACTION_SPACE_STYLE_1)
        else:
            self.action_space = spaces.Discrete(config.ACTION_SPACE_STYLE_2)

        # Dict observation space with semantic grouping
        self.observation_space = spaces.Dict(
            {
                # Player state: position (2), velocity (2), rotation (1), health (1),
                # shoot cooldown (1), total = 7 dims
                "player_state": spaces.Box(
                    low=-1.0, high=1.0, shape=(7,), dtype=np.float32
                ),
                # Combat targets: nearest enemies (2x3=6), nearest spawners (2x4=8),
                # nearest projectiles (5x3=15), total = 29 dims
                "combat_targets": spaces.Box(
                    low=-1.0, high=1.0, shape=(29,), dtype=np.float32
                ),
                # Mission progress: phase (1), spawners remaining (1), time remaining (1),
                # total = 3 dims
                "mission_progress": spaces.Box(
                    low=0.0, high=1.0, shape=(3,), dtype=np.float32
                ),
                # Spatial awareness: wall distances (4), total = 4 dims
                "spatial_awareness": spaces.Box(
                    low=0.0, high=1.0, shape=(4,), dtype=np.float32
                ),
                # Enemy count: number of active enemies (1), total = 1 dim
                "enemy_count": spaces.Box(
                    low=0.0, high=1.0, shape=(1,), dtype=np.float32
                ),
            }
        )

        self.renderer = None
        self._owns_renderer = False
        if render_mode == "human":
            self.renderer = ArenaRenderer()
            self._owns_renderer = True

        # Game state
        self.player = None
        self.enemies = []
        self.spawners = []
        self.projectiles = []

        self.current_step = 0
        self.current_phase = 0
        self.enemies_destroyed = 0
        self.spawners_destroyed = 0
        self.enemies_destroyed_this_step = 0
        self.spawners_destroyed_this_step = 0
        self.episode_reward = 0
        self.win = False
        self.win_step = None
        self.first_spawner_kill_step = None

        # Reward shaping state tracking
        self._prev_spawner_total_health = None
        self._prev_enemy_count = None
        self._prev_player_health = None
        self.phase_start_step = 0

    @property
    def curriculum_stage(self) -> CurriculumStage:
        """Get current curriculum stage, or None if curriculum disabled."""
        if self.curriculum_manager and self.curriculum_manager.enabled:
            return self.curriculum_manager.current_stage
        return None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.current_phase = 0
        self.enemies_destroyed = 0
        self.spawners_destroyed = 0
        self.enemies_destroyed_this_step = 0
        self.spawners_destroyed_this_step = 0
        self.episode_reward = 0
        self.phase_start_step = 0
        self.win = False
        self.win_step = None
        self.first_spawner_kill_step = None

        self._prev_spawner_total_health = None
        self._prev_enemy_count = None
        self._prev_player_health = None

        self.player = Player(config.GAME_WIDTH / 2, config.GAME_HEIGHT / 2)
        self.enemies = []
        self.spawners = []
        self.projectiles = []

        self._init_phase()

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.current_step += 1
        reward = 0.0
        done = False

        # Reset step-level counters
        self.enemies_destroyed_this_step = 0
        self.spawners_destroyed_this_step = 0

        # Update player based on control style
        if self.control_style == 1:
            self.player.update_style_1(action)
        else:
            self.player.update_style_2(action)

        # Handle shooting
        if (self.control_style == 1 and action == 4) or (
            self.control_style == 2 and action == 5
        ):
            if self.player.shoot():
                reward += float(config.REWARD_SHOT_FIRED)
                proj = Projectile(
                    self.player.pos[0],
                    self.player.pos[1],
                    self.player.rotation,
                    is_player_projectile=True,
                )
                self.projectiles.append(proj)

        # Update enemies
        for enemy in self.enemies:
            if enemy.alive:
                enemy.update(self.player.pos)
                if enemy.shoot(self.player.pos, self.np_random):
                    angle = utils.angle_to_point(enemy.pos, self.player.pos)
                    self.projectiles.append(
                        Projectile(enemy.pos[0], enemy.pos[1], angle, False)
                    )

        phase_cfg = config.PHASE_CONFIG[self.current_phase]

        # Apply curriculum modifiers
        max_enemies = config.SPAWNER_MAX_ENEMIES
        enemy_speed = phase_cfg["enemy_speed_mult"]
        if self.curriculum_stage:
            max_enemies = int(max_enemies * self.curriculum_stage.max_enemies_mult)
            enemy_speed *= self.curriculum_stage.enemy_speed_mult

        # Update spawners
        for spawner in self.spawners:
            if spawner.alive:
                spawner.update()
                if spawner.can_spawn(
                    len([e for e in self.enemies if e.alive]), max_enemies
                ):
                    new_enemy = spawner.spawn_enemy(
                        self.np_random, enemy_speed, existing_enemies=self.enemies
                    )
                    if new_enemy:
                        self.enemies.append(new_enemy)

        # Update projectiles
        for proj in self.projectiles:
            if proj.alive:
                proj.update()

        # Handle collisions and rewards
        reward += self._handle_collisions()
        reward += float(config.REWARD_STEP_SURVIVAL)
        reward += self._calculate_shaping_reward()

        # Cleanup dead entities
        self.enemies = [e for e in self.enemies if e.alive]
        self.spawners = [s for s in self.spawners if s.alive]
        self.projectiles = [p for p in self.projectiles if p.alive]

        # Phase progression
        if len(self.spawners) == 0:
            reward += config.REWARD_PHASE_COMPLETE
            self.current_phase += 1
            if self.current_phase < config.MAX_PHASES:
                self._init_phase()
            else:
                self.win = True
                self.win_step = self.current_step
                done = True

        # Check death condition
        if not self.player.alive:
            reward += config.REWARD_DEATH
            done = True

        # Check time limit
        if self.current_step >= config.MAX_STEPS:
            done = True

        self.episode_reward += reward
        truncated = False
        return self._get_observation(), reward, done, truncated, self._get_info()

    def render(self):
        if self.render_mode is None:
            return
        if self.renderer is None and self.render_mode == "human":
            self.renderer = ArenaRenderer()

        if self.renderer:
            metrics = {
                "episode": 0,
                "episode_reward": self.episode_reward,
                "total_reward": self.episode_reward,
                "timesteps": self.current_step,
                "show_inputs": True,  # Show model inputs during rendering
            }
            self.renderer.render(self, metrics)

        try:
            pygame.event.pump()
        except:
            pass

    def close(self):
        if self.renderer and self._owns_renderer:
            self.renderer.close()
        self.renderer = None
        self._owns_renderer = False

    def _init_phase(self):
        """Initialize a new phase with spawners."""
        phase_cfg = config.PHASE_CONFIG[self.current_phase]
        num = phase_cfg["spawners"]
        self.enemies = []

        for i in range(num):
            x, y = self._get_spawner_position_smart(i, num)

            # Apply curriculum modifiers
            spawn_rate_mult = phase_cfg["spawn_rate_mult"]
            health_mult = 1.0
            if self.curriculum_stage:
                spawn_rate_mult *= self.curriculum_stage.spawn_cooldown_mult
                health_mult = self.curriculum_stage.spawner_health_mult

            spawner = Spawner(x, y, spawn_rate_mult)
            if health_mult != 1.0:
                spawner.health = int(spawner.max_health * health_mult)
            self.spawners.append(spawner)

        self.phase_start_step = self.current_step
        self._prev_spawner_total_health = None
        self._prev_enemy_count = None
        self._prev_player_health = None

    def _get_spawner_position_smart(self, spawner_index, total_spawners):
        """Generate random spawner position with minimum distance from player spawn."""
        margin = 100  # Minimum distance from arena edges
        min_dist_from_player = 250  # Minimum distance from player spawn (center-bottom)
        min_dist_between_spawners = 150  # Minimum distance between spawners

        w, h = config.GAME_WIDTH, config.GAME_HEIGHT
        player_spawn = (w / 2, h - 50)  # Player spawns at center-bottom

        max_attempts = 100
        for _ in range(max_attempts):
            x = self.np_random.uniform(margin, w - margin)
            y = self.np_random.uniform(margin, h - margin)

            # Check distance from player spawn
            dist_to_player = math.sqrt(
                (x - player_spawn[0]) ** 2 + (y - player_spawn[1]) ** 2
            )
            if dist_to_player < min_dist_from_player:
                continue

            # Check distance from existing spawners
            too_close = False
            for spawner in self.spawners:
                dist = math.sqrt((x - spawner.pos[0]) ** 2 + (y - spawner.pos[1]) ** 2)
                if dist < min_dist_between_spawners:
                    too_close = True
                    break

            if not too_close:
                return x, y

        # Fallback: return random position if no valid spot found
        return self.np_random.uniform(margin, w - margin), self.np_random.uniform(
            margin, h - margin
        )

    def _get_observation(self):
        """
        Build dictionary observation with semantic grouping.

        Returns:
            Dict with keys: player_state, combat_targets, mission_progress,
                           spatial_awareness, enemy_count
        """
        max_dist = math.sqrt(config.GAME_WIDTH**2 + config.GAME_HEIGHT**2)

        # --- Player State (7 dims) ---
        player_state = np.zeros(7, dtype=np.float32)
        player_state[0] = self.player.pos[0] / config.GAME_WIDTH  # x position
        player_state[1] = self.player.pos[1] / config.GAME_HEIGHT  # y position
        player_state[2] = np.clip(
            self.player.velocity[0] / config.PLAYER_MAX_VELOCITY, -1, 1
        )  # vx
        player_state[3] = np.clip(
            self.player.velocity[1] / config.PLAYER_MAX_VELOCITY, -1, 1
        )  # vy
        player_state[4] = self.player.rotation / (2 * math.pi)  # rotation
        player_state[5] = self.player.get_health_ratio()  # health ratio
        player_state[6] = (
            self.player.shoot_cooldown / config.PLAYER_SHOOT_COOLDOWN
        )  # cooldown

        # --- Combat Targets (29 dims) ---
        combat = np.zeros(29, dtype=np.float32)

        # Nearest 2 enemies (dist, angle, exists) x2 = 6 dims
        nearest_enemies = self._find_k_nearest_entities(self.enemies, k=2)
        for i, enemy in enumerate(nearest_enemies):
            base_idx = i * 3
            if enemy:
                combat[base_idx] = utils.distance(self.player.pos, enemy.pos) / max_dist
                combat[base_idx + 1] = utils.normalize_angle(
                    utils.relative_angle(
                        self.player.rotation,
                        utils.angle_to_point(self.player.pos, enemy.pos),
                    )
                )
                combat[base_idx + 2] = 1.0
            else:
                combat[base_idx], combat[base_idx + 1], combat[base_idx + 2] = (
                    1.0,
                    0.5,
                    0.0,
                )

        # Nearest 2 spawners (dist, angle, exists, health) x2 = 8 dims (but we use 4 per spawner)
        nearest_spawners = self._find_k_nearest_entities(self.spawners, k=2)
        for i, spawner in enumerate(nearest_spawners):
            base_idx = 6 + i * 4  # Start after 6 enemy dims
            if spawner:
                combat[base_idx] = (
                    utils.distance(self.player.pos, spawner.pos) / max_dist
                )
                combat[base_idx + 1] = utils.normalize_angle(
                    utils.relative_angle(
                        self.player.rotation,
                        utils.angle_to_point(self.player.pos, spawner.pos),
                    )
                )
                combat[base_idx + 2] = 1.0
                combat[base_idx + 3] = spawner.health / spawner.max_health
            else:
                combat[base_idx : base_idx + 4] = [1.0, 0.5, 0.0, 0.0]

        # Nearest 5 projectiles (dist, angle, exists) x5 = 15 dims at indices 14-28
        nearest_projectiles = self._get_nearest_projectiles(k=5)
        for i, proj_info in enumerate(nearest_projectiles):
            base_idx = 14 + i * 3
            if proj_info:
                combat[base_idx] = proj_info["dist"] / max_dist
                combat[base_idx + 1] = proj_info["angle"]
                combat[base_idx + 2] = 1.0
            else:
                combat[base_idx : base_idx + 3] = [1.0, 0.5, 0.0]

        # --- Mission Progress (3 dims) ---
        mission = np.zeros(3, dtype=np.float32)
        mission[0] = self.current_phase / config.MAX_PHASES  # current phase

        # Spawners remaining ratio
        phase_idx = min(self.current_phase, config.MAX_PHASES - 1)
        initial_spawners = config.PHASE_CONFIG[phase_idx]["spawners"]
        mission[1] = len([s for s in self.spawners if s.alive]) / max(
            initial_spawners, 1
        )

        mission[2] = 1.0 - (self.current_step / config.MAX_STEPS)  # time remaining

        # --- Spatial Awareness (4 dims) ---
        spatial = np.zeros(4, dtype=np.float32)
        spatial[0] = self.player.pos[0] / config.GAME_WIDTH  # distance from left
        spatial[1] = 1.0 - (
            self.player.pos[0] / config.GAME_WIDTH
        )  # distance from right
        spatial[2] = self.player.pos[1] / config.GAME_HEIGHT  # distance from top
        spatial[3] = 1.0 - (
            self.player.pos[1] / config.GAME_HEIGHT
        )  # distance from bottom

        # --- Enemy Count (1 dim) ---
        enemy_count = np.zeros(1, dtype=np.float32)
        enemy_count[0] = (
            len([e for e in self.enemies if e.alive]) / config.SPAWNER_MAX_ENEMIES
        )

        return {
            "player_state": player_state,
            "combat_targets": combat,
            "mission_progress": mission,
            "spatial_awareness": spatial,
            "enemy_count": enemy_count,
        }

    def _find_k_nearest_entities(self, entities, k=2):
        """Find k nearest entities, returns list padded with None if fewer exist."""
        alive_entities = [
            (e, utils.distance(self.player.pos, e.pos)) for e in entities if e.alive
        ]
        alive_entities.sort(key=lambda x: x[1])

        result = [e for e, _ in alive_entities[:k]]
        while len(result) < k:
            result.append(None)
        return result

    def _get_nearest_projectiles(self, k=5):
        """Get info about k nearest threatening projectiles (enemy projectiles only).

        Returns:
            List of k dicts with 'dist' and 'angle' keys, padded with None if fewer exist.
        """
        enemy_projectiles = [
            p for p in self.projectiles if p.alive and not p.is_player_projectile
        ]

        # Calculate distance and angle for each projectile
        proj_info = []
        for proj in enemy_projectiles:
            dist = utils.distance(self.player.pos, proj.pos)
            angle = utils.normalize_angle(
                utils.relative_angle(
                    self.player.rotation,
                    utils.angle_to_point(self.player.pos, proj.pos),
                )
            )
            proj_info.append({"dist": dist, "angle": angle})

        # Sort by distance and take k nearest
        proj_info.sort(key=lambda x: x["dist"])
        result = proj_info[:k]

        # Pad with None if fewer than k projectiles
        while len(result) < k:
            result.append(None)

        return result

    def _handle_collisions(self):
        """Handle all collision detection and return reward."""
        reward = 0.0

        damage_penalty = config.REWARD_DAMAGE_TAKEN
        if self.curriculum_stage:
            damage_penalty *= self.curriculum_stage.damage_penalty_mult

        # Player projectiles vs enemies and spawners
        for proj in self.projectiles:
            if not proj.alive or not proj.is_player_projectile:
                continue
            for enemy in self.enemies:
                if enemy.alive and utils.check_collision(
                    proj.pos, proj.radius, enemy.pos, enemy.radius
                ):
                    enemy.take_damage(proj.damage)
                    proj.hit()
                    reward += float(config.REWARD_HIT_ENEMY)
                    if not enemy.alive:
                        reward += config.REWARD_ENEMY_DESTROYED
                        self.enemies_destroyed += 1
                        self.enemies_destroyed_this_step += 1
                    break
            if not proj.alive:
                continue
            for spawner in self.spawners:
                if spawner.alive and utils.check_collision(
                    proj.pos, proj.radius, spawner.pos, spawner.radius
                ):
                    spawner.take_damage(proj.damage)
                    proj.hit()
                    reward += float(config.REWARD_HIT_SPAWNER)
                    if not spawner.alive:
                        reward += config.REWARD_SPAWNER_DESTROYED
                        self.spawners_destroyed += 1
                        self.spawners_destroyed_this_step += 1
                        if self.first_spawner_kill_step is None:
                            self.first_spawner_kill_step = self.current_step
                        if (self.current_step - self.phase_start_step) < 500:
                            reward += config.REWARD_QUICK_SPAWNER_KILL
                    break

        # Enemy projectiles vs player
        for proj in self.projectiles:
            if not proj.alive or proj.is_player_projectile:
                continue
            if self.player.alive and utils.check_collision(
                proj.pos, proj.radius, self.player.pos, self.player.radius
            ):
                self.player.take_damage(proj.damage)
                proj.hit()
                reward += damage_penalty

        # Enemy collision with player
        for enemy in self.enemies:
            if (
                enemy.alive
                and self.player.alive
                and utils.check_collision(
                    enemy.pos, enemy.radius, self.player.pos, self.player.radius
                )
            ):
                self.player.take_damage(config.ENEMY_DAMAGE)
                enemy.take_damage(enemy.max_health)
                reward += damage_penalty
                if not enemy.alive:
                    self.enemies_destroyed += 1
                    self.enemies_destroyed_this_step += 1

        return reward

    def _calculate_shaping_reward(self):
        """Combat efficiency reward shaping."""
        if config.SHAPING_MODE == "off":
            return 0.0

        if self._prev_spawner_total_health is None:
            self._prev_spawner_total_health = sum(
                s.health for s in self.spawners if s.alive
            )
            self._prev_enemy_count = len([e for e in self.enemies if e.alive])
            self._prev_player_health = self.player.health
            return 0.0

        current_spawner_health = sum(s.health for s in self.spawners if s.alive)
        spawner_damage = max(
            0, self._prev_spawner_total_health - current_spawner_health
        )
        enemy_kills_this_step = self.enemies_destroyed_this_step
        offensive_score = spawner_damage + (
            enemy_kills_this_step * config.ENEMY_HEALTH * 0.5
        )
        player_damage_taken = max(0, self._prev_player_health - self.player.health)
        health_ratio = self.player.get_health_ratio()
        efficiency = (offensive_score * (0.5 + 0.5 * health_ratio)) - (
            player_damage_taken * 0.8
        )

        self._prev_spawner_total_health = current_spawner_health
        self._prev_enemy_count = len([e for e in self.enemies if e.alive])
        self._prev_player_health = self.player.health

        shaping_scale = config.SHAPING_SCALE
        if self.curriculum_stage:
            shaping_scale *= self.curriculum_stage.shaping_scale_mult

        reward = efficiency * shaping_scale * 0.01
        return float(np.clip(reward, -config.SHAPING_CLIP, config.SHAPING_CLIP))

    def _get_info(self):
        return {
            "phase": self.current_phase,
            "enemies_destroyed": self.enemies_destroyed_this_step,
            "spawners_destroyed": self.spawners_destroyed_this_step,
            "total_enemies_destroyed": self.enemies_destroyed,
            "total_spawners_destroyed": self.spawners_destroyed,
            "player_health": self.player.health,
            "episode_reward": self.episode_reward,
            "episode_steps": self.current_step,
            "win": bool(self.win),
            "win_step": self.win_step or -1,
            "first_spawner_kill_step": self.first_spawner_kill_step or -1,
        }
