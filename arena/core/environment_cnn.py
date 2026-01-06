"""
CNN Environment for Deep RL Arena.
Uses multi-channel heatmap observations for CNN-based policies.
"""

import warnings

warnings.filterwarnings(
    "ignore", message=".*pkg_resources is deprecated.*", category=UserWarning
)

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math

from arena.core import config
from arena.core.curriculum import CurriculumManager
from arena.game import utils
from arena.game.entities import Player, Enemy, Spawner, Projectile
from arena.ui.renderer import ArenaRenderer


# Heatmap configuration
HEATMAP_SIZE = 64  # 64x64 resolution
NUM_CHANNELS = 5  # player, enemies, spawners, projectiles, walls
SCALAR_DIM = 7  # auxiliary scalar features
GAUSSIAN_SIGMA = 3.0  # Sigma for entity blobs


class ArenaCNNEnv(gym.Env):
    """
    CNN-observation variant of ArenaEnv.

    Uses multi-channel heatmap images combined with scalar features:

    Image Channels (5, 64, 64):
    - Channel 0: Player position (Gaussian blob)
    - Channel 1: Enemy positions (Gaussian blobs, intensity = threat)
    - Channel 2: Spawner positions (Gaussian blobs, intensity = health ratio)
    - Channel 3: Enemy projectiles (Gaussian blobs, intensity = proximity danger)
    - Channel 4: Wall proximity (distance field from boundaries)

    Scalar Features (7):
    - Player health ratio
    - Shoot cooldown ratio
    - Phase progress ratio
    - Time remaining ratio
    - Enemy count (normalized)
    - Spawners remaining ratio
    - Player velocity magnitude (normalized)

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

        # Action space
        if control_style == 1:
            self.action_space = spaces.Discrete(config.ACTION_SPACE_STYLE_1)
        else:
            self.action_space = spaces.Discrete(config.ACTION_SPACE_STYLE_2)

        # Dict observation space with image + scalars
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(NUM_CHANNELS, HEATMAP_SIZE, HEATMAP_SIZE),
                    dtype=np.float32,
                ),
                "scalars": spaces.Box(
                    low=-1.0, high=1.0, shape=(SCALAR_DIM,), dtype=np.float32
                ),
            }
        )

        # Pre-compute coordinate grids for efficient Gaussian rendering
        self._init_gaussian_cache()

        # Pre-compute wall distance field (static)
        self._wall_field = self._compute_wall_field()

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

        # Reward shaping state
        self._prev_spawner_total_health = None
        self._prev_enemy_count = None
        self._prev_player_health = None
        self.phase_start_step = 0

    def _init_gaussian_cache(self):
        """Pre-compute coordinate grids for Gaussian blob rendering."""
        y_coords = np.arange(HEATMAP_SIZE)
        x_coords = np.arange(HEATMAP_SIZE)
        self._yy, self._xx = np.meshgrid(y_coords, x_coords, indexing="ij")

    def _compute_wall_field(self):
        """Compute static wall proximity field."""
        field = np.zeros((HEATMAP_SIZE, HEATMAP_SIZE), dtype=np.float32)

        for y in range(HEATMAP_SIZE):
            for x in range(HEATMAP_SIZE):
                # Distance to nearest edge (normalized)
                dist_left = x / HEATMAP_SIZE
                dist_right = (HEATMAP_SIZE - 1 - x) / HEATMAP_SIZE
                dist_top = y / HEATMAP_SIZE
                dist_bottom = (HEATMAP_SIZE - 1 - y) / HEATMAP_SIZE

                min_dist = min(dist_left, dist_right, dist_top, dist_bottom)
                # Invert so edges have high values
                field[y, x] = 1.0 - min(
                    min_dist * 4.0, 1.0
                )  # Steep gradient near edges

        return field

    def _world_to_heatmap(self, world_x, world_y):
        """Convert world coordinates to heatmap coordinates."""
        hx = int((world_x / config.GAME_WIDTH) * (HEATMAP_SIZE - 1))
        hy = int((world_y / config.GAME_HEIGHT) * (HEATMAP_SIZE - 1))
        return np.clip(hx, 0, HEATMAP_SIZE - 1), np.clip(hy, 0, HEATMAP_SIZE - 1)

    def _add_gaussian_blob(
        self, channel, world_x, world_y, intensity=1.0, sigma=GAUSSIAN_SIGMA
    ):
        """Add a Gaussian blob to a channel at the given world position."""
        hx, hy = self._world_to_heatmap(world_x, world_y)

        # Compute Gaussian using pre-computed grids
        dist_sq = (self._xx - hx) ** 2 + (self._yy - hy) ** 2
        blob = intensity * np.exp(-dist_sq / (2 * sigma**2))

        # Additive blending with clipping
        np.maximum(channel, blob, out=channel)

    def _render_heatmap(self):
        """Generate multi-channel heatmap observation."""
        heatmap = np.zeros((NUM_CHANNELS, HEATMAP_SIZE, HEATMAP_SIZE), dtype=np.float32)

        # Channel 0: Player position
        if self.player and self.player.alive:
            self._add_gaussian_blob(
                heatmap[0],
                self.player.pos[0],
                self.player.pos[1],
                intensity=1.0,
                sigma=GAUSSIAN_SIGMA,
            )

        # Channel 1: Enemy positions (intensity = threat based on proximity)
        max_dist = math.sqrt(config.GAME_WIDTH**2 + config.GAME_HEIGHT**2)
        for enemy in self.enemies:
            if enemy.alive:
                dist = (
                    utils.distance(self.player.pos, enemy.pos)
                    if self.player
                    else max_dist
                )
                threat = 1.0 - (dist / max_dist)  # Closer = higher threat
                self._add_gaussian_blob(
                    heatmap[1],
                    enemy.pos[0],
                    enemy.pos[1],
                    intensity=max(0.3, threat),
                    sigma=GAUSSIAN_SIGMA * 1.2,
                )

        # Channel 2: Spawner positions (intensity = health ratio)
        for spawner in self.spawners:
            if spawner.alive:
                health_ratio = spawner.health / spawner.max_health
                self._add_gaussian_blob(
                    heatmap[2],
                    spawner.pos[0],
                    spawner.pos[1],
                    intensity=health_ratio,
                    sigma=GAUSSIAN_SIGMA * 1.5,
                )

        # Channel 3: Enemy projectiles (intensity = proximity danger)
        for proj in self.projectiles:
            if proj.alive and not proj.is_player_projectile:
                if self.player:
                    dist = utils.distance(self.player.pos, proj.pos)
                    danger = max(0, 1.0 - (dist / config.PROJECTILE_DANGER_RADIUS))
                else:
                    danger = 0.5
                self._add_gaussian_blob(
                    heatmap[3],
                    proj.pos[0],
                    proj.pos[1],
                    intensity=max(0.5, danger),
                    sigma=GAUSSIAN_SIGMA * 0.8,
                )

        # Channel 4: Wall proximity (static field)
        heatmap[4] = self._wall_field.copy()

        return heatmap

    def _get_scalar_features(self):
        """Get auxiliary scalar features with agent-relative coordinates."""
        scalars = np.zeros(SCALAR_DIM, dtype=np.float32)
        max_dist = math.sqrt(config.GAME_WIDTH**2 + config.GAME_HEIGHT**2)

        if self.player:
            # Arena center
            center_x = config.GAME_WIDTH / 2
            center_y = config.GAME_HEIGHT / 2

            # [0] Distance from center (normalized)
            dist_from_center = math.sqrt(
                (self.player.pos[0] - center_x) ** 2
                + (self.player.pos[1] - center_y) ** 2
            )
            scalars[0] = dist_from_center / max_dist

            # [1] Velocity magnitude (speed, normalized)
            velocity_magnitude = math.sqrt(
                self.player.velocity[0] ** 2 + self.player.velocity[1] ** 2
            )
            scalars[1] = np.clip(velocity_magnitude / config.PLAYER_MAX_VELOCITY, 0, 1)

            # [2] Player health ratio
            scalars[2] = self.player.get_health_ratio()

            # [3] Shoot cooldown ratio
            scalars[3] = self.player.shoot_cooldown / config.PLAYER_SHOOT_COOLDOWN

            # [4] Phase progress ratio
            scalars[4] = self.current_phase / config.MAX_PHASES

            # [5] Time remaining ratio
            scalars[5] = 1.0 - (self.current_step / config.MAX_STEPS)

            # [6] Enemy count (normalized)
            scalars[6] = (
                len([e for e in self.enemies if e.alive]) / config.SPAWNER_MAX_ENEMIES
            )

        return scalars

    def _get_observation(self):
        """Build observation dict with heatmap image and scalar features."""
        return {
            "image": self._render_heatmap(),
            "scalars": self._get_scalar_features(),
        }

    @property
    def curriculum_stage(self):
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

        self.enemies_destroyed_this_step = 0
        self.spawners_destroyed_this_step = 0

        # Update player
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

        # Check death
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
                'episode': 0,
                'episode_reward': self.episode_reward,
                'total_reward': self.episode_reward,
                'timesteps': self.current_step,
                'show_inputs': True,  # Show model inputs during rendering
            }
            self.renderer.render(self, metrics)

        try:
            import pygame

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
            dist_to_player = math.sqrt((x - player_spawn[0])**2 + (y - player_spawn[1])**2)
            if dist_to_player < min_dist_from_player:
                continue
            
            # Check distance from existing spawners
            too_close = False
            for spawner in self.spawners:
                dist = math.sqrt((x - spawner.pos[0])**2 + (y - spawner.pos[1])**2)
                if dist < min_dist_between_spawners:
                    too_close = True
                    break
            
            if not too_close:
                return x, y
        
        # Fallback: return random position if no valid spot found
        return self.np_random.uniform(margin, w - margin), self.np_random.uniform(margin, h - margin)
    
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
