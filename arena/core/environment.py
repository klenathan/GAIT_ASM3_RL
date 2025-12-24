"""
Deep RL Arena - Gym Environment.
Main environment implementing the Gym API with dual control schemes.
"""

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

class ArenaEnv(gym.Env):
    """
    Pygame-based Deep RL Arena Environment.
    
    Supports two control schemes:
    - Style 1: Rotation + Thrust (5 actions)
    - Style 2: Directional Movement (6 actions)
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": config.FPS}
    
    def __init__(self, control_style=1, render_mode=None, curriculum_manager: CurriculumManager = None):
        super().__init__()
        
        self.control_style = control_style
        self.render_mode = render_mode
        self.curriculum_manager = curriculum_manager
        
        if control_style == 1:
            self.action_space = spaces.Discrete(config.ACTION_SPACE_STYLE_1)
        else:
            self.action_space = spaces.Discrete(config.ACTION_SPACE_STYLE_2)
        
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(config.OBS_DIM,), dtype=np.float32
        )
        
        self.renderer = None
        self._owns_renderer = False
        if render_mode == "human":
            self.renderer = ArenaRenderer()
            self._owns_renderer = True
        
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
        
        self.previous_spawner_distance = None
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
        
        if self.control_style == 1:
            self.player.update_style_1(action)
        else:
            self.player.update_style_2(action)
        
        if ((self.control_style == 1 and action == 4) or 
            (self.control_style == 2 and action == 5)):
            if self.player.shoot():
                reward += float(config.REWARD_SHOT_FIRED)
                proj = Projectile(self.player.pos[0], self.player.pos[1], 
                                self.player.rotation, is_player_projectile=True)
                self.projectiles.append(proj)
        
        for enemy in self.enemies:
            if enemy.alive:
                enemy.update(self.player.pos)
                if enemy.shoot(self.player.pos, self.np_random):
                    angle = utils.angle_to_point(enemy.pos, self.player.pos)
                    self.projectiles.append(Projectile(enemy.pos[0], enemy.pos[1], angle, False))
        
        phase_cfg = config.PHASE_CONFIG[self.current_phase]
        
        # Apply curriculum modifiers
        max_enemies = config.SPAWNER_MAX_ENEMIES
        enemy_speed = phase_cfg['enemy_speed_mult']
        if self.curriculum_stage:
            max_enemies = int(max_enemies * self.curriculum_stage.max_enemies_mult)
            enemy_speed *= self.curriculum_stage.enemy_speed_mult
        
        for spawner in self.spawners:
            if spawner.alive:
                spawner.update()
                if spawner.can_spawn(len([e for e in self.enemies if e.alive]), max_enemies):
                    new_enemy = spawner.spawn_enemy(self.np_random, enemy_speed)
                    if new_enemy: self.enemies.append(new_enemy)
        
        for proj in self.projectiles:
            if proj.alive: proj.update()
        
        reward += self._handle_collisions()
        reward += float(config.REWARD_STEP_SURVIVAL)
        reward += self._calculate_shaping_reward()
        
        self.enemies = [e for e in self.enemies if e.alive]
        self.spawners = [s for s in self.spawners if s.alive]
        self.projectiles = [p for p in self.projectiles if p.alive]
        
        if len(self.spawners) == 0:
            reward += config.REWARD_PHASE_COMPLETE
            self.current_phase += 1
            if self.current_phase < config.MAX_PHASES:
                self._init_phase()
            else:
                self.win = True
                self.win_step = self.current_step
                done = True
        
        if not self.player.alive:
            reward += config.REWARD_DEATH
            done = True
        
        if self.current_step >= config.MAX_STEPS:
            done = True
        
        self.episode_reward += reward
        truncated = False
        return self._get_observation(), reward, done, truncated, self._get_info()
    
    def render(self):
        if self.render_mode is None: return
        if self.renderer is None and self.render_mode == "human":
            self.renderer = ArenaRenderer()
        
        if self.renderer:
            metrics = {
                'episode': 0,
                'episode_reward': self.episode_reward,
                'total_reward': self.episode_reward,
                'timesteps': self.current_step,
            }
            self.renderer.render(self, metrics)
        
        try: pygame.event.pump()
        except: pass
    
    def close(self):
        if self.renderer and self._owns_renderer:
            self.renderer.close()
        self.renderer = None
        self._owns_renderer = False
    
    def _init_phase(self):
        phase_cfg = config.PHASE_CONFIG[self.current_phase]
        num = phase_cfg['spawners']
        self.enemies = []
        base_angle = self.np_random.uniform(0, 2 * math.pi)
        
        for i in range(num):
            angle = (2 * math.pi / num) * i + base_angle + self.np_random.uniform(-0.2, 0.2)
            dist = min(config.GAME_WIDTH, config.GAME_HEIGHT) * 0.4 * self.np_random.uniform(0.8, 1.2)
            x = config.GAME_WIDTH / 2 + math.cos(angle) * dist
            y = config.GAME_HEIGHT / 2 + math.sin(angle) * dist
            x = utils.clamp(x, config.SPAWNER_RADIUS, config.GAME_WIDTH - config.SPAWNER_RADIUS)
            y = utils.clamp(y, config.SPAWNER_RADIUS, config.GAME_HEIGHT - config.SPAWNER_RADIUS)
            
            # Apply curriculum modifiers to spawner
            spawn_rate_mult = phase_cfg['spawn_rate_mult']
            health_mult = 1.0
            if self.curriculum_stage:
                spawn_rate_mult *= self.curriculum_stage.spawn_cooldown_mult
                health_mult = self.curriculum_stage.spawner_health_mult
            
            spawner = Spawner(x, y, spawn_rate_mult)
            if health_mult != 1.0:
                spawner.health = int(spawner.max_health * health_mult)
            self.spawners.append(spawner)
        
        self.phase_start_step = self.current_step
        self.previous_spawner_distance = None
    
    def _get_observation(self):
        """Build expanded observation vector (32 dims)."""
        obs = np.zeros(config.OBS_DIM, dtype=np.float32)
        max_dist = math.sqrt(config.GAME_WIDTH**2 + config.GAME_HEIGHT**2)
        
        # [0-1] Player position
        obs[0] = self.player.pos[0] / config.GAME_WIDTH
        obs[1] = self.player.pos[1] / config.GAME_HEIGHT
        
        # [2-3] Player velocity
        obs[2] = np.clip(self.player.velocity[0] / config.PLAYER_MAX_VELOCITY, -1, 1)
        obs[3] = np.clip(self.player.velocity[1] / config.PLAYER_MAX_VELOCITY, -1, 1)
        
        # [4] Player rotation
        obs[4] = self.player.rotation / (2 * math.pi)
        
        # [5] Player health ratio
        obs[5] = self.player.get_health_ratio()
        
        # [6] Shoot cooldown ratio (0 = ready to shoot)
        obs[6] = self.player.shoot_cooldown / config.PLAYER_SHOOT_COOLDOWN
        
        # [7] Current phase ratio
        obs[7] = self.current_phase / config.MAX_PHASES
        
        # [8] Spawners remaining ratio (for current phase objectives)
        # Clamp phase index to valid range (handles edge case when game ends after final phase)
        phase_idx = min(self.current_phase, config.MAX_PHASES - 1)
        initial_spawners = config.PHASE_CONFIG[phase_idx]['spawners']
        obs[8] = len([s for s in self.spawners if s.alive]) / max(initial_spawners, 1)
        
        # [9] Time remaining ratio
        obs[9] = 1.0 - (self.current_step / config.MAX_STEPS)
        
        # [10-15] Nearest 2 enemies (dist, angle, exists) x2
        nearest_enemies = self._find_k_nearest_entities(self.enemies, k=2)
        for i, enemy in enumerate(nearest_enemies):
            base_idx = 10 + i * 3
            if enemy:
                obs[base_idx] = utils.distance(self.player.pos, enemy.pos) / max_dist
                obs[base_idx + 1] = utils.normalize_angle(
                    utils.relative_angle(self.player.rotation, 
                                        utils.angle_to_point(self.player.pos, enemy.pos)))
                obs[base_idx + 2] = 1.0
            else:
                obs[base_idx], obs[base_idx + 1], obs[base_idx + 2] = 1.0, 0.5, 0.0
        
        # [16-23] Nearest 2 spawners (dist, angle, exists, health) x2
        nearest_spawners = self._find_k_nearest_entities(self.spawners, k=2)
        for i, spawner in enumerate(nearest_spawners):
            base_idx = 16 + i * 4
            if spawner:
                obs[base_idx] = utils.distance(self.player.pos, spawner.pos) / max_dist
                obs[base_idx + 1] = utils.normalize_angle(
                    utils.relative_angle(self.player.rotation,
                                        utils.angle_to_point(self.player.pos, spawner.pos)))
                obs[base_idx + 2] = 1.0
                obs[base_idx + 3] = spawner.health / spawner.max_health
            else:
                obs[base_idx], obs[base_idx + 1], obs[base_idx + 2], obs[base_idx + 3] = 1.0, 0.5, 0.0, 0.0
        
        # [24-26] Projectile threat info (nearest dist, angle, count nearby)
        proj_dist, proj_angle, proj_count = self._get_projectile_threat_info()
        obs[24] = proj_dist / max_dist
        obs[25] = proj_angle
        obs[26] = min(proj_count / 5.0, 1.0)  # Normalize, cap at 5 projectiles
        
        # [27-30] Wall distances (left, right, top, bottom) - normalized
        obs[27] = self.player.pos[0] / config.GAME_WIDTH  # Distance from left
        obs[28] = 1.0 - (self.player.pos[0] / config.GAME_WIDTH)  # Distance from right
        obs[29] = self.player.pos[1] / config.GAME_HEIGHT  # Distance from top
        obs[30] = 1.0 - (self.player.pos[1] / config.GAME_HEIGHT)  # Distance from bottom
        
        # [31] Enemy count
        obs[31] = len([e for e in self.enemies if e.alive]) / config.SPAWNER_MAX_ENEMIES
        
        return obs
    
    def _find_k_nearest_entities(self, entities, k=2):
        """Find k nearest entities, returns list padded with None if fewer exist."""
        alive_entities = [(e, utils.distance(self.player.pos, e.pos)) 
                          for e in entities if e.alive]
        alive_entities.sort(key=lambda x: x[1])
        
        result = [e for e, _ in alive_entities[:k]]
        # Pad with None if fewer than k entities
        while len(result) < k:
            result.append(None)
        return result
    
    def _get_projectile_threat_info(self):
        """Get info about threatening projectiles (enemy projectiles only)."""
        max_dist = math.sqrt(config.GAME_WIDTH**2 + config.GAME_HEIGHT**2)
        enemy_projectiles = [p for p in self.projectiles 
                            if p.alive and not p.is_player_projectile]
        
        if not enemy_projectiles:
            return max_dist, 0.5, 0  # Return max_dist, not inf
        
        # Find nearest
        min_dist = float('inf')
        nearest_angle = 0.5
        for proj in enemy_projectiles:
            dist = utils.distance(self.player.pos, proj.pos)
            if dist < min_dist:
                min_dist = dist
                nearest_angle = utils.normalize_angle(
                    utils.relative_angle(self.player.rotation,
                                        utils.angle_to_point(self.player.pos, proj.pos)))
        
        # Count projectiles within danger radius
        danger_count = sum(1 for p in enemy_projectiles 
                          if utils.distance(self.player.pos, p.pos) < config.PROJECTILE_DANGER_RADIUS)
        
        return min_dist, nearest_angle, danger_count
    
    def _find_nearest_entity(self, entities):
        """Find single nearest entity (kept for backward compatibility)."""
        nearest, min_dist = None, float('inf')
        for e in entities:
            if e.alive:
                d = utils.distance(self.player.pos, e.pos)
                if d < min_dist: min_dist, nearest = d, e
        return nearest
    
    def _handle_collisions(self):
        reward = 0.0
        
        # Calculate damage penalty with curriculum scaling
        damage_penalty = config.REWARD_DAMAGE_TAKEN
        if self.curriculum_stage:
            damage_penalty *= self.curriculum_stage.damage_penalty_mult
        
        for proj in self.projectiles:
            if not proj.alive or not proj.is_player_projectile: continue
            for enemy in self.enemies:
                if enemy.alive and utils.check_collision(proj.pos, proj.radius, enemy.pos, enemy.radius):
                    enemy.take_damage(proj.damage); proj.hit()
                    reward += float(config.REWARD_HIT_ENEMY)
                    if not enemy.alive:
                        reward += config.REWARD_ENEMY_DESTROYED
                        self.enemies_destroyed += 1
                        self.enemies_destroyed_this_step += 1
                    break
            if not proj.alive: continue
            for spawner in self.spawners:
                if spawner.alive and utils.check_collision(proj.pos, proj.radius, spawner.pos, spawner.radius):
                    spawner.take_damage(proj.damage); proj.hit()
                    reward += float(config.REWARD_HIT_SPAWNER)
                    if not spawner.alive:
                        reward += config.REWARD_SPAWNER_DESTROYED
                        self.spawners_destroyed += 1
                        self.spawners_destroyed_this_step += 1
                        if self.first_spawner_kill_step is None: self.first_spawner_kill_step = self.current_step
                        if (self.current_step - self.phase_start_step) < 500: reward += config.REWARD_QUICK_SPAWNER_KILL
                    break
        
        # Apply curriculum-scaled damage penalty for projectile hits
        for proj in self.projectiles:
            if not proj.alive or proj.is_player_projectile: continue
            if self.player.alive and utils.check_collision(proj.pos, proj.radius, self.player.pos, self.player.radius):
                self.player.take_damage(proj.damage); proj.hit(); reward += damage_penalty
        
        # Apply curriculum-scaled damage penalty for enemy collisions
        for enemy in self.enemies:
            if enemy.alive and self.player.alive and utils.check_collision(enemy.pos, enemy.radius, self.player.pos, self.player.radius):
                self.player.take_damage(config.ENEMY_DAMAGE); enemy.take_damage(enemy.max_health); reward += damage_penalty
                if not enemy.alive:
                    self.enemies_destroyed += 1
                    self.enemies_destroyed_this_step += 1
        return reward
    
    def _calculate_shaping_reward(self):
        if config.SHAPING_MODE == "off" or not self.spawners: return 0.0
        near_s = self._find_nearest_entity(self.spawners)
        if not near_s: return 0.0
        dist = utils.distance(self.player.pos, near_s.pos)
        if self.previous_spawner_distance is None:
            self.previous_spawner_distance = dist; return 0.0
        prev = self.previous_spawner_distance
        self.previous_spawner_distance = dist
        
        # Get shaping scale with curriculum modifier
        shaping_scale = config.SHAPING_SCALE
        if self.curriculum_stage:
            shaping_scale *= self.curriculum_stage.shaping_scale_mult
        
        if config.SHAPING_MODE == "binary":
            return float(shaping_scale) if dist < prev else 0.0
        max_dist = math.sqrt(config.GAME_WIDTH**2 + config.GAME_HEIGHT**2)
        delta = (prev - dist) / max_dist
        return float(np.clip(shaping_scale * delta, -config.SHAPING_CLIP, config.SHAPING_CLIP))
    
    def _get_info(self):
        return {
            'phase': self.current_phase,
            'enemies_destroyed': self.enemies_destroyed_this_step,  # Incremental this step
            'spawners_destroyed': self.spawners_destroyed_this_step,  # Incremental this step
            'total_enemies_destroyed': self.enemies_destroyed,  # Cumulative episode total
            'total_spawners_destroyed': self.spawners_destroyed,  # Cumulative episode total
            'player_health': self.player.health,
            'episode_reward': self.episode_reward,
            'episode_steps': self.current_step,
            'win': bool(self.win),
            'win_step': self.win_step or -1,
            'first_spawner_kill_step': self.first_spawner_kill_step or -1,
        }
