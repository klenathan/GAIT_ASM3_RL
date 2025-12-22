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
    
    def __init__(self, control_style=1, render_mode=None):
        super().__init__()
        
        self.control_style = control_style
        self.render_mode = render_mode
        
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
        for spawner in self.spawners:
            if spawner.alive:
                spawner.update()
                if spawner.can_spawn(len([e for e in self.enemies if e.alive])):
                    new_enemy = spawner.spawn_enemy(self.np_random, phase_cfg['enemy_speed_mult'])
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
            self.spawners.append(Spawner(x, y, phase_cfg['spawn_rate_mult']))
        
        self.phase_start_step = self.current_step
        self.previous_spawner_distance = None
    
    def _get_observation(self):
        obs = np.zeros(config.OBS_DIM, dtype=np.float32)
        obs[0] = self.player.pos[0] / config.GAME_WIDTH
        obs[1] = self.player.pos[1] / config.GAME_HEIGHT
        obs[2] = np.clip(self.player.velocity[0] / config.PLAYER_MAX_VELOCITY, -1, 1)
        obs[3] = np.clip(self.player.velocity[1] / config.PLAYER_MAX_VELOCITY, -1, 1)
        obs[4] = self.player.rotation / (2 * math.pi)
        obs[5] = self.player.get_health_ratio()
        obs[6] = self.current_phase / config.MAX_PHASES
        
        near_e = self._find_nearest_entity(self.enemies)
        max_dist = math.sqrt(config.GAME_WIDTH**2 + config.GAME_HEIGHT**2)
        if near_e:
            obs[7] = utils.distance(self.player.pos, near_e.pos) / max_dist
            obs[8] = utils.normalize_angle(utils.relative_angle(self.player.rotation, utils.angle_to_point(self.player.pos, near_e.pos)))
            obs[9] = 1.0
        else:
            obs[7], obs[8], obs[9] = 1.0, 0.5, 0.0
            
        near_s = self._find_nearest_entity(self.spawners)
        if near_s:
            obs[10] = utils.distance(self.player.pos, near_s.pos) / max_dist
            obs[11] = utils.normalize_angle(utils.relative_angle(self.player.rotation, utils.angle_to_point(self.player.pos, near_s.pos)))
            obs[12] = 1.0
        else:
            obs[10], obs[11], obs[12] = 1.0, 0.5, 0.0
            
        obs[13] = len(self.enemies) / config.SPAWNER_MAX_ENEMIES
        return obs
    
    def _find_nearest_entity(self, entities):
        nearest, min_dist = None, float('inf')
        for e in entities:
            if e.alive:
                d = utils.distance(self.player.pos, e.pos)
                if d < min_dist: min_dist, nearest = d, e
        return nearest
    
    def _handle_collisions(self):
        reward = 0.0
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
        
        for proj in self.projectiles:
            if not proj.alive or proj.is_player_projectile: continue
            if self.player.alive and utils.check_collision(proj.pos, proj.radius, self.player.pos, self.player.radius):
                self.player.take_damage(proj.damage); proj.hit(); reward += config.REWARD_DAMAGE_TAKEN
        
        for enemy in self.enemies:
            if enemy.alive and self.player.alive and utils.check_collision(enemy.pos, enemy.radius, self.player.pos, self.player.radius):
                self.player.take_damage(config.ENEMY_DAMAGE); enemy.take_damage(enemy.max_health); reward += config.REWARD_DAMAGE_TAKEN
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
        if config.SHAPING_MODE == "binary":
            return float(config.SHAPING_SCALE) if dist < prev else 0.0
        max_dist = math.sqrt(config.GAME_WIDTH**2 + config.GAME_HEIGHT**2)
        delta = (prev - dist) / max_dist
        return float(np.clip(config.SHAPING_SCALE * delta, -config.SHAPING_CLIP, config.SHAPING_CLIP))
    
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
