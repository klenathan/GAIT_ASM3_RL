"""
Deep RL Arena - Gym Environment
Main environment implementing the Gym API with dual control schemes
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import pygame

from arena import config
from arena import utils
from arena.entities import Player, Enemy, Spawner, Projectile
from arena.renderer import ArenaRenderer


class ArenaEnv(gym.Env):
    """
    Pygame-based Deep RL Arena Environment
    
    Supports two control schemes:
    - Style 1: Rotation + Thrust (5 actions)
    - Style 2: Directional Movement (6 actions)
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": config.FPS}
    
    def __init__(self, control_style=1, render_mode=None):
        """
        Initialize environment
        
        Args:
            control_style: 1 for rotation/thrust, 2 for directional
            render_mode: "human" for display, "rgb_array" for recording
        """
        super().__init__()
        
        self.control_style = control_style
        self.render_mode = render_mode
        
        # Define action space based on control style
        if control_style == 1:
            self.action_space = spaces.Discrete(config.ACTION_SPACE_STYLE_1)
        else:
            self.action_space = spaces.Discrete(config.ACTION_SPACE_STYLE_2)
        
        # Define observation space (14-dimensional)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(config.OBS_DIM,),
            dtype=np.float32
        )
        
        # Initialize renderer if needed
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
        
        # Episode tracking
        self.current_step = 0
        self.current_phase = 0
        self.enemies_destroyed = 0
        self.spawners_destroyed = 0
        self.episode_reward = 0
        self.win = False
        self.win_step = None
        self.first_spawner_kill_step = None
        
        # For reward shaping
        self.previous_spawner_distance = None
        self.phase_start_step = 0
        self.reward_step_survival = float(config.REWARD_STEP_SURVIVAL)
        self.shaping_mode = str(config.SHAPING_MODE)
        self.shaping_scale = float(config.SHAPING_SCALE)
        self.shaping_clip = float(config.SHAPING_CLIP)
        
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # Reset game state
        self.current_step = 0
        self.current_phase = 0
        self.enemies_destroyed = 0
        self.spawners_destroyed = 0
        self.episode_reward = 0
        self.phase_start_step = 0
        self.win = False
        self.win_step = None
        self.first_spawner_kill_step = None
        
        # Create player at center
        self.player = Player(config.GAME_WIDTH / 2, config.GAME_HEIGHT / 2)
        
        # Clear entities
        self.enemies = []
        self.spawners = []
        self.projectiles = []
        
        # Initialize first phase
        self._init_phase()
        
        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action):
        """Execute one step in the environment"""
        self.current_step += 1
        reward = 0.0
        done = False
        
        # Update player based on control style
        if self.control_style == 1:
            self.player.update_style_1(action)
        else:
            self.player.update_style_2(action)
        
        # Handle shooting
        if ((self.control_style == 1 and action == 4) or 
            (self.control_style == 2 and action == 5)):
            if self.player.shoot():
                # Optional reward/penalty for actually firing a shot (configurable)
                reward += float(getattr(config, "REWARD_SHOT_FIRED", 0.0))
                # Create projectile
                proj = Projectile(
                    self.player.pos[0], 
                    self.player.pos[1], 
                    self.player.rotation,
                    is_player_projectile=True
                )
                self.projectiles.append(proj)
        
        # Update enemies
        for enemy in self.enemies:
            if enemy.alive:
                enemy.update(self.player.pos)
                
                # Enemy shooting
                if enemy.shoot(self.player.pos):
                    angle = utils.angle_to_point(enemy.pos, self.player.pos)
                    proj = Projectile(
                        enemy.pos[0],
                        enemy.pos[1],
                        angle,
                        is_player_projectile=False
                    )
                    self.projectiles.append(proj)
        
        # Update spawners
        phase_cfg = config.PHASE_CONFIG[self.current_phase]
        for spawner in self.spawners:
            if spawner.alive:
                spawner.update()
                
                # Spawn enemies
                if spawner.can_spawn(len([e for e in self.enemies if e.alive])):
                    new_enemy = spawner.spawn_enemy(phase_cfg['enemy_speed_mult'])
                    if new_enemy:
                        self.enemies.append(new_enemy)
        
        # Update projectiles
        for proj in self.projectiles:
            if proj.alive:
                proj.update()
        
        # Handle collisions and calculate rewards
        reward += self._handle_collisions()
        
        # Survival reward
        reward += self.reward_step_survival
        
        # Reward shaping: encourage approaching spawners
        reward += self._calculate_shaping_reward()
        
        # Remove dead entities
        self.enemies = [e for e in self.enemies if e.alive]
        self.spawners = [s for s in self.spawners if s.alive]
        self.projectiles = [p for p in self.projectiles if p.alive]
        
        # Check for phase completion
        if len(self.spawners) == 0:
            reward += config.REWARD_PHASE_COMPLETE
            self.current_phase += 1
            
            if self.current_phase < config.MAX_PHASES:
                self._init_phase()
            else:
                # Game won!
                self.win = True
                self.win_step = self.current_step
                done = True
        
        # Check for death
        if not self.player.alive:
            reward += config.REWARD_DEATH
            done = True
        
        # Check for max steps
        if self.current_step >= config.MAX_STEPS:
            done = True
        
        self.episode_reward += reward
        
        # Get observation and info
        obs = self._get_observation()
        info = self._get_info()
        
        truncated = False
        
        return obs, reward, done, truncated, info
    
    def render(self):
        """Render the environment"""
        # Skip all rendering logic if render_mode is None
        if self.render_mode is None:
            return
        
        if self.renderer is None:
            if self.render_mode == "human":
                self.renderer = ArenaRenderer()
        
        if self.renderer:
            metrics = {
                'episode': 0,  # Will be updated by training script
                'episode_reward': self.episode_reward,
                'total_reward': self.episode_reward,
                'timesteps': self.current_step,
            }
            self.renderer.render(self, metrics)
        
        # Keep the OS window responsive without consuming events.
        # (Interactive evaluation loops handle QUIT/keys themselves.)
        try:
            pygame.event.pump()
        except pygame.error:
            pass
    
    def close(self):
        """Clean up resources"""
        # Only close pygame if this env created/owns the renderer.
        # In interactive evaluation we often inject a shared renderer.
        if self.renderer and self._owns_renderer:
            self.renderer.close()
        self.renderer = None
        self._owns_renderer = False
    
    def _init_phase(self):
        """Initialize a new phase with spawners"""
        phase_cfg = config.PHASE_CONFIG[self.current_phase]
        num_spawners = phase_cfg['spawners']
        
        # Clear existing enemies from previous phase
        self.enemies = []
        
        # Create spawners in a randomized circular arrangement
        # Randomize the base rotation of the entire circle
        base_angle_offset = self.np_random.uniform(0, 2 * math.pi)
        
        for i in range(num_spawners):
            # Base angle for this spawner
            angle = (2 * math.pi / num_spawners) * i + base_angle_offset
            
            # Add small random perturbation to angle (+/- 15 degrees)
            angle += self.np_random.uniform(-math.pi/12, math.pi/12)
            
            # Place spawners away from center with some distance variation
            # Base distance is 40% of min dimension, vary by +/- 10%
            base_distance = min(config.GAME_WIDTH, config.GAME_HEIGHT) * 0.4
            distance = base_distance * self.np_random.uniform(0.8, 1.2)
            
            x = config.GAME_WIDTH / 2 + math.cos(angle) * distance
            y = config.GAME_HEIGHT / 2 + math.sin(angle) * distance
            
            # Keep spawners within game area boundaries
            x = utils.clamp(x, config.SPAWNER_RADIUS, config.GAME_WIDTH - config.SPAWNER_RADIUS)
            y = utils.clamp(y, config.SPAWNER_RADIUS, config.GAME_HEIGHT - config.SPAWNER_RADIUS)
            
            spawner = Spawner(x, y, phase_cfg['spawn_rate_mult'])
            self.spawners.append(spawner)
        
        self.phase_start_step = self.current_step
        self.previous_spawner_distance = None
    
    def _get_observation(self):
        """
        Get current observation vector (14 dimensions)
        
        Returns:
            numpy array with normalized features
        """
        obs = np.zeros(config.OBS_DIM, dtype=np.float32)
        
        # Player position (normalized 0-1)
        obs[0] = self.player.pos[0] / config.GAME_WIDTH
        obs[1] = self.player.pos[1] / config.GAME_HEIGHT
        
        # Player velocity (normalized -1 to 1)
        obs[2] = np.clip(self.player.velocity[0] / config.PLAYER_MAX_VELOCITY, -1, 1)
        obs[3] = np.clip(self.player.velocity[1] / config.PLAYER_MAX_VELOCITY, -1, 1)
        
        # Player rotation (normalized 0-1)
        obs[4] = self.player.rotation / (2 * math.pi)
        
        # Player health (normalized 0-1)
        obs[5] = self.player.get_health_ratio()
        
        # Current phase (normalized 0-1)
        obs[6] = self.current_phase / config.MAX_PHASES
        
        # Nearest enemy info
        nearest_enemy = self._find_nearest_entity(self.enemies)
        if nearest_enemy:
            dist = utils.distance(self.player.pos, nearest_enemy.pos)
            angle = utils.angle_to_point(self.player.pos, nearest_enemy.pos)
            rel_angle = utils.relative_angle(self.player.rotation, angle)
            
            max_dist = math.sqrt(config.GAME_WIDTH**2 + config.GAME_HEIGHT**2)
            obs[7] = dist / max_dist  # Distance (normalized 0-1)
            obs[8] = utils.normalize_angle(rel_angle)  # Angle (normalized 0-1)
            obs[9] = 1.0  # Enemy exists
        else:
            obs[7] = 1.0  # Max distance
            obs[8] = 0.5  # Neutral angle
            obs[9] = 0.0  # No enemy
        
        # Nearest spawner info
        nearest_spawner = self._find_nearest_entity(self.spawners)
        if nearest_spawner:
            dist = utils.distance(self.player.pos, nearest_spawner.pos)
            angle = utils.angle_to_point(self.player.pos, nearest_spawner.pos)
            rel_angle = utils.relative_angle(self.player.rotation, angle)
            
            max_dist = math.sqrt(config.GAME_WIDTH**2 + config.GAME_HEIGHT**2)
            obs[10] = dist / max_dist  # Distance (normalized 0-1)
            obs[11] = utils.normalize_angle(rel_angle)  # Angle (normalized 0-1)
            obs[12] = 1.0  # Spawner exists
        else:
            obs[10] = 1.0  # Max distance
            obs[11] = 0.5  # Neutral angle
            obs[12] = 0.0  # No spawner
        
        # Active enemy count (normalized 0-1)
        obs[13] = len(self.enemies) / config.SPAWNER_MAX_ENEMIES
        
        return obs
    
    def _find_nearest_entity(self, entities):
        """Find nearest entity to player"""
        nearest = None
        min_dist = float('inf')
        
        for entity in entities:
            if entity.alive:
                dist = utils.distance(self.player.pos, entity.pos)
                if dist < min_dist:
                    min_dist = dist
                    nearest = entity
        
        return nearest
    
    def _handle_collisions(self):
        """Handle all collision detection and return reward"""
        reward = 0.0
        
        # Player projectiles vs enemies
        for proj in self.projectiles:
            if not proj.alive or not proj.is_player_projectile:
                continue
            
            for enemy in self.enemies:
                if enemy.alive and utils.check_collision(
                    proj.pos, proj.radius, enemy.pos, enemy.radius
                ):
                    enemy.take_damage(proj.damage)
                    proj.hit()
                    # Dense reward for a successful hit (even if not lethal)
                    reward += float(getattr(config, "REWARD_HIT_ENEMY", 0.0))
                    
                    if not enemy.alive:
                        reward += config.REWARD_ENEMY_DESTROYED
                        self.enemies_destroyed += 1
                    break
        
        # Player projectiles vs spawners
        for proj in self.projectiles:
            if not proj.alive or not proj.is_player_projectile:
                continue
            
            for spawner in self.spawners:
                if spawner.alive and utils.check_collision(
                    proj.pos, proj.radius, spawner.pos, spawner.radius
                ):
                    spawner.take_damage(proj.damage)
                    proj.hit()
                    # Dense reward for a successful hit (even if not lethal)
                    reward += float(getattr(config, "REWARD_HIT_SPAWNER", 0.0))
                    
                    if not spawner.alive:
                        reward += config.REWARD_SPAWNER_DESTROYED
                        self.spawners_destroyed += 1
                        if self.first_spawner_kill_step is None:
                            self.first_spawner_kill_step = self.current_step
                        
                        # Bonus for quick kill
                        steps_in_phase = self.current_step - self.phase_start_step
                        if steps_in_phase < 500:
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
                reward += config.REWARD_DAMAGE_TAKEN
        
        # Enemy collision with player
        for enemy in self.enemies:
            if enemy.alive and self.player.alive and utils.check_collision(
                enemy.pos, enemy.radius, self.player.pos, self.player.radius
            ):
                self.player.take_damage(config.ENEMY_DAMAGE)
                enemy.take_damage(enemy.max_health)  # Enemy also dies
                reward += config.REWARD_DAMAGE_TAKEN
                
                if not enemy.alive:
                    self.enemies_destroyed += 1
        
        return reward
    
    def _calculate_shaping_reward(self):
        """
        Reward shaping for approaching spawners.

        Modes:
        - off: no shaping
        - binary: reward a small constant if closer than previous step
        - delta: reward proportional to normalized distance delta (can be negative), clipped
        """
        mode = str(self.shaping_mode).lower().strip()
        if mode == "off":
            return 0.0

        if len(self.spawners) == 0:
            self.previous_spawner_distance = None
            return 0.0

        nearest_spawner = self._find_nearest_entity(self.spawners)
        if nearest_spawner is None:
            self.previous_spawner_distance = None
            return 0.0

        current_dist = utils.distance(self.player.pos, nearest_spawner.pos)

        if self.previous_spawner_distance is None:
            self.previous_spawner_distance = current_dist
            return 0.0

        prev = float(self.previous_spawner_distance)
        self.previous_spawner_distance = current_dist

        if mode == "binary":
            return float(config.REWARD_APPROACH_SPAWNER) if current_dist < prev else 0.0

        # Default: delta shaping
        max_dist = math.sqrt(config.GAME_WIDTH**2 + config.GAME_HEIGHT**2)
        delta_norm = (prev - current_dist) / max_dist  # >0 if moved closer
        shaped = self.shaping_scale * float(delta_norm)
        shaped = float(np.clip(shaped, -self.shaping_clip, self.shaping_clip))
        return shaped
    
    def _get_info(self):
        """Get additional info dict"""
        return {
            'phase': self.current_phase,
            'enemies_destroyed': self.enemies_destroyed,
            'spawners_destroyed': self.spawners_destroyed,
            'player_health': self.player.health,
            'episode_reward': self.episode_reward,
            'episode_steps': self.current_step,
            'steps_in_phase': self.current_step - self.phase_start_step,
            'win': bool(self.win),
            'win_step': -1 if self.win_step is None else int(self.win_step),
            'first_spawner_kill_step': -1 if self.first_spawner_kill_step is None else int(self.first_spawner_kill_step),
            'shaping_mode': str(self.shaping_mode),
        }
