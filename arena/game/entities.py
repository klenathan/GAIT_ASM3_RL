"""
Game entities for the Arena environment.
"""

import math
import numpy as np

from arena.core import config
from arena.game import utils

class Player:
    """Player-controlled ship."""
    
    def __init__(self, x, y):
        self.pos = np.array([x, y], dtype=np.float32)
        self.velocity = np.array([0, 0], dtype=np.float32)
        self.rotation = 0  # Radians, 0 = right
        self.health = config.PLAYER_MAX_HEALTH
        self.max_health = config.PLAYER_MAX_HEALTH
        self.radius = config.PLAYER_RADIUS
        self.shoot_cooldown = 0
        self.alive = True
        
    def update_style_1(self, action):
        """Update with rotation/thrust controls."""
        if action == 1:  # Thrust
            thrust_vec = utils.vector_from_angle(self.rotation, config.PLAYER_THRUST)
            self.velocity += thrust_vec
        elif action == 2:  # Rotate left
            self.rotation += math.radians(config.PLAYER_ROTATION_SPEED)
        elif action == 3:  # Rotate right
            self.rotation -= math.radians(config.PLAYER_ROTATION_SPEED)
        
        self.rotation = self.rotation % (2 * math.pi)
        self.velocity *= config.PLAYER_FRICTION
        self.velocity = utils.limit_magnitude(self.velocity, config.PLAYER_MAX_VELOCITY)
        self.pos += self.velocity
        
        self.pos = utils.keep_in_bounds(
            self.pos, 0, config.GAME_WIDTH, 0, config.GAME_HEIGHT, self.radius
        )
        
        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1
    
    def update_style_2(self, action):
        """Update with directional movement controls."""
        self.velocity = np.array([0, 0], dtype=np.float32)
        
        if action == 1:  # Up
            self.velocity[1] = -config.PLAYER_SPEED
        elif action == 2:  # Down
            self.velocity[1] = config.PLAYER_SPEED
        elif action == 3:  # Left
            self.velocity[0] = -config.PLAYER_SPEED
        elif action == 4:  # Right
            self.velocity[0] = config.PLAYER_SPEED
        
        if utils.magnitude(self.velocity) > 0:
            self.rotation = math.atan2(self.velocity[1], self.velocity[0])
        
        self.pos += self.velocity
        self.pos = utils.keep_in_bounds(
            self.pos, 0, config.GAME_WIDTH, 0, config.GAME_HEIGHT, self.radius
        )
        
        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1
    
    def shoot(self):
        """Attempt to shoot."""
        if self.shoot_cooldown == 0:
            self.shoot_cooldown = config.PLAYER_SHOOT_COOLDOWN
            return True
        return False
    
    def take_damage(self, amount):
        """Handle taking damage."""
        self.health -= amount
        if self.health <= 0:
            self.health = 0
            self.alive = False
    
    def get_health_ratio(self):
        """Returns health as a ratio 0-1."""
        return self.health / self.max_health

class Enemy:
    """Enemy ship that chases the player."""
    
    def __init__(self, x, y, speed_multiplier=1.0):
        self.pos = np.array([x, y], dtype=np.float32)
        self.health = config.ENEMY_HEALTH
        self.max_health = config.ENEMY_HEALTH
        self.radius = config.ENEMY_RADIUS
        self.speed = config.ENEMY_SPEED * speed_multiplier
        self.shoot_cooldown = 0
        self.alive = True
        
    def update(self, player_pos):
        """Move toward player position."""
        direction = player_pos - self.pos
        dist = utils.magnitude(direction)
        
        if dist > 0:
            direction = (direction / dist) * self.speed
            self.pos += direction
        
        self.pos = utils.keep_in_bounds(
            self.pos, 0, config.GAME_WIDTH, 0, config.GAME_HEIGHT, self.radius
        )
        
        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1
    
    def can_shoot(self):
        return self.shoot_cooldown == 0
    
    def shoot(self, player_pos, np_random=np.random):
        """Attempt to shoot at player."""
        if self.can_shoot() and np_random.random() < config.ENEMY_SHOOT_PROBABILITY:
            self.shoot_cooldown = config.ENEMY_SHOOT_COOLDOWN
            return True
        return False
    
    def take_damage(self, amount):
        self.health -= amount
        if self.health <= 0:
            self.alive = False

class Spawner:
    """Spawner that creates enemies."""
    
    def __init__(self, x, y, spawn_rate_multiplier=1.0):
        self.pos = np.array([x, y], dtype=np.float32)
        self.health = config.SPAWNER_HEALTH
        self.max_health = config.SPAWNER_HEALTH
        self.radius = config.SPAWNER_RADIUS
        self.spawn_cooldown = 0
        self.spawn_rate = int(config.SPAWNER_SPAWN_COOLDOWN * spawn_rate_multiplier)
        self.alive = True
        self.enemies_spawned = 0
        
    def update(self):
        if self.spawn_cooldown > 0:
            self.spawn_cooldown -= 1
    
    def can_spawn(self, current_enemy_count, max_enemies=None):
        if max_enemies is None:
            max_enemies = config.SPAWNER_MAX_ENEMIES
        return (self.spawn_cooldown == 0 and 
                current_enemy_count < max_enemies)
    
    def spawn_enemy(self, np_random=np.random, speed_multiplier=1.0):
        """Create a new enemy around the spawner."""
        if self.spawn_cooldown == 0:
            self.spawn_cooldown = self.spawn_rate
            angle = np_random.random() * 2 * math.pi
            offset = utils.vector_from_angle(angle, self.radius + config.ENEMY_RADIUS + 5)
            spawn_pos = self.pos + offset
            
            spawn_pos = utils.keep_in_bounds(
                spawn_pos, 0, config.GAME_WIDTH, 0, config.GAME_HEIGHT, config.ENEMY_RADIUS
            )
            
            self.enemies_spawned += 1
            return Enemy(spawn_pos[0], spawn_pos[1], speed_multiplier)
        return None
    
    def take_damage(self, amount):
        self.health -= amount
        if self.health <= 0:
            self.alive = False

class Projectile:
    """Bullet projectile."""
    
    def __init__(self, x, y, angle, is_player_projectile=True):
        self.pos = np.array([x, y], dtype=np.float32)
        self.velocity = utils.vector_from_angle(angle, config.PROJECTILE_SPEED)
        self.radius = config.PROJECTILE_RADIUS
        self.damage = config.PROJECTILE_DAMAGE
        self.lifetime = config.PROJECTILE_LIFETIME
        self.is_player_projectile = is_player_projectile
        self.alive = True
        
    def update(self):
        self.pos += self.velocity
        self.lifetime -= 1
        
        if (self.pos[0] < 0 or self.pos[0] > config.GAME_WIDTH or
            self.pos[1] < 0 or self.pos[1] > config.GAME_HEIGHT or
            self.lifetime <= 0):
            self.alive = False
    
    def hit(self):
        self.alive = False
