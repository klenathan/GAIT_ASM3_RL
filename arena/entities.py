"""
Game entities for the Arena environment
Player, Enemy, Spawner, and Projectile classes
"""

import numpy as np
import math
from arena import config
from arena import utils


class Player:
    """Player-controlled ship"""
    
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
        """Update with rotation/thrust controls"""
        # Apply action
        if action == 1:  # Thrust
            thrust_vec = utils.vector_from_angle(self.rotation, config.PLAYER_THRUST)
            self.velocity += thrust_vec
        elif action == 2:  # Rotate left
            self.rotation += math.radians(config.PLAYER_ROTATION_SPEED)
        elif action == 3:  # Rotate right
            self.rotation -= math.radians(config.PLAYER_ROTATION_SPEED)
        
        # Normalize rotation
        self.rotation = self.rotation % (2 * math.pi)
        
        # Apply friction
        self.velocity *= config.PLAYER_FRICTION
        
        # Limit velocity
        self.velocity = utils.limit_magnitude(self.velocity, config.PLAYER_MAX_VELOCITY)
        
        # Update position
        self.pos += self.velocity
        
        # Keep in bounds
        self.pos = utils.keep_in_bounds(
            self.pos, 0, config.GAME_WIDTH, 0, config.GAME_HEIGHT, self.radius
        )
        
        # Update cooldown
        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1
    
    def update_style_2(self, action):
        """Update with directional movement controls"""
        # Reset velocity for direct control
        self.velocity = np.array([0, 0], dtype=np.float32)
        
        # Apply action
        if action == 1:  # Up
            self.velocity[1] = -config.PLAYER_SPEED
        elif action == 2:  # Down
            self.velocity[1] = config.PLAYER_SPEED
        elif action == 3:  # Left
            self.velocity[0] = -config.PLAYER_SPEED
        elif action == 4:  # Right
            self.velocity[0] = config.PLAYER_SPEED
        
        # Update rotation to face movement direction
        if utils.magnitude(self.velocity) > 0:
            self.rotation = math.atan2(self.velocity[1], self.velocity[0])
        
        # Update position
        self.pos += self.velocity
        
        # Keep in bounds
        self.pos = utils.keep_in_bounds(
            self.pos, 0, config.GAME_WIDTH, 0, config.GAME_HEIGHT, self.radius
        )
        
        # Update cooldown
        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1
    
    def shoot(self):
        """Attempt to shoot, returns True if successful"""
        if self.shoot_cooldown == 0:
            self.shoot_cooldown = config.PLAYER_SHOOT_COOLDOWN
            return True
        return False
    
    def take_damage(self, amount):
        """Take damage"""
        self.health -= amount
        if self.health <= 0:
            self.health = 0
            self.alive = False
    
    def get_health_ratio(self):
        """Get health as ratio 0-1"""
        return self.health / self.max_health


class Enemy:
    """Enemy that navigates toward player"""
    
    def __init__(self, x, y, speed_multiplier=1.0):
        self.pos = np.array([x, y], dtype=np.float32)
        self.health = config.ENEMY_HEALTH
        self.max_health = config.ENEMY_HEALTH
        self.radius = config.ENEMY_RADIUS
        self.speed = config.ENEMY_SPEED * speed_multiplier
        self.shoot_cooldown = 0
        self.alive = True
        
    def update(self, player_pos):
        """Move toward player"""
        # Calculate direction to player
        direction = player_pos - self.pos
        distance = utils.magnitude(direction)
        
        if distance > 0:
            # Normalize and scale by speed
            direction = (direction / distance) * self.speed
            self.pos += direction
        
        # Keep in bounds
        self.pos = utils.keep_in_bounds(
            self.pos, 0, config.GAME_WIDTH, 0, config.GAME_HEIGHT, self.radius
        )
        
        # Update cooldown
        if self.shoot_cooldown > 0:
            self.shoot_cooldown -= 1
    
    def can_shoot(self):
        """Check if enemy can shoot"""
        return self.shoot_cooldown == 0
    
    def shoot(self, player_pos):
        """Attempt to shoot at player, returns True if successful"""
        if self.can_shoot() and np.random.random() < config.ENEMY_SHOOT_PROBABILITY:
            self.shoot_cooldown = config.ENEMY_SHOOT_COOLDOWN
            return True
        return False
    
    def take_damage(self, amount):
        """Take damage"""
        self.health -= amount
        if self.health <= 0:
            self.health = 0
            self.alive = False


class Spawner:
    """Spawner that periodically creates enemies"""
    
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
        """Update spawner"""
        if self.spawn_cooldown > 0:
            self.spawn_cooldown -= 1
    
    def can_spawn(self, current_enemy_count):
        """Check if spawner can spawn an enemy"""
        return (self.spawn_cooldown == 0 and 
                current_enemy_count < config.SPAWNER_MAX_ENEMIES)
    
    def spawn_enemy(self, speed_multiplier=1.0):
        """Spawn an enemy at a random offset from spawner"""
        if self.spawn_cooldown == 0:
            self.spawn_cooldown = self.spawn_rate
            
            # Spawn at random angle around spawner
            angle = np.random.random() * 2 * math.pi
            offset = utils.vector_from_angle(angle, self.radius + config.ENEMY_RADIUS + 5)
            spawn_pos = self.pos + offset
            
            # Keep in bounds
            spawn_pos = utils.keep_in_bounds(
                spawn_pos, 0, config.GAME_WIDTH, 0, config.GAME_HEIGHT, config.ENEMY_RADIUS
            )
            
            self.enemies_spawned += 1
            return Enemy(spawn_pos[0], spawn_pos[1], speed_multiplier)
        return None
    
    def take_damage(self, amount):
        """Take damage"""
        self.health -= amount
        if self.health <= 0:
            self.health = 0
            self.alive = False


class Projectile:
    """Projectile/bullet entity"""
    
    def __init__(self, x, y, angle, is_player_projectile=True):
        self.pos = np.array([x, y], dtype=np.float32)
        self.velocity = utils.vector_from_angle(angle, config.PROJECTILE_SPEED)
        self.radius = config.PROJECTILE_RADIUS
        self.damage = config.PROJECTILE_DAMAGE
        self.lifetime = config.PROJECTILE_LIFETIME
        self.is_player_projectile = is_player_projectile
        self.alive = True
        
    def update(self):
        """Update projectile position"""
        self.pos += self.velocity
        self.lifetime -= 1
        
        # Check if out of bounds or lifetime expired
        if (self.pos[0] < 0 or self.pos[0] > config.GAME_WIDTH or
            self.pos[1] < 0 or self.pos[1] > config.GAME_HEIGHT or
            self.lifetime <= 0):
            self.alive = False
    
    def hit(self):
        """Mark projectile as hit"""
        self.alive = False
