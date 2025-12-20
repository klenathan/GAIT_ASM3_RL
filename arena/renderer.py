"""
Beautiful Pygame renderer for the Arena environment
Displays game state with on-screen training metrics
"""

import pygame
import math
import numpy as np
from arena import config


class ArenaRenderer:
    """Handles all rendering for the Arena environment"""
    
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
        pygame.display.set_caption("Deep RL Arena")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.title_font = pygame.font.Font(None, 32)
        self.small_font = pygame.font.Font(None, 20)
        
        # Debug flags
        self.show_health = True
        self.show_vision = False
        
        # Metrics panel position
        self.panel_x = config.GAME_WIDTH + 10
        self.panel_width = config.SCREEN_WIDTH - config.GAME_WIDTH - 20
        
    def render(self, env, training_metrics=None):
        """
        Render the entire scene
        
        Args:
            env: ArenaEnv instance
            training_metrics: Dict with training stats (episode, reward, etc.)
        """
        # Clear screen
        self.screen.fill(config.COLOR_BACKGROUND)
        
        # Draw game boundary
        pygame.draw.rect(self.screen, (50, 50, 70), 
                        (0, 0, config.GAME_WIDTH, config.GAME_HEIGHT), 2)
        
        # Draw entities
        self._draw_spawners(env.spawners)
        self._draw_enemies(env.enemies)
        self._draw_projectiles(env.projectiles)
        self._draw_player(env.player)
        
        # Draw metrics panel
        if training_metrics:
            self._draw_metrics_panel(env, training_metrics)
            
        # Draw vision debug
        if self.show_vision:
            self._draw_vision_debug(env)
        
        # Update display
        pygame.display.flip()
        self.clock.tick(config.FPS)
        
    def _draw_player(self, player):
        """Draw player ship"""
        if not player.alive:
            return
            
        pos = (int(player.pos[0]), int(player.pos[1]))
        
        # Draw ship as triangle pointing in rotation direction
        points = self._get_ship_points(player.pos, player.rotation, player.radius)
        pygame.draw.polygon(self.screen, config.COLOR_PLAYER, points)
        pygame.draw.polygon(self.screen, (255, 255, 255), points, 2)
        
        # Draw health bar
        self._draw_health_bar(player.pos, player.radius, player.get_health_ratio())
        
    def _draw_enemies(self, enemies):
        """Draw all enemies"""
        for enemy in enemies:
            if enemy.alive:
                pos = (int(enemy.pos[0]), int(enemy.pos[1]))
                pygame.draw.circle(self.screen, config.COLOR_ENEMY, pos, enemy.radius)
                pygame.draw.circle(self.screen, (255, 100, 100), pos, enemy.radius, 2)
                
                # Small health indicator
                health_ratio = enemy.health / enemy.max_health
                if self.show_health and health_ratio < 1.0:
                    self._draw_health_bar(enemy.pos, enemy.radius, health_ratio, small=True)
    
    def _draw_spawners(self, spawners):
        """Draw all spawners"""
        for spawner in spawners:
            if spawner.alive:
                pos = (int(spawner.pos[0]), int(spawner.pos[1]))
                
                # Draw pulsing effect
                pulse = math.sin(pygame.time.get_ticks() * 0.003) * 3
                radius = spawner.radius + pulse
                
                # Main body
                pygame.draw.circle(self.screen, config.COLOR_SPAWNER, pos, int(radius))
                pygame.draw.circle(self.screen, (255, 150, 255), pos, int(radius), 3)
                
                # Inner core
                pygame.draw.circle(self.screen, (255, 100, 255), pos, int(radius * 0.5))
                
                # Health bar
                if self.show_health:
                    self._draw_health_bar(spawner.pos, spawner.radius, 
                                         spawner.health / spawner.max_health)
    
    def _draw_projectiles(self, projectiles):
        """Draw all projectiles"""
        for proj in projectiles:
            if proj.alive:
                pos = (int(proj.pos[0]), int(proj.pos[1]))
                color = config.COLOR_PLAYER_PROJECTILE if proj.is_player_projectile else config.COLOR_PROJECTILE
                
                # Draw with trail effect
                pygame.draw.circle(self.screen, color, pos, proj.radius + 2)
                pygame.draw.circle(self.screen, (255, 255, 255), pos, proj.radius)
    
    def _draw_health_bar(self, pos, entity_radius, health_ratio, small=False):
        """Draw health bar above entity"""
        bar_width = 30 if not small else 20
        bar_height = 4 if not small else 3
        bar_y_offset = entity_radius + 8 if not small else entity_radius + 5
        
        # Background (convert to int to avoid pygame rect errors)
        bar_x = int(pos[0] - bar_width // 2)
        bar_y = int(pos[1] - bar_y_offset)
        pygame.draw.rect(self.screen, (50, 50, 50), 
                        (bar_x, bar_y, bar_width, bar_height))
        
        # Health
        health_width = int(bar_width * health_ratio)
        if health_ratio > 0.6:
            color = config.COLOR_HEALTH_GOOD
        elif health_ratio > 0.3:
            color = config.COLOR_HEALTH_MEDIUM
        else:
            color = config.COLOR_HEALTH_BAD
            
        pygame.draw.rect(self.screen, color, 
                        (bar_x, bar_y, health_width, bar_height))
    
    def _draw_metrics_panel(self, env, metrics):
        """Draw training metrics panel on the right side"""
        x = self.panel_x
        y = 20
        line_height = 30
        
        # Panel background
        panel_rect = pygame.Rect(x - 5, y - 5, self.panel_width - 10, 
                                config.SCREEN_HEIGHT - 30)
        pygame.draw.rect(self.screen, config.COLOR_PANEL_BG, panel_rect)
        pygame.draw.rect(self.screen, config.COLOR_PANEL_BORDER, panel_rect, 2)
        
        # Title
        title = self.title_font.render("Training Metrics", True, config.COLOR_TEXT)
        self.screen.blit(title, (x + 10, y))
        y += 45
        
        # Separator
        pygame.draw.line(self.screen, config.COLOR_PANEL_BORDER, 
                        (x, y), (x + self.panel_width - 20, y), 1)
        y += 15
        
        # Episode info
        self._draw_metric(x, y, "Episode:", metrics.get('episode', 0))
        y += line_height
        
        self._draw_metric(x, y, "Step:", env.current_step)
        y += line_height
        
        self._draw_metric(x, y, "Phase:", f"{env.current_phase + 1}/{config.MAX_PHASES}")
        y += line_height
        
        y += 10
        pygame.draw.line(self.screen, config.COLOR_PANEL_BORDER, 
                        (x, y), (x + self.panel_width - 20, y), 1)
        y += 15
        
        # Reward info
        episode_reward = metrics.get('episode_reward', 0)
        self._draw_metric(x, y, "Episode Reward:", f"{episode_reward:.1f}")
        y += line_height
        
        total_reward = metrics.get('total_reward', 0)
        self._draw_metric(x, y, "Total Reward:", f"{total_reward:.1f}")
        y += line_height
        
        y += 10
        pygame.draw.line(self.screen, config.COLOR_PANEL_BORDER, 
                        (x, y), (x + self.panel_width - 20, y), 1)
        y += 15
        
        # Combat stats
        self._draw_metric(x, y, "Enemies Killed:", env.enemies_destroyed)
        y += line_height
        
        self._draw_metric(x, y, "Spawners Killed:", env.spawners_destroyed)
        y += line_height
        
        self._draw_metric(x, y, "Active Enemies:", len(env.enemies))
        y += line_height
        
        self._draw_metric(x, y, "Active Spawners:", len(env.spawners))
        y += line_height
        
        y += 10
        pygame.draw.line(self.screen, config.COLOR_PANEL_BORDER, 
                        (x, y), (x + self.panel_width - 20, y), 1)
        y += 15
        
        # Player stats
        health_pct = int(env.player.get_health_ratio() * 100)
        self._draw_metric(x, y, "Player Health:", f"{health_pct}%")
        y += line_height
        
        # FPS
        fps = int(self.clock.get_fps())
        self._draw_metric(x, y, "FPS:", fps)
        y += line_height
        
        # Training rate (if available)
        if 'timesteps' in metrics:
            self._draw_metric(x, y, "Timesteps:", metrics['timesteps'])
            y += line_height
    
    def _draw_metric(self, x, y, label, value):
        """Draw a single metric line"""
        label_surface = self.font.render(label, True, (180, 180, 180))
        value_surface = self.font.render(str(value), True, config.COLOR_TEXT)
        
        self.screen.blit(label_surface, (x + 10, y))
        self.screen.blit(value_surface, (x + 130, y))
    
    def _get_ship_points(self, pos, rotation, radius):
        """Get triangle points for ship based on rotation"""
        # Ship is a triangle pointing in the rotation direction
        points = []
        angles = [0, 2.4, -2.4]  # Front, back-left, back-right
        
        for angle in angles:
            total_angle = rotation + angle
            point_x = pos[0] + math.cos(total_angle) * radius
            point_y = pos[1] + math.sin(total_angle) * radius
            points.append((int(point_x), int(point_y)))
        
        return points
    
    def _draw_vision_debug(self, env):
        """Draw lines to nearest entities as seen by the player"""
        player_pos = env.player.pos
        
        # Nearest Enemy
        nearest_enemy = env._find_nearest_entity(env.enemies)
        if nearest_enemy:
            pygame.draw.line(self.screen, (255, 100, 100, 150), player_pos, nearest_enemy.pos, 1)
            
        # Nearest Spawner
        nearest_spawner = env._find_nearest_entity(env.spawners)
        if nearest_spawner:
            pygame.draw.line(self.screen, (255, 100, 255, 150), player_pos, nearest_spawner.pos, 1)

    def render_menu(self, menu):
        """Render the selection menu"""
        menu.render()
        pygame.display.flip()
        self.clock.tick(config.FPS)
    
    def close(self):
        """Clean up pygame resources"""
        pygame.quit()
