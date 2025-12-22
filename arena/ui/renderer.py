"""
Beautiful Pygame renderer for the Arena environment.
Displays game state with on-screen training metrics.
"""

import pygame
import math
import numpy as np

from arena.core import config

STYLE_1_LABELS = ["Rotate L", "Rotate R", "Thrust", "Idle", "Shoot"]
STYLE_2_LABELS = ["Up", "Down", "Left", "Right", "Idle", "Shoot"]

class ArenaRenderer:
    """Handles all rendering for the Arena environment."""
    
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
        
        self.model_output = None
        self.control_style = 1
        
    def render(self, env, training_metrics=None):
        """Render the entire scene."""
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
            
        # Draw model output panel (below metrics)
        if self.model_output:
            self._draw_model_output_panel()
            
        if self.show_vision:
            self._draw_vision_debug(env)
        
        pygame.display.flip()
        self.clock.tick(config.FPS)

    def set_model_output(self, output, style=1):
        """Update model output for the next render call."""
        self.model_output = output
        self.control_style = style
        
    def _draw_player(self, player):
        """Draw player ship."""
        if not player.alive:
            return
            
        points = self._get_ship_points(player.pos, player.rotation, player.radius)
        pygame.draw.polygon(self.screen, config.COLOR_PLAYER, points)
        pygame.draw.polygon(self.screen, (255, 255, 255), points, 2)
        
        self._draw_health_bar(player.pos, player.radius, player.get_health_ratio())
        
    def _draw_enemies(self, enemies):
        """Draw all enemies."""
        for enemy in enemies:
            if enemy.alive:
                pos = (int(enemy.pos[0]), int(enemy.pos[1]))
                pygame.draw.circle(self.screen, config.COLOR_ENEMY, pos, enemy.radius)
                pygame.draw.circle(self.screen, (255, 100, 100), pos, enemy.radius, 2)
                
                health_ratio = enemy.health / enemy.max_health
                if self.show_health and health_ratio < 1.0:
                    self._draw_health_bar(enemy.pos, enemy.radius, health_ratio, small=True)
    
    def _draw_spawners(self, spawners):
        """Draw all spawners."""
        for spawner in spawners:
            if spawner.alive:
                pos = (int(spawner.pos[0]), int(spawner.pos[1]))
                pulse = math.sin(pygame.time.get_ticks() * 0.003) * 3
                radius = spawner.radius + pulse
                
                pygame.draw.circle(self.screen, config.COLOR_SPAWNER, pos, int(radius))
                pygame.draw.circle(self.screen, (255, 150, 255), pos, int(radius), 3)
                pygame.draw.circle(self.screen, (255, 100, 255), pos, int(radius * 0.5))
                
                if self.show_health:
                    self._draw_health_bar(spawner.pos, spawner.radius, 
                                         spawner.health / spawner.max_health)
    
    def _draw_projectiles(self, projectiles):
        """Draw all projectiles."""
        for proj in projectiles:
            if proj.alive:
                pos = (int(proj.pos[0]), int(proj.pos[1]))
                color = config.COLOR_PLAYER_PROJECTILE if proj.is_player_projectile else config.COLOR_PROJECTILE
                pygame.draw.circle(self.screen, color, pos, proj.radius + 2)
                pygame.draw.circle(self.screen, (255, 255, 255), pos, proj.radius)
    
    def _draw_health_bar(self, pos, entity_radius, health_ratio, small=False):
        """Draw health bar above entity."""
        bar_width = 30 if not small else 20
        bar_height = 4 if not small else 3
        bar_y_offset = entity_radius + 8 if not small else entity_radius + 5
        
        bar_x = int(pos[0] - bar_width // 2)
        bar_y = int(pos[1] - bar_y_offset)
        pygame.draw.rect(self.screen, (50, 50, 50), 
                        (bar_x, bar_y, bar_width, bar_height))
        
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
        """Draw training metrics panel."""
        x = self.panel_x
        y = 20
        line_height = 30
        
        panel_rect = pygame.Rect(x - 5, y - 5, self.panel_width - 10, 
                                config.SCREEN_HEIGHT - 30)
        pygame.draw.rect(self.screen, config.COLOR_PANEL_BG, panel_rect)
        pygame.draw.rect(self.screen, config.COLOR_PANEL_BORDER, panel_rect, 2)
        
        title = self.title_font.render("Training Metrics", True, config.COLOR_TEXT)
        self.screen.blit(title, (x + 10, y))
        y += 45
        
        pygame.draw.line(self.screen, config.COLOR_PANEL_BORDER, 
                        (x, y), (x + self.panel_width - 20, y), 1)
        y += 15
        
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
        
        self._draw_metric(x, y, "Enemies Killed:", env.enemies_destroyed)
        y += line_height
        self._draw_metric(x, y, "Spawners Killed:", env.spawners_destroyed)
        y += line_height
        
        y += 10
        pygame.draw.line(self.screen, config.COLOR_PANEL_BORDER, 
                        (x, y), (x + self.panel_width - 20, y), 1)
        y += 15
        
        health_pct = int(env.player.get_health_ratio() * 100)
        self._draw_metric(x, y, "Player Health:", f"{health_pct}%")
        y += line_height
        fps = int(self.clock.get_fps())
        self._draw_metric(x, y, "FPS:", fps)
        y += line_height

    def _draw_model_output_panel(self):
        """Draw model introspection panel."""
        output = self.model_output
        if not output: return
        
        # Determine labels based on style
        labels = STYLE_1_LABELS if self.control_style == 1 else STYLE_2_LABELS
        
        x = self.panel_x
        # Position below the metrics panel. Metrics panel uses ~config.SCREEN_HEIGHT - 30.
        # Let's split it or use a fixed offset if we know metrics size.
        # Actually, let's draw it in its own rect, maybe at the bottom half.
        y_start = 450 
        panel_rect = pygame.Rect(x - 5, y_start, self.panel_width - 10, 
                                config.SCREEN_HEIGHT - y_start - 10)
        
        pygame.draw.rect(self.screen, config.COLOR_PANEL_BG, panel_rect)
        pygame.draw.rect(self.screen, config.COLOR_PANEL_BORDER, panel_rect, 2)
        
        title_text = "Action Distribution" if not output.is_q_value else "Q-Values"
        title = self.title_font.render(title_text, True, config.COLOR_TEXT)
        self.screen.blit(title, (x + 10, y_start + 10))
        
        y = y_start + 45
        
        # Draw probabilities or Q-values
        values = output.action_probs if not output.is_q_value else output.q_values
        if values is not None:
            # For Q-values, we might want to normalize them for the bars
            if output.is_q_value:
                # Basic normalization for visualization (0 to 1 range relative to min/max)
                v_min, v_max = np.min(values), np.max(values)
                if v_max > v_min:
                    display_probs = (values - v_min) / (v_max - v_min)
                else:
                    display_probs = np.zeros_like(values)
            else:
                display_probs = values
                
            bar_max_width = self.panel_width - 120
            bar_height = 16
            
            for i, (label, prob) in enumerate(zip(labels, display_probs)):
                # Label
                label_color = config.COLOR_ACTION_SELECTED if i == output.action_taken else (180, 180, 180)
                txt = self.small_font.render(label, True, label_color)
                self.screen.blit(txt, (x + 10, y))
                
                # Bar background
                bar_x = x + 70
                pygame.draw.rect(self.screen, (40, 40, 60), (bar_x, y + 2, bar_max_width, bar_height))
                
                # Probability bar
                bar_width = int(bar_max_width * prob)
                color = config.COLOR_ACTION_BAR_HIGH if i == output.action_taken else config.COLOR_ACTION_BAR_LOW
                pygame.draw.rect(self.screen, color, (bar_x, y + 2, bar_width, bar_height))
                
                # Highlight if chosen
                if i == output.action_taken:
                    pygame.draw.rect(self.screen, config.COLOR_ACTION_SELECTED, (bar_x, y + 2, bar_max_width, bar_height), 1)
                
                # Percentage text
                val_text = f"{values[i]:.2f}" if output.is_q_value else f"{int(values[i]*100)}%"
                val_surf = self.small_font.render(val_text, True, config.COLOR_TEXT)
                self.screen.blit(val_surf, (bar_x + bar_max_width + 5, y))
                
                y += 24

        # Draw Value and Entropy at bottom
        y = panel_rect.bottom - 60
        pygame.draw.line(self.screen, config.COLOR_PANEL_BORDER, (x, y), (x + self.panel_width - 20, y))
        y += 10
        
        if output.value is not None:
            v_color = config.COLOR_VALUE_POSITIVE if output.value >= 0 else config.COLOR_VALUE_NEGATIVE
            self._draw_output_field(x + 10, y, "V-Estimate:", f"{output.value:.2f}", v_color)
            
        if output.entropy is not None:
            # Lower entropy = more confident
            # Max entropy = log(n_actions): Style 1 has 5 actions, Style 2 has 6 actions
            n_actions = len(labels)
            max_entropy = math.log(n_actions)  # ln(5)≈1.61, ln(6)≈1.79
            conf_ratio = 1.0 - min(output.entropy / max_entropy, 1.0)
            e_color = config.COLOR_ENTROPY_HIGH if conf_ratio > 0.7 else config.COLOR_ENTROPY_LOW
            self._draw_output_field(x + 10, y + 22, "Confidence:", f"{int(conf_ratio*100)}%", e_color)

    def _draw_output_field(self, x, y, label, value, val_color):
        """Draw a small field in the output panel."""
        lbl = self.small_font.render(label, True, (150, 150, 150))
        val = self.small_font.render(value, True, val_color)
        self.screen.blit(lbl, (x, y))
        self.screen.blit(val, (x + 85, y))

    def _draw_metric(self, x, y, label, value):
        """Draw a single metric line."""
        label_surface = self.font.render(label, True, (180, 180, 180))
        value_surface = self.font.render(str(value), True, config.COLOR_TEXT)
        self.screen.blit(label_surface, (x + 10, y))
        self.screen.blit(value_surface, (x + 130, y))
    
    def _get_ship_points(self, pos, rotation, radius):
        """Get triangle points for ship."""
        points = []
        angles = [0, 2.4, -2.4]
        for angle in angles:
            total_angle = rotation + angle
            point_x = pos[0] + math.cos(total_angle) * radius
            point_y = pos[1] + math.sin(total_angle) * radius
            points.append((int(point_x), int(point_y)))
        return points
    
    def _draw_vision_debug(self, env):
        """Draw vision lines."""
        player_pos = env.player.pos
        nearest_enemy = env._find_nearest_entity(env.enemies)
        if nearest_enemy:
            pygame.draw.line(self.screen, (255, 100, 100, 150), player_pos, nearest_enemy.pos, 1)
        nearest_spawner = env._find_nearest_entity(env.spawners)
        if nearest_spawner:
            pygame.draw.line(self.screen, (255, 100, 255, 150), player_pos, nearest_spawner.pos, 1)

    def render_menu(self, menu):
        """Render selection menu."""
        menu.render()
        pygame.display.flip()
        self.clock.tick(config.FPS)
    
    def close(self):
        pygame.quit()
