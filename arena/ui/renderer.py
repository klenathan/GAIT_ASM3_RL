"""
Beautiful Pygame renderer for the Arena environment.
Displays game state with on-screen training metrics.
"""

import pygame
import math
import numpy as np
from pygame_emojis import load_emoji

from arena.core import config

STYLE_1_LABELS = ["Idle", "Thrust", "Rotate L", "Rotate R", "Shoot"]
STYLE_2_LABELS = ["Idle", "Up", "Down", "Left", "Right", "Shoot"]

class ArenaRenderer:
    """Handles all rendering for the Arena environment."""
    
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
        pygame.display.set_caption("Deep RL Arena")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.title_font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 20)
        self.section_font = pygame.font.Font(None, 18)
        
        # Debug flags
        self.show_health = True
        self.show_vision = False
        self.show_debug = False  # Comprehensive debug mode (D key)
        
        # Metrics panel position
        self.panel_x = config.GAME_WIDTH + 10
        self.panel_width = config.SCREEN_WIDTH - config.GAME_WIDTH - 20
        
        self.model_output = None
        self.control_style = 1
        
        # Emoji cache for performance
        self._emoji_cache = {}
        self._load_emojis()
    
    def _load_emojis(self):
        """Pre-load emoji surfaces for panel icons."""
        emoji_size = (14, 14)
        emoji_list = ['⚔️', '🎯', '💀', '❤️', '🚀', '📍', '🔄', '⏱️']
        for emoji in emoji_list:
            try:
                self._emoji_cache[emoji] = load_emoji(emoji, emoji_size)
            except Exception:
                self._emoji_cache[emoji] = None
    
    def _get_emoji(self, emoji):
        """Get cached emoji surface or None."""
        return self._emoji_cache.get(emoji)
        
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
        
        # Draw metrics panel (includes action distribution if model output available)
        if training_metrics:
            self._draw_metrics_panel(env, training_metrics)
            
        if self.show_vision:
            self._draw_vision_debug(env)
        
        if self.show_debug:
            self._draw_debug_overlay(env)
        
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
        """Draw training metrics panel with organized sections."""
        x = self.panel_x
        y = 10
        line_height = 22
        section_gap = 6
        
        available_width = config.SCREEN_WIDTH - config.GAME_WIDTH - 20
        
        # Panel background
        panel_rect = pygame.Rect(x - 5, y - 5, available_width, config.SCREEN_HEIGHT - 30)
        pygame.draw.rect(self.screen, config.COLOR_PANEL_BG, panel_rect)
        pygame.draw.rect(self.screen, config.COLOR_PANEL_BORDER, panel_rect, 2)
        
        # Title
        title = self.title_font.render("Training Metrics", True, config.COLOR_TEXT)
        self.screen.blit(title, (x + 10, y))
        y += 32
        
        # ═══ SESSION SECTION ═══
        y = self._draw_section_header(x, y, "SESSION", available_width)
        self._draw_metric_row(x, y, "Episode", metrics.get('episode', 0), available_width)
        y += line_height
        self._draw_metric_row(x, y, "Step", env.current_step, available_width)
        y += line_height
        self._draw_metric_row(x, y, "Phase", f"{env.current_phase + 1}/{config.MAX_PHASES}", available_width)
        y += line_height + section_gap
        
        # ═══ REWARDS SECTION ═══
        y = self._draw_section_header(x, y, "REWARDS", available_width)
        ep_reward = metrics.get('episode_reward', 0)
        reward_color = (100, 255, 100) if ep_reward >= 0 else (255, 100, 100)
        self._draw_metric_row(x, y, "Episode", f"{ep_reward:.1f}", available_width, value_color=reward_color)
        y += line_height
        total_reward = metrics.get('total_reward', 0)
        total_color = (100, 255, 100) if total_reward >= 0 else (255, 100, 100)
        self._draw_metric_row(x, y, "Total", f"{total_reward:.1f}", available_width, value_color=total_color)
        y += line_height + section_gap
        
        # ═══ COMBAT SECTION ═══
        y = self._draw_section_header(x, y, "COMBAT", available_width)
        self._draw_metric_row(x, y, "Enemies", env.enemies_destroyed, available_width, emoji='⚔️')
        y += line_height
        self._draw_metric_row(x, y, "Spawners", env.spawners_destroyed, available_width, emoji='🎯')
        y += line_height
        health_pct = int(env.player.get_health_ratio() * 100)
        health_color = config.COLOR_HEALTH_GOOD if health_pct > 60 else config.COLOR_HEALTH_MEDIUM if health_pct > 30 else config.COLOR_HEALTH_BAD
        self._draw_metric_row(x, y, "Health", f"{health_pct}%", available_width, emoji='❤️', value_color=health_color)
        y += line_height + section_gap
        
        # ═══ ENTITIES SECTION ═══
        y = self._draw_section_header(x, y, "ENTITIES", available_width)
        alive_enemies = len([e for e in env.enemies if e.alive])
        self._draw_metric_row(x, y, "Enemies", alive_enemies, available_width)
        y += line_height
        alive_spawners = len([s for s in env.spawners if s.alive])
        self._draw_metric_row(x, y, "Spawners", alive_spawners, available_width)
        y += line_height
        alive_projectiles = len([p for p in env.projectiles if p.alive])
        self._draw_metric_row(x, y, "Projectiles", alive_projectiles, available_width)
        y += line_height + section_gap
        
        # ═══ PLAYER SECTION ═══
        if env.player.alive:
            y = self._draw_section_header(x, y, "PLAYER", available_width)
            player_speed = np.linalg.norm(env.player.velocity)
            self._draw_metric_row(x, y, "Speed", f"{player_speed:.1f}", available_width, emoji='🚀')
            y += line_height
            self._draw_metric_row(x, y, "Position", f"({int(env.player.pos[0])},{int(env.player.pos[1])})", available_width, emoji='📍')
            y += line_height
            rotation_deg = math.degrees(env.player.rotation) % 360
            self._draw_metric_row(x, y, "Rotation", f"{rotation_deg:.0f}°", available_width, emoji='🔄')
            y += line_height
            self._draw_metric_row(x, y, "Cooldown", env.player.shoot_cooldown, available_width, emoji='⏱️')
            y += line_height + section_gap
        
        # ═══ SYSTEM SECTION ═══
        y = self._draw_section_header(x, y, "SYSTEM", available_width)
        fps = int(self.clock.get_fps())
        fps_color = (100, 255, 100) if fps >= 55 else (255, 200, 100) if fps >= 30 else (255, 100, 100)
        self._draw_metric_row(x, y, "FPS", fps, available_width, value_color=fps_color)
        y += line_height + section_gap
        
        # ═══ ACTION SECTION (if model output available) ═══
        if self.model_output:
            y = self._draw_action_section(x, y, available_width, line_height)
    
    def _draw_section_header(self, x, y, title, width):
        """Draw a section header with subtle line."""
        # Draw separator line
        pygame.draw.line(self.screen, (60, 60, 80), (x + 5, y), (x + width - 15, y), 1)
        y += 8
        # Draw section title
        header = self.section_font.render(title, True, (120, 120, 150))
        self.screen.blit(header, (x + 10, y))
        return y + 18
    
    def _draw_metric_row(self, x, y, label, value, width, emoji=None, value_color=None):
        """Draw a metric row with optional emoji and right-aligned value."""
        label_x = x + 10
        value_x = x + width - 20
        
        # Draw emoji if provided
        if emoji and self._get_emoji(emoji):
            emoji_surf = self._get_emoji(emoji)
            self.screen.blit(emoji_surf, (label_x, y + 2))
            label_x += 18
        
        # Draw label
        label_surface = self.small_font.render(label, True, (160, 160, 180))
        self.screen.blit(label_surface, (label_x, y))
        
        # Draw value (right-aligned)
        color = value_color if value_color else config.COLOR_TEXT
        value_surface = self.small_font.render(str(value), True, color)
        value_rect = value_surface.get_rect()
        value_rect.right = value_x
        value_rect.top = y
        self.screen.blit(value_surface, value_rect)
    
    def _draw_action_section(self, x, y, available_width, line_height):
        """Draw action distribution section integrated into metrics panel."""
        output = self.model_output
        if not output:
            return y
        
        # Section header
        title_text = "ACTION" if not output.is_q_value else "Q-VALUES"
        y = self._draw_section_header(x, y, title_text, available_width)
        
        labels = STYLE_1_LABELS if self.control_style == 1 else STYLE_2_LABELS
        values = output.action_probs if not output.is_q_value else output.q_values
        
        if values is not None:
            # Normalize Q-values for bar display
            if output.is_q_value:
                v_min, v_max = np.min(values), np.max(values)
                display_probs = (values - v_min) / (v_max - v_min) if v_max > v_min else np.zeros_like(values)
            else:
                display_probs = values
            
            bar_max_width = available_width - 100
            bar_height = 12
            
            for i, (label, prob) in enumerate(zip(labels, display_probs)):
                is_selected = i == output.action_taken
                
                # Label (compact)
                label_color = config.COLOR_ACTION_SELECTED if is_selected else (150, 150, 170)
                txt = self.small_font.render(label[:6], True, label_color)  # Truncate long labels
                self.screen.blit(txt, (x + 10, y))
                
                # Bar background
                bar_x = x + 60
                pygame.draw.rect(self.screen, (40, 40, 60), (bar_x, y + 2, bar_max_width, bar_height))
                
                # Probability bar
                bar_width = int(bar_max_width * prob)
                bar_color = config.COLOR_ACTION_BAR_HIGH if is_selected else config.COLOR_ACTION_BAR_LOW
                pygame.draw.rect(self.screen, bar_color, (bar_x, y + 2, bar_width, bar_height))
                
                # Selection highlight
                if is_selected:
                    pygame.draw.rect(self.screen, config.COLOR_ACTION_SELECTED, (bar_x, y + 2, bar_max_width, bar_height), 1)
                
                # Value text (right-aligned)
                val_text = f"{values[i]:.2f}" if output.is_q_value else f"{int(values[i]*100)}%"
                val_surf = self.small_font.render(val_text, True, config.COLOR_TEXT)
                val_rect = val_surf.get_rect(right=x + available_width - 15, top=y)
                self.screen.blit(val_surf, val_rect)
                
                y += 18
        
        # V-Estimate and Confidence on same line
        y += 4
        if output.value is not None:
            v_color = config.COLOR_VALUE_POSITIVE if output.value >= 0 else config.COLOR_VALUE_NEGATIVE
            v_text = self.small_font.render(f"V:{output.value:.2f}", True, v_color)
            self.screen.blit(v_text, (x + 10, y))
        
        if output.entropy is not None:
            n_actions = len(labels)
            max_entropy = math.log(n_actions)
            conf_ratio = 1.0 - min(output.entropy / max_entropy, 1.0)
            e_color = config.COLOR_ENTROPY_HIGH if conf_ratio > 0.7 else config.COLOR_ENTROPY_LOW
            conf_text = self.small_font.render(f"Conf:{int(conf_ratio*100)}%", True, e_color)
            self.screen.blit(conf_text, (x + 80, y))
        
        return y + line_height

    def _draw_model_output_panel(self, y_start=None):
        """Draw model introspection panel."""
        output = self.model_output
        if not output: return
        
        # Determine labels based on style
        labels = STYLE_1_LABELS if self.control_style == 1 else STYLE_2_LABELS
        
        x = self.panel_x
        # Position below the metrics panel dynamically
        y_start = y_start if y_start else 400
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
        """Draw a single metric line (legacy method for compatibility)."""
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

    def _draw_debug_overlay(self, env):
        """Draw comprehensive debug information overlay."""
        # Draw HP values next to health bars for all entities
        if env.player.alive:
            self._draw_debug_hp(env.player.pos, env.player.radius, env.player.health, env.player.max_health)
            self._draw_debug_velocity(env.player.pos, env.player.velocity, (100, 255, 100))
            
        for enemy in env.enemies:
            if enemy.alive:
                self._draw_debug_hp(enemy.pos, enemy.radius, enemy.health, enemy.max_health, small=True)
                self._draw_debug_velocity(enemy.pos, np.array([enemy.speed, 0]), (255, 100, 100), scale=2.0)
                
        for spawner in env.spawners:
            if spawner.alive:
                self._draw_debug_hp(spawner.pos, spawner.radius, spawner.health, spawner.max_health)
                # Show spawn cooldown
                cooldown_text = f"CD:{spawner.spawn_cooldown}"
                cd_surf = self.small_font.render(cooldown_text, True, (255, 200, 100))
                self.screen.blit(cd_surf, (int(spawner.pos[0] - 15), int(spawner.pos[1] + spawner.radius + 20)))
                # Show enemies spawned
                spawned_text = f"Spawned:{spawner.enemies_spawned}"
                sp_surf = self.small_font.render(spawned_text, True, (200, 150, 255))
                self.screen.blit(sp_surf, (int(spawner.pos[0] - 30), int(spawner.pos[1] + spawner.radius + 35)))
    
    def _draw_debug_hp(self, pos, entity_radius, health, max_health, small=False):
        """Draw HP value text next to health bar."""
        hp_text = f"{int(health)}/{int(max_health)}"
        color = (255, 255, 255) if health > max_health * 0.5 else (255, 200, 100) if health > max_health * 0.25 else (255, 100, 100)
        font = self.small_font if small else self.font
        hp_surf = font.render(hp_text, True, color)
        
        # Position to the right of the entity
        text_x = int(pos[0] + entity_radius + 5)
        text_y = int(pos[1] - entity_radius - 10)
        
        # Draw semi-transparent background
        bg_rect = hp_surf.get_rect()
        bg_rect.topleft = (text_x, text_y)
        bg_surface = pygame.Surface((bg_rect.width + 4, bg_rect.height + 2))
        bg_surface.set_alpha(180)
        bg_surface.fill((0, 0, 0))
        self.screen.blit(bg_surface, (text_x - 2, text_y - 1))
        
        self.screen.blit(hp_surf, (text_x, text_y))
    
    def _draw_debug_velocity(self, pos, velocity, color, scale=5.0):
        """Draw velocity vector as an arrow."""
        if np.linalg.norm(velocity) > 0.1:
            end_pos = pos + velocity * scale
            pygame.draw.line(self.screen, color, 
                           (int(pos[0]), int(pos[1])), 
                           (int(end_pos[0]), int(end_pos[1])), 2)
            # Draw arrowhead
            angle = math.atan2(velocity[1], velocity[0])
            arrow_size = 5
            left_angle = angle + 2.5
            right_angle = angle - 2.5
            left_point = (int(end_pos[0] - arrow_size * math.cos(left_angle)),
                         int(end_pos[1] - arrow_size * math.sin(left_angle)))
            right_point = (int(end_pos[0] - arrow_size * math.cos(right_angle)),
                          int(end_pos[1] - arrow_size * math.sin(right_angle)))
            pygame.draw.polygon(self.screen, color, 
                              [(int(end_pos[0]), int(end_pos[1])), left_point, right_point])
    
    def _draw_debug_line(self, x, y, label, value):
        """Draw a single debug info line."""
        label_surf = self.small_font.render(label, True, (180, 180, 180))
        value_surf = self.small_font.render(str(value), True, (255, 255, 255))
        self.screen.blit(label_surf, (x, y))
        self.screen.blit(value_surf, (x + 120, y))


    def render_menu(self, menu):
        """Render selection menu."""
        menu.render()
        pygame.display.flip()
        self.clock.tick(config.FPS)
    
    def close(self):
        pygame.quit()
