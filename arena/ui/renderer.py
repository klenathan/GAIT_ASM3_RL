"""
Beautiful Pygame renderer for the Arena environment.
Displays game state with on-screen training metrics.
"""

from arena.core import config
import numpy as np
import math
import pygame
import warnings
# Suppress pkg_resources deprecation warning from pygame
warnings.filterwarnings(
    "ignore", message=".*pkg_resources is deprecated.*", category=UserWarning)


try:
    from pygame_emojis import load_emoji
    HAS_PYGAME_EMOJIS = True
except ImportError:
    HAS_PYGAME_EMOJIS = False


STYLE_1_LABELS = ["Idle", "Thrust", "Rotate L", "Rotate R", "Shoot"]
STYLE_2_LABELS = ["Idle", "Up", "Down", "Left", "Right", "Shoot"]


class ArenaRenderer:
    """Handles all rendering for the Arena environment."""

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(
            (config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
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
        self.panel_x = config.WINDOW_GAME_WIDTH + 10
        self.panel_width = config.SIDEBAR_WIDTH - 20
        
        # Scrolling for metrics panel
        self.scroll_offset = 0
        self.max_scroll = 0

        self.model_output = None
        self.control_style = 1

        # Emoji cache for performance
        self._emoji_cache = {}
        self._load_emojis()

    def _load_emojis(self):
        """Pre-load emoji surfaces for panel icons."""
        if not HAS_PYGAME_EMOJIS:
            return

        emoji_size = (14, 14)
        emoji_list = ['⚔️', '🎯', '💀', '❤️', '🚀', '📍', '🔄', '⏱️', '👾']
        for emoji in emoji_list:
            try:
                self._emoji_cache[emoji] = load_emoji(emoji, emoji_size)
            except Exception:
                self._emoji_cache[emoji] = None

    def _get_emoji(self, emoji):
        """Get cached emoji surface or None."""
        return self._emoji_cache.get(emoji)

    def _s(self, value):
        """Scale a value by RENDER_SCALE."""
        return value * config.RENDER_SCALE

    def _spos(self, pos):
        """Scale a position (x, y) by RENDER_SCALE."""
        return (int(pos[0] * config.RENDER_SCALE), int(pos[1] * config.RENDER_SCALE))

    def render(self, env, training_metrics=None):
        """Render the entire scene."""
        self.screen.fill(config.COLOR_BACKGROUND)

        is_human = training_metrics.get(
            'is_human', False) if training_metrics else False

        # Draw game boundary
        pygame.draw.rect(self.screen, (50, 50, 70),
                         (0, 0, config.WINDOW_GAME_WIDTH, config.WINDOW_GAME_HEIGHT), 2)

        # Draw entities
        self._draw_spawners(env.spawners)
        self._draw_enemies(env.enemies)
        self._draw_projectiles(env.projectiles)
        self._draw_player(env.player)

        # Draw metrics panel (includes action distribution if model output available)
        if training_metrics:
            self._draw_metrics_panel(env, training_metrics, is_human=is_human)

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

        scaled_pos = player.pos * config.RENDER_SCALE
        scaled_radius = player.radius * config.RENDER_SCALE
        points = self._get_ship_points(
            scaled_pos, player.rotation, scaled_radius)
        
        # Apply red hue when damage is taken
        color = config.COLOR_PLAYER
        if hasattr(player, 'damage_flash_timer') and player.damage_flash_timer > 0:
            # Blend player color with red based on flash intensity
            flash_intensity = player.damage_flash_timer / 15.0
            red_overlay = (255, 50, 50)
            color = tuple(int(c * (1 - flash_intensity * 0.7) + r * flash_intensity * 0.7) 
                         for c, r in zip(config.COLOR_PLAYER, red_overlay))
        
        pygame.draw.polygon(self.screen, color, points)
        pygame.draw.polygon(self.screen, (255, 255, 255), points, 2)

        self._draw_health_bar(player.pos, player.radius,
                              player.get_health_ratio())

    def _draw_enemies(self, enemies):
        """Draw all enemies."""
        for enemy in enemies:
            if enemy.alive:
                pos = self._spos(enemy.pos)
                radius = int(self._s(enemy.radius))
                pygame.draw.circle(
                    self.screen, config.COLOR_ENEMY, pos, radius)
                pygame.draw.circle(
                    self.screen, (255, 100, 100), pos, radius, 2)

                health_ratio = enemy.health / enemy.max_health
                if self.show_health and health_ratio < 1.0:
                    self._draw_health_bar(
                        enemy.pos, enemy.radius, health_ratio, small=True)

    def _draw_spawners(self, spawners):
        """Draw all spawners."""
        for spawner in spawners:
            if spawner.alive:
                pos = self._spos(spawner.pos)
                pulse = math.sin(pygame.time.get_ticks() * 0.003) * self._s(3)
                radius = self._s(spawner.radius) + pulse

                pygame.draw.circle(
                    self.screen, config.COLOR_SPAWNER, pos, int(radius))
                pygame.draw.circle(
                    self.screen, (255, 150, 255), pos, int(radius), 3)
                pygame.draw.circle(
                    self.screen, (255, 100, 255), pos, int(radius * 0.5))

                if self.show_health:
                    self._draw_health_bar(spawner.pos, spawner.radius,
                                          spawner.health / spawner.max_health)

    def _draw_projectiles(self, projectiles):
        """Draw all projectiles."""
        for proj in projectiles:
            if proj.alive:
                pos = self._spos(proj.pos)
                radius = int(self._s(proj.radius))
                color = config.COLOR_PLAYER_PROJECTILE if proj.is_player_projectile else config.COLOR_PROJECTILE
                pygame.draw.circle(self.screen, color, pos, radius + 2)
                pygame.draw.circle(self.screen, (255, 255, 255), pos, radius)

    def _draw_health_bar(self, pos, entity_radius, health_ratio, small=False):
        """Draw health bar above entity."""
        bar_width = self._s(30 if not small else 20)
        bar_height = self._s(4 if not small else 3)
        bar_y_offset = self._s(
            entity_radius + 8 if not small else entity_radius + 5)

        scaled_pos = self._spos(pos)
        bar_x = int(scaled_pos[0] - bar_width // 2)
        bar_y = int(scaled_pos[1] - bar_y_offset)
        pygame.draw.rect(self.screen, (50, 50, 50),
                         (bar_x, bar_y, int(bar_width), int(bar_height)))

        health_width = int(bar_width * health_ratio)
        if health_ratio > 0.6:
            color = config.COLOR_HEALTH_GOOD
        elif health_ratio > 0.3:
            color = config.COLOR_HEALTH_MEDIUM
        else:
            color = config.COLOR_HEALTH_BAD

        pygame.draw.rect(self.screen, color,
                         (bar_x, bar_y, health_width, int(bar_height)))

    def _draw_metrics_panel(self, env, metrics, is_human=False):
        """Draw training metrics panel with organized sections."""
        x = self.panel_x
        y = 10
        line_height = 22
        section_gap = 6

        available_width = self.panel_width
        panel_height = config.SCREEN_HEIGHT - 30

        # Panel background and border
        panel_rect = pygame.Rect(x - 5, y - 5, available_width, panel_height)
        pygame.draw.rect(self.screen, config.COLOR_PANEL_BG, panel_rect)
        pygame.draw.rect(self.screen, config.COLOR_PANEL_BORDER, panel_rect, 2)

        # Create a temporary surface for all metrics content
        # Make it tall enough to hold all content
        temp_surface = pygame.Surface((available_width - 10, 2000), pygame.SRCALPHA)
        temp_surface.fill((0, 0, 0, 0))  # Transparent background
        
        # Draw all content to temporary surface with relative coordinates
        content_y = 0

        # Title
        title_text = "Training Metrics" if not is_human else "Human Gameplay"
        title = self.title_font.render(
            title_text, True, (255, 255, 255) if is_human else config.COLOR_TEXT)
        temp_surface.blit(title, (10, content_y))

        if is_human:
            human_label = self.small_font.render(
                "PLAYER CONTROLLED", True, (100, 200, 255))
            temp_surface.blit(human_label, (10, content_y + 20))
            content_y += 40
        else:
            content_y += 32

        # ═══ SESSION SECTION ═══
        content_y = self._draw_section_header_to_surface(temp_surface, 0, content_y, "SESSION", available_width)
        self._draw_metric_row_to_surface(temp_surface, 0, content_y, "Episode", metrics.get(
            'episode', 0), available_width)
        content_y += line_height
        self._draw_metric_row_to_surface(temp_surface, 0, content_y, "Step", env.current_step, available_width)
        content_y += line_height
        self._draw_metric_row_to_surface(
            temp_surface, 0, content_y, "Phase", f"{env.current_phase + 1}/{config.MAX_PHASES}", available_width)
        content_y += line_height + section_gap

        # ═══ REWARDS SECTION ═══
        content_y = self._draw_section_header_to_surface(temp_surface, 0, content_y, "REWARDS", available_width)
        ep_reward = metrics.get('episode_reward', 0)
        reward_color = (100, 255, 100) if ep_reward >= 0 else (255, 100, 100)
        self._draw_metric_row_to_surface(
            temp_surface, 0, content_y, "Episode", f"{ep_reward:.1f}", available_width, value_color=reward_color)
        content_y += line_height
        total_reward = metrics.get('total_reward', 0)
        total_color = (100, 255, 100) if total_reward >= 0 else (255, 100, 100)
        self._draw_metric_row_to_surface(
            temp_surface, 0, content_y, "Total", f"{total_reward:.1f}", available_width, value_color=total_color)
        content_y += line_height + section_gap

        # ═══ COMBAT SECTION ═══
        content_y = self._draw_section_header_to_surface(temp_surface, 0, content_y, "COMBAT", available_width)
        self._draw_metric_row_to_surface(
            temp_surface, 0, content_y, "Enemies", env.enemies_destroyed, available_width, emoji='⚔️')
        content_y += line_height
        self._draw_metric_row_to_surface(
            temp_surface, 0, content_y, "Spawners", env.spawners_destroyed, available_width, emoji='🎯')
        content_y += line_height
        health_pct = int(env.player.get_health_ratio() * 100)
        health_color = config.COLOR_HEALTH_GOOD if health_pct > 60 else config.COLOR_HEALTH_MEDIUM if health_pct > 30 else config.COLOR_HEALTH_BAD
        self._draw_metric_row_to_surface(
            temp_surface, 0, content_y, "Health", f"{health_pct}%", available_width, emoji='❤️', value_color=health_color)
        content_y += line_height + section_gap

        # ═══ ENTITIES SECTION ═══
        content_y = self._draw_section_header_to_surface(temp_surface, 0, content_y, "ENTITIES", available_width)
        alive_enemies = len([e for e in env.enemies if e.alive])
        self._draw_metric_row_to_surface(temp_surface, 0, content_y, "Enemies", alive_enemies, available_width)
        content_y += line_height
        alive_spawners = len([s for s in env.spawners if s.alive])
        self._draw_metric_row_to_surface(
            temp_surface, 0, content_y, "Spawners", alive_spawners, available_width)
        content_y += line_height
        alive_projectiles = len([p for p in env.projectiles if p.alive])
        self._draw_metric_row_to_surface(temp_surface, 0, content_y, "Projectiles",
                              alive_projectiles, available_width)
        content_y += line_height + section_gap

        # ═══ PLAYER SECTION ═══
        if env.player.alive:
            content_y = self._draw_section_header_to_surface(temp_surface, 0, content_y, "PLAYER", available_width)
            player_speed = np.linalg.norm(env.player.velocity)
            self._draw_metric_row_to_surface(
                temp_surface, 0, content_y, "Speed", f"{player_speed:.1f}", available_width, emoji='🚀')
            content_y += line_height
            self._draw_metric_row_to_surface(
                temp_surface, 0, content_y, "Position", f"({int(env.player.pos[0])},{int(env.player.pos[1])})", available_width, emoji='📍')
            content_y += line_height
            rotation_deg = math.degrees(env.player.rotation) % 360
            self._draw_metric_row_to_surface(
                temp_surface, 0, content_y, "Rotation", f"{rotation_deg:.0f}°", available_width, emoji='🔄')
            content_y += line_height
            self._draw_metric_row_to_surface(
                temp_surface, 0, content_y, "Cooldown", env.player.shoot_cooldown, available_width, emoji='⏱️')
            content_y += line_height + section_gap

        # ═══ MODEL INPUTS SECTION (during evaluation) ═══
        if not is_human and metrics.get('show_inputs', False):
            content_y = self._draw_model_inputs_section_to_surface(temp_surface, 0, content_y, env, available_width, line_height, section_gap)

        # ═══ SYSTEM SECTION ═══
        content_y = self._draw_section_header_to_surface(temp_surface, 0, content_y, "SYSTEM", available_width)
        fps = int(self.clock.get_fps())
        fps_color = (100, 255, 100) if fps >= 55 else (
            255, 200, 100) if fps >= 30 else (255, 100, 100)
        self._draw_metric_row_to_surface(
            temp_surface, 0, content_y, "FPS", fps, available_width, value_color=fps_color)
        content_y += line_height + section_gap

        # ═══ ACTION SECTION (if model output available) ═══
        if not is_human and self.model_output:
            content_y = self._draw_action_section_to_surface(temp_surface, 0, content_y, available_width, line_height)
        elif is_human:
            content_y = self._draw_section_header_to_surface(temp_surface, 0, content_y, "CONTROLS", available_width)
            if self.control_style == 1:
                self._draw_metric_row_to_surface(
                    temp_surface, 0, content_y, "Thrust", "W / UP", available_width)
                content_y += line_height
                self._draw_metric_row_to_surface(
                    temp_surface, 0, content_y, "Rotate L", "A / LEFT", available_width)
                content_y += line_height
                self._draw_metric_row_to_surface(
                    temp_surface, 0, content_y, "Rotate R", "D / RIGHT", available_width)
                content_y += line_height
                self._draw_metric_row_to_surface(temp_surface, 0, content_y, "Shoot", "SPACE", available_width)
            else:
                self._draw_metric_row_to_surface(
                    temp_surface, 0, content_y, "Move", "WASD / ARROWS", available_width)
                content_y += line_height
                self._draw_metric_row_to_surface(temp_surface, 0, content_y, "Shoot", "SPACE", available_width)

            content_y += line_height + section_gap
            self._draw_metric_row_to_surface(temp_surface, 0, content_y, "Return to Menu",
                                  "ESC", available_width)
            content_y += line_height
        
        # Calculate max scroll based on content height
        total_content_height = content_y
        self.max_scroll = max(0, total_content_height - panel_height + 20)
        
        # Clamp scroll offset
        self.scroll_offset = max(0, min(self.scroll_offset, self.max_scroll))
        
        # Blit the visible portion of the temporary surface to the screen
        # Set up clipping rectangle for the panel content area
        clip_rect = pygame.Rect(x, y, available_width - 10, panel_height - 10)
        self.screen.set_clip(clip_rect)
        
        # Draw the scrolled content
        self.screen.blit(temp_surface, (x, y - self.scroll_offset))
        
        # Remove clipping
        self.screen.set_clip(None)
        
        # Draw scroll indicator if content is scrollable
        if self.max_scroll > 0:
            self._draw_scroll_indicator(x, y, available_width, panel_height)
            # Draw scroll hint at bottom if not fully scrolled
            if self.scroll_offset < self.max_scroll:
                hint_text = self.small_font.render("\u2193 Scroll Down", True, (100, 150, 200))
                hint_rect = hint_text.get_rect(center=(x + available_width // 2, y + panel_height - 15))
                # Semi-transparent background
                bg_surf = pygame.Surface((hint_rect.width + 10, hint_rect.height + 4))
                bg_surf.set_alpha(180)
                bg_surf.fill((20, 20, 30))
                self.screen.blit(bg_surf, (hint_rect.x - 5, hint_rect.y - 2))
                self.screen.blit(hint_text, hint_rect)
            # Draw scroll hint at top if scrolled down
            if self.scroll_offset > 0:
                hint_text = self.small_font.render("\u2191 Scroll Up", True, (100, 150, 200))
                hint_rect = hint_text.get_rect(center=(x + available_width // 2, y + 15))
                # Semi-transparent background
                bg_surf = pygame.Surface((hint_rect.width + 10, hint_rect.height + 4))
                bg_surf.set_alpha(180)
                bg_surf.fill((20, 20, 30))
                self.screen.blit(bg_surf, (hint_rect.x - 5, hint_rect.y - 2))
                self.screen.blit(hint_text, hint_rect)

    def _draw_scroll_indicator(self, x, y, width, height):
        """Draw a scroll indicator on the right side of the panel."""
        indicator_x = x + width - 15
        indicator_y = y + 5
        indicator_height = height - 20
        
        # Background track (more visible)
        pygame.draw.rect(self.screen, (60, 60, 70), 
                        (indicator_x, indicator_y, 8, indicator_height), border_radius=4)
        
        # Calculate thumb position and size
        content_ratio = (height - 20) / (self.max_scroll + height - 20)
        thumb_height = max(30, int(indicator_height * content_ratio))
        scroll_ratio = self.scroll_offset / self.max_scroll if self.max_scroll > 0 else 0
        thumb_y = indicator_y + int((indicator_height - thumb_height) * scroll_ratio)
        
        # Scrollbar thumb (more visible with border)
        pygame.draw.rect(self.screen, (120, 140, 160), 
                        (indicator_x, thumb_y, 8, thumb_height), border_radius=4)
        pygame.draw.rect(self.screen, (150, 170, 190), 
                        (indicator_x, thumb_y, 8, thumb_height), 1, border_radius=4)

    def handle_scroll(self, event):
        """Handle mouse wheel scroll events for the metrics panel."""
        if event.type == pygame.MOUSEWHEEL:
            # Scroll up (negative y) or down (positive y)
            scroll_amount = 40  # pixels per scroll notch
            self.scroll_offset -= event.y * scroll_amount
            # Clamping is done in _draw_metrics_panel

    def _draw_section_header(self, x, y, title, width):
        """Draw a section header with subtle line."""
        # Draw separator line
        pygame.draw.line(self.screen, (60, 60, 80),
                         (x + 5, y), (x + width - 15, y), 1)
        y += 8
        # Draw section title
        header = self.section_font.render(title, True, (120, 120, 150))
        self.screen.blit(header, (x + 10, y))
        return y + 18
    
    def _draw_section_header_to_surface(self, surface, x, y, title, width):
        """Draw a section header with subtle line to a given surface."""
        # Draw separator line
        pygame.draw.line(surface, (60, 60, 80),
                         (x + 5, y), (x + width - 15, y), 1)
        y += 8
        # Draw section title
        header = self.section_font.render(title, True, (120, 120, 150))
        surface.blit(header, (x + 10, y))
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
    
    def _draw_metric_row_to_surface(self, surface, x, y, label, value, width, emoji=None, value_color=None):
        """Draw a metric row with optional emoji and right-aligned value to a given surface."""
        label_x = x + 10
        value_x = x + width - 20

        # Draw emoji if provided
        if emoji and self._get_emoji(emoji):
            emoji_surf = self._get_emoji(emoji)
            surface.blit(emoji_surf, (label_x, y + 2))
            label_x += 18

        # Draw label
        label_surface = self.small_font.render(label, True, (160, 160, 180))
        surface.blit(label_surface, (label_x, y))

        # Draw value (right-aligned)
        color = value_color if value_color else config.COLOR_TEXT
        value_surface = self.small_font.render(str(value), True, color)
        value_rect = value_surface.get_rect()
        value_rect.right = value_x
        value_rect.top = y
        surface.blit(value_surface, value_rect)

    def _draw_model_inputs_section(self, x, y, env, available_width, line_height, section_gap):
        """Draw model inputs section showing what the agent observes."""
        y = self._draw_section_header(x, y, "MODEL INPUTS", available_width)
        
        # Player state inputs
        player_speed = np.linalg.norm(env.player.velocity)
        self._draw_metric_row(
            x, y, "Velocity", f"{player_speed:.2f}", available_width)
        y += line_height
        
        rotation_deg = math.degrees(env.player.rotation) % 360
        self._draw_metric_row(
            x, y, "Rotation", f"{rotation_deg:.0f}°", available_width)
        y += line_height
        
        cooldown_ratio = env.player.shoot_cooldown / config.PLAYER_SHOOT_COOLDOWN
        self._draw_metric_row(
            x, y, "Cooldown", f"{cooldown_ratio:.2f}", available_width)
        y += line_height
        
        # Add small separator
        y += 3
        
        # Get the k nearest entities
        nearest_enemies = env._find_k_nearest_entities(env.enemies, k=2)
        nearest_spawners = env._find_k_nearest_entities(env.spawners, k=2)
        
        # Display closest enemy info
        for i, enemy in enumerate(nearest_enemies):
            if enemy:
                from arena.game import utils
                dist = utils.distance(env.player.pos, enemy.pos)
                angle = utils.angle_to_point(env.player.pos, enemy.pos)
                angle_deg = math.degrees(angle) % 360
                
                label = f"Enemy {i+1} Dist" if i > 0 else "Enemy Dist"
                self._draw_metric_row(
                    x, y, label, f"{dist:.1f}", available_width, emoji='👾' if i == 0 else None)
                y += line_height
                
                label = f"Enemy {i+1} Ang" if i > 0 else "Enemy Ang"
                self._draw_metric_row(
                    x, y, label, f"{angle_deg:.0f}°", available_width)
                y += line_height
            else:
                # No enemy at this index
                label = f"Enemy {i+1}" if i > 0 else "Enemy"
                self._draw_metric_row(
                    x, y, label, "None", available_width, value_color=(100, 100, 100))
                y += line_height
        
        # Add small separator
        y += 3
        
        # Display closest spawner info
        for i, spawner in enumerate(nearest_spawners):
            if spawner:
                from arena.game import utils
                dist = utils.distance(env.player.pos, spawner.pos)
                angle = utils.angle_to_point(env.player.pos, spawner.pos)
                angle_deg = math.degrees(angle) % 360
                health_pct = int((spawner.health / spawner.max_health) * 100)
                
                label = f"Spawn {i+1} Dist" if i > 0 else "Spawn Dist"
                self._draw_metric_row(
                    x, y, label, f"{dist:.1f}", available_width, emoji='🎯' if i == 0 else None)
                y += line_height
                
                label = f"Spawn {i+1} Ang" if i > 0 else "Spawn Ang"
                self._draw_metric_row(
                    x, y, label, f"{angle_deg:.0f}°", available_width)
                y += line_height
                
                label = f"Spawn {i+1} HP" if i > 0 else "Spawn HP"
                self._draw_metric_row(
                    x, y, label, f"{health_pct}%", available_width)
                y += line_height
            else:
                # No spawner at this index
                label = f"Spawn {i+1}" if i > 0 else "Spawn"
                self._draw_metric_row(
                    x, y, label, "None", available_width, value_color=(100, 100, 100))
                y += line_height
        
        return y + section_gap
    
    def _draw_model_inputs_section_to_surface(self, surface, x, y, env, available_width, line_height, section_gap):
        """Draw model inputs section showing what the agent observes to a given surface."""
        y = self._draw_section_header_to_surface(surface, x, y, "MODEL INPUTS", available_width)
        
        # Player state inputs
        player_speed = np.linalg.norm(env.player.velocity)
        self._draw_metric_row_to_surface(
            surface, x, y, "Velocity", f"{player_speed:.2f}", available_width)
        y += line_height
        
        rotation_deg = math.degrees(env.player.rotation) % 360
        self._draw_metric_row_to_surface(
            surface, x, y, "Rotation", f"{rotation_deg:.0f}°", available_width)
        y += line_height
        
        cooldown_ratio = env.player.shoot_cooldown / config.PLAYER_SHOOT_COOLDOWN
        self._draw_metric_row_to_surface(
            surface, x, y, "Cooldown", f"{cooldown_ratio:.2f}", available_width)
        y += line_height
        
        # Add small separator
        y += 3
        
        # Get the k nearest entities
        nearest_enemies = env._find_k_nearest_entities(env.enemies, k=2)
        nearest_spawners = env._find_k_nearest_entities(env.spawners, k=2)
        
        # Display closest enemy info
        for i, enemy in enumerate(nearest_enemies):
            if enemy:
                from arena.game import utils
                dist = utils.distance(env.player.pos, enemy.pos)
                angle = utils.angle_to_point(env.player.pos, enemy.pos)
                angle_deg = math.degrees(angle) % 360
                
                label = f"Enemy {i+1} Dist" if i > 0 else "Enemy Dist"
                self._draw_metric_row_to_surface(
                    surface, x, y, label, f"{dist:.1f}", available_width, emoji='👾' if i == 0 else None)
                y += line_height
                
                label = f"Enemy {i+1} Ang" if i > 0 else "Enemy Ang"
                self._draw_metric_row_to_surface(
                    surface, x, y, label, f"{angle_deg:.0f}°", available_width)
                y += line_height
            else:
                # No enemy at this index
                label = f"Enemy {i+1}" if i > 0 else "Enemy"
                self._draw_metric_row_to_surface(
                    surface, x, y, label, "None", available_width, value_color=(100, 100, 100))
                y += line_height
        
        # Add small separator
        y += 3
        
        # Display closest spawner info
        for i, spawner in enumerate(nearest_spawners):
            if spawner:
                from arena.game import utils
                dist = utils.distance(env.player.pos, spawner.pos)
                angle = utils.angle_to_point(env.player.pos, spawner.pos)
                angle_deg = math.degrees(angle) % 360
                health_pct = int((spawner.health / spawner.max_health) * 100)
                
                label = f"Spawn {i+1} Dist" if i > 0 else "Spawn Dist"
                self._draw_metric_row_to_surface(
                    surface, x, y, label, f"{dist:.1f}", available_width, emoji='🎯' if i == 0 else None)
                y += line_height
                
                label = f"Spawn {i+1} Ang" if i > 0 else "Spawn Ang"
                self._draw_metric_row_to_surface(
                    surface, x, y, label, f"{angle_deg:.0f}°", available_width)
                y += line_height
                
                label = f"Spawn {i+1} HP" if i > 0 else "Spawn HP"
                self._draw_metric_row_to_surface(
                    surface, x, y, label, f"{health_pct}%", available_width)
                y += line_height
            else:
                # No spawner at this index
                label = f"Spawn {i+1}" if i > 0 else "Spawn"
                self._draw_metric_row_to_surface(
                    surface, x, y, label, "None", available_width, value_color=(100, 100, 100))
                y += line_height
        
        return y + section_gap

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
                display_probs = (values - v_min) / (v_max -
                                                    v_min) if v_max > v_min else np.zeros_like(values)
            else:
                display_probs = values

            bar_max_width = available_width - 100
            bar_height = 12

            for i, (label, prob) in enumerate(zip(labels, display_probs)):
                is_selected = i == output.action_taken

                # Label (compact)
                label_color = config.COLOR_ACTION_SELECTED if is_selected else (
                    150, 150, 170)
                txt = self.small_font.render(
                    label[:6], True, label_color)  # Truncate long labels
                self.screen.blit(txt, (x + 10, y))

                # Bar background
                bar_x = x + 60
                pygame.draw.rect(self.screen, (40, 40, 60),
                                 (bar_x, y + 2, bar_max_width, bar_height))

                # Probability bar
                bar_width = int(bar_max_width * prob)
                bar_color = config.COLOR_ACTION_BAR_HIGH if is_selected else config.COLOR_ACTION_BAR_LOW
                pygame.draw.rect(self.screen, bar_color,
                                 (bar_x, y + 2, bar_width, bar_height))

                # Selection highlight
                if is_selected:
                    pygame.draw.rect(self.screen, config.COLOR_ACTION_SELECTED,
                                     (bar_x, y + 2, bar_max_width, bar_height), 1)

                # Value text (right-aligned)
                val_text = f"{values[i]:.2f}" if output.is_q_value else f"{int(values[i]*100)}%"
                val_surf = self.small_font.render(
                    val_text, True, config.COLOR_TEXT)
                val_rect = val_surf.get_rect(
                    right=x + available_width - 15, top=y)
                self.screen.blit(val_surf, val_rect)

                y += 18

        # V-Estimate and Confidence on same line
        y += 4
        if output.value is not None:
            v_color = config.COLOR_VALUE_POSITIVE if output.value >= 0 else config.COLOR_VALUE_NEGATIVE
            v_text = self.small_font.render(
                f"V:{output.value:.2f}", True, v_color)
            self.screen.blit(v_text, (x + 10, y))

        if output.entropy is not None:
            n_actions = len(labels)
            max_entropy = math.log(n_actions)
            conf_ratio = 1.0 - min(output.entropy / max_entropy, 1.0)
            e_color = config.COLOR_ENTROPY_HIGH if conf_ratio > 0.7 else config.COLOR_ENTROPY_LOW
            conf_text = self.small_font.render(
                f"Conf:{int(conf_ratio*100)}%", True, e_color)
            self.screen.blit(conf_text, (x + 80, y))

        return y + line_height
    
    def _draw_action_section_to_surface(self, surface, x, y, available_width, line_height):
        """Draw action distribution section to a given surface."""
        output = self.model_output
        if not output:
            return y

        # Section header
        title_text = "ACTION" if not output.is_q_value else "Q-VALUES"
        y = self._draw_section_header_to_surface(surface, x, y, title_text, available_width)

        labels = STYLE_1_LABELS if self.control_style == 1 else STYLE_2_LABELS
        values = output.action_probs if not output.is_q_value else output.q_values

        if values is not None:
            # Normalize Q-values for bar display
            if output.is_q_value:
                v_min, v_max = np.min(values), np.max(values)
                display_probs = (values - v_min) / (v_max -
                                                    v_min) if v_max > v_min else np.zeros_like(values)
            else:
                display_probs = values

            bar_max_width = available_width - 100
            bar_height = 12

            for i, (label, prob) in enumerate(zip(labels, display_probs)):
                is_selected = i == output.action_taken

                # Label (compact)
                label_color = config.COLOR_ACTION_SELECTED if is_selected else (
                    150, 150, 170)
                txt = self.small_font.render(
                    label[:6], True, label_color)  # Truncate long labels
                surface.blit(txt, (x + 10, y))

                # Bar background
                bar_x = x + 60
                pygame.draw.rect(surface, (40, 40, 60),
                                 (bar_x, y + 2, bar_max_width, bar_height))

                # Probability bar
                bar_width = int(bar_max_width * prob)
                bar_color = config.COLOR_ACTION_BAR_HIGH if is_selected else config.COLOR_ACTION_BAR_LOW
                pygame.draw.rect(surface, bar_color,
                                 (bar_x, y + 2, bar_width, bar_height))

                # Selection highlight
                if is_selected:
                    pygame.draw.rect(surface, config.COLOR_ACTION_SELECTED,
                                     (bar_x, y + 2, bar_max_width, bar_height), 1)

                # Value text (right-aligned)
                val_text = f"{values[i]:.2f}" if output.is_q_value else f"{int(values[i]*100)}%"
                val_surf = self.small_font.render(
                    val_text, True, config.COLOR_TEXT)
                val_rect = val_surf.get_rect(
                    right=x + available_width - 15, top=y)
                surface.blit(val_surf, val_rect)

                y += 18

        # V-Estimate and Confidence on same line
        y += 4
        if output.value is not None:
            v_color = config.COLOR_VALUE_POSITIVE if output.value >= 0 else config.COLOR_VALUE_NEGATIVE
            v_text = self.small_font.render(
                f"V:{output.value:.2f}", True, v_color)
            surface.blit(v_text, (x + 10, y))

        if output.entropy is not None:
            n_actions = len(labels)
            max_entropy = math.log(n_actions)
            conf_ratio = 1.0 - min(output.entropy / max_entropy, 1.0)
            e_color = config.COLOR_ENTROPY_HIGH if conf_ratio > 0.7 else config.COLOR_ENTROPY_LOW
            conf_text = self.small_font.render(
                f"Conf:{int(conf_ratio*100)}%", True, e_color)
            surface.blit(conf_text, (x + 80, y))

        return y + line_height

    def _draw_model_output_panel(self, y_start=None):
        """Draw model introspection panel."""
        output = self.model_output
        if not output:
            return

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
                label_color = config.COLOR_ACTION_SELECTED if i == output.action_taken else (
                    180, 180, 180)
                txt = self.small_font.render(label, True, label_color)
                self.screen.blit(txt, (x + 10, y))

                # Bar background
                bar_x = x + 70
                pygame.draw.rect(self.screen, (40, 40, 60),
                                 (bar_x, y + 2, bar_max_width, bar_height))

                # Probability bar
                bar_width = int(bar_max_width * prob)
                color = config.COLOR_ACTION_BAR_HIGH if i == output.action_taken else config.COLOR_ACTION_BAR_LOW
                pygame.draw.rect(self.screen, color,
                                 (bar_x, y + 2, bar_width, bar_height))

                # Highlight if chosen
                if i == output.action_taken:
                    pygame.draw.rect(self.screen, config.COLOR_ACTION_SELECTED,
                                     (bar_x, y + 2, bar_max_width, bar_height), 1)

                # Percentage text
                val_text = f"{values[i]:.2f}" if output.is_q_value else f"{int(values[i]*100)}%"
                val_surf = self.small_font.render(
                    val_text, True, config.COLOR_TEXT)
                self.screen.blit(val_surf, (bar_x + bar_max_width + 5, y))

                y += 24

        # Draw Value and Entropy at bottom
        y = panel_rect.bottom - 60
        pygame.draw.line(self.screen, config.COLOR_PANEL_BORDER,
                         (x, y), (x + self.panel_width - 20, y))
        y += 10

        if output.value is not None:
            v_color = config.COLOR_VALUE_POSITIVE if output.value >= 0 else config.COLOR_VALUE_NEGATIVE
            self._draw_output_field(
                x + 10, y, "V-Estimate:", f"{output.value:.2f}", v_color)

        if output.entropy is not None:
            # Lower entropy = more confident
            # Max entropy = log(n_actions): Style 1 has 5 actions, Style 2 has 6 actions
            n_actions = len(labels)
            max_entropy = math.log(n_actions)  # ln(5)≈1.61, ln(6)≈1.79
            conf_ratio = 1.0 - min(output.entropy / max_entropy, 1.0)
            e_color = config.COLOR_ENTROPY_HIGH if conf_ratio > 0.7 else config.COLOR_ENTROPY_LOW
            self._draw_output_field(
                x + 10, y + 22, "Confidence:", f"{int(conf_ratio*100)}%", e_color)

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
        player_pos = self._spos(env.player.pos)
        nearest_enemy = env._find_nearest_entity(env.enemies)
        if nearest_enemy:
            pygame.draw.line(self.screen, (255, 100, 100, 150),
                             player_pos, self._spos(nearest_enemy.pos), 1)
        nearest_spawner = env._find_nearest_entity(env.spawners)
        if nearest_spawner:
            pygame.draw.line(self.screen, (255, 100, 255, 150),
                             player_pos, self._spos(nearest_spawner.pos), 1)

    def _draw_debug_overlay(self, env):
        """Draw comprehensive debug information overlay."""
        # Draw HP values next to health bars for all entities
        if env.player.alive:
            self._draw_debug_hp(env.player.pos, env.player.radius,
                                env.player.health, env.player.max_health)
            self._draw_debug_velocity(
                env.player.pos, env.player.velocity, (100, 255, 100))
            
            # Draw shooting angle indicator for Style 2
            if env.control_style == 2:
                self._draw_shooting_angle(env.player.pos, env.player.rotation)
            
            # Draw aim lines to spawners showing actual angles needed
            self._draw_aim_lines_to_spawners(env)

        for enemy in env.enemies:
            if enemy.alive:
                self._draw_debug_hp(enemy.pos, enemy.radius,
                                    enemy.health, enemy.max_health, small=True)
                self._draw_debug_velocity(enemy.pos, np.array(
                    [enemy.speed, 0]), (255, 100, 100), scale=2.0)

        for spawner in env.spawners:
            if spawner.alive:
                self._draw_debug_hp(spawner.pos, spawner.radius,
                                    spawner.health, spawner.max_health)
                # Show spawn cooldown
                cooldown_text = f"CD:{spawner.spawn_cooldown}"
                cd_surf = self.small_font.render(
                    cooldown_text, True, (255, 200, 100))
                scaled_spos = self._spos(spawner.pos)
                self.screen.blit(cd_surf, (int(
                    scaled_spos[0] - self._s(15)), int(scaled_spos[1] + self._s(spawner.radius + 20))))
                # Show enemies spawned
                spawned_text = f"Spawned:{spawner.enemies_spawned}"
                sp_surf = self.small_font.render(
                    spawned_text, True, (200, 150, 255))
                self.screen.blit(sp_surf, (int(
                    scaled_spos[0] - self._s(30)), int(scaled_spos[1] + self._s(spawner.radius + 35))))

    def _draw_debug_hp(self, pos, entity_radius, health, max_health, small=False):
        """Draw HP value text next to health bar."""
        hp_text = f"{int(health)}/{int(max_health)}"
        color = (255, 255, 255) if health > max_health * 0.5 else (255,
                                                                   200, 100) if health > max_health * 0.25 else (255, 100, 100)
        font = self.small_font if small else self.font
        hp_surf = font.render(hp_text, True, color)

        # Position to the right of the entity
        scaled_pos = self._spos(pos)
        scaled_radius = self._s(entity_radius)
        text_x = int(scaled_pos[0] + scaled_radius + self._s(5))
        text_y = int(scaled_pos[1] - scaled_radius - self._s(10))

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
            scaled_pos = self._spos(pos)
            scaled_vel = velocity * config.RENDER_SCALE * scale
            end_pos = (scaled_pos[0] + scaled_vel[0],
                       scaled_pos[1] + scaled_vel[1])

            pygame.draw.line(self.screen, color,
                             (int(scaled_pos[0]), int(scaled_pos[1])),
                             (int(end_pos[0]), int(end_pos[1])), 2)
            # Draw arrowhead
            angle = math.atan2(velocity[1], velocity[0])
            arrow_size = self._s(5)
            left_angle = angle + 2.5
            right_angle = angle - 2.5
            left_point = (int(end_pos[0] - arrow_size * math.cos(left_angle)),
                          int(end_pos[1] - arrow_size * math.sin(left_angle)))
            right_point = (int(end_pos[0] - arrow_size * math.cos(right_angle)),
                           int(end_pos[1] - arrow_size * math.sin(right_angle)))
            pygame.draw.polygon(self.screen, color,
                                [(int(end_pos[0]), int(end_pos[1])), left_point, right_point])

    def _draw_shooting_angle(self, pos, angle):
        """Draw shooting angle indicator for Style 2 (fixed shooting direction)."""
        scaled_pos = self._spos(pos)
        arrow_length = self._s(60)  # Length of the shooting direction arrow
        
        # Calculate end point
        end_x = scaled_pos[0] + math.cos(angle) * arrow_length
        end_y = scaled_pos[1] + math.sin(angle) * arrow_length
        
        # Draw main line (cyan color to distinguish from velocity)
        color = (0, 255, 255)  # Cyan
        pygame.draw.line(self.screen, color,
                         (int(scaled_pos[0]), int(scaled_pos[1])),
                         (int(end_x), int(end_y)), 3)
        
        # Draw arrowhead
        arrow_size = self._s(10)
        left_angle = angle + 2.5
        right_angle = angle - 2.5
        left_point = (int(end_x - arrow_size * math.cos(left_angle)),
                      int(end_y - arrow_size * math.sin(left_angle)))
        right_point = (int(end_x - arrow_size * math.cos(right_angle)),
                       int(end_y - arrow_size * math.sin(right_angle)))
        pygame.draw.polygon(self.screen, color,
                            [(int(end_x), int(end_y)), left_point, right_point])
        
        # Draw angle label with MORE PRECISION
        angle_deg = math.degrees(angle) % 360
        angle_text = f"Shoot: {angle_deg:.2f}°"  # Show 2 decimal places
        text_surf = self.small_font.render(angle_text, True, color)
        text_x = int(scaled_pos[0] + math.cos(angle) * (arrow_length + self._s(15)))
        text_y = int(scaled_pos[1] + math.sin(angle) * (arrow_length + self._s(15)))
        self.screen.blit(text_surf, (text_x, text_y))
    
    def _draw_aim_lines_to_spawners(self, env):
        """Draw lines from player to spawners showing required aim angles."""
        from arena.game import utils
        
        if not env.player.alive:
            return
        
        player_pos_scaled = self._spos(env.player.pos)
        
        for i, spawner in enumerate(env.spawners):
            if not spawner.alive:
                continue
            
            spawner_pos_scaled = self._spos(spawner.pos)
            
            # Calculate actual angle needed to hit this spawner
            actual_angle = utils.angle_to_point(env.player.pos, spawner.pos)
            
            # Draw thin line to spawner (yellow)
            pygame.draw.line(self.screen, (255, 255, 0, 128),
                           player_pos_scaled, spawner_pos_scaled, 1)
            
            # Calculate angle difference from shooting angle
            if env.control_style == 2:
                angle_diff = actual_angle - env.player.rotation
                # Normalize to -pi to pi
                while angle_diff > math.pi:
                    angle_diff -= 2 * math.pi
                while angle_diff < -math.pi:
                    angle_diff += 2 * math.pi
                
                # Draw angle info near the spawner
                actual_deg = math.degrees(actual_angle) % 360
                diff_deg = math.degrees(angle_diff)
                shoot_deg = math.degrees(env.player.rotation) % 360
                
                info_text = f"Tgt:{actual_deg:.1f}° | Δ:{diff_deg:.1f}°"
                text_surf = self.small_font.render(info_text, True, (255, 255, 100))
                text_pos = (int(spawner_pos_scaled[0] - 60), 
                          int(spawner_pos_scaled[1] - self._s(spawner.radius + 50)))
                
                # Draw background for text
                bg_rect = text_surf.get_rect()
                bg_rect.topleft = text_pos
                bg_surface = pygame.Surface((bg_rect.width + 4, bg_rect.height + 2))
                bg_surface.set_alpha(200)
                bg_surface.fill((0, 0, 0))
                self.screen.blit(bg_surface, (text_pos[0] - 2, text_pos[1] - 1))
                
                self.screen.blit(text_surf, text_pos)

    def _draw_debug_line(self, x, y, label, value):
        """Draw a single debug info line."""
        label_surf = self.small_font.render(label, True, (180, 180, 180))
        value_surf = self.small_font.render(str(value), True, (255, 255, 255))
        self.screen.blit(label_surf, (x, y))
        self.screen.blit(value_surf, (x + 120, y))

    def draw_victory_screen(self, win_step, episode_reward, current_phase):
        """Draw centered victory overlay when all phases are completed."""
        # Semi-transparent dark overlay
        overlay = pygame.Surface((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
        overlay.set_alpha(220)
        overlay.fill((0, 0, 20))
        self.screen.blit(overlay, (0, 0))
        
        # Main victory message
        victory_text = self.title_font.render("🎉 YOU WIN! 🎉", True, (255, 215, 0))
        victory_rect = victory_text.get_rect(center=(config.SCREEN_WIDTH // 2, config.SCREEN_HEIGHT // 2 - 80))
        self.screen.blit(victory_text, victory_rect)
        
        # All phases completed message
        phase_text = self.font.render(f"All {config.MAX_PHASES} Phases Completed!", True, (100, 255, 100))
        phase_rect = phase_text.get_rect(center=(config.SCREEN_WIDTH // 2, config.SCREEN_HEIGHT // 2 - 40))
        self.screen.blit(phase_text, phase_rect)
        
        # Stats
        stats_y = config.SCREEN_HEIGHT // 2 + 10
        
        # Steps taken
        steps_text = self.small_font.render(f"Steps: {win_step:,}", True, (200, 200, 255))
        steps_rect = steps_text.get_rect(center=(config.SCREEN_WIDTH // 2, stats_y))
        self.screen.blit(steps_text, steps_rect)
        
        # Final reward
        reward_text = self.small_font.render(f"Total Reward: {episode_reward:.1f}", True, (200, 200, 255))
        reward_rect = reward_text.get_rect(center=(config.SCREEN_WIDTH // 2, stats_y + 30))
        self.screen.blit(reward_text, reward_rect)
        
        # Continue instruction
        continue_text = self.small_font.render("Press SPACE to continue or ESC for menu", True, (150, 150, 150))
        continue_rect = continue_text.get_rect(center=(config.SCREEN_WIDTH // 2, config.SCREEN_HEIGHT // 2 + 100))
        self.screen.blit(continue_text, continue_rect)

    def render_menu(self, menu):
        """Render selection menu."""
        menu.render()
        pygame.display.flip()
        self.clock.tick(config.FPS)

    def close(self):
        pygame.quit()
