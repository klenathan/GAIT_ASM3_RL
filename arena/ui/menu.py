"""
In-game menu for evaluation configuration.
"""

import os
import warnings

import pygame

from arena.core import config
from arena.core.curriculum import CurriculumConfig, CurriculumManager
from arena.training.algorithms import dqn, ppo, ppo_lstm
from arena.training.registry import AlgorithmRegistry
from arena.training.training_state import find_training_state

# Suppress pkg_resources deprecation warning from pygame
warnings.filterwarnings(
    "ignore", message=".*pkg_resources is deprecated.*", category=UserWarning
)


class Menu:
    """In-game menu for evaluation configuration."""

    def __init__(self, screen):
        self.screen = screen
        self.font = pygame.font.Font(None, 32)
        self.title_font = pygame.font.Font(None, 48)
        self.small_font = pygame.font.Font(None, 24)

        self.models_dir = config.MODEL_SAVE_DIR
        self.algos = AlgorithmRegistry.list_algorithms()
        self.selected_algo_idx = 0

        self.gameplay_modes = ["Model", "Human Player"]
        self.selected_mode_idx = 0

        self.all_models = self._scan_models()  # Store all models
        self.models = []  # Filtered models to display
        self.selected_model_idx = 0
        self.scroll_offset = 0

        self.styles = [1, 2]
        self.selected_style_idx = 0

        self.deterministic = False
        self.show_all_models = False  # Toggle for showing all vs filtered

        # Curriculum options: None (full difficulty), 0-6 (stages), or "auto"
        # Style 1 has stages 0-5, Style 2 has stages 0-5 (6 total stages each)
        self.curriculum_options = [
            "None", "Auto", "0", "1", "2", "3", "4", "5"]
        self.selected_curriculum_idx = 1  # Default to "Auto"

        # Apply initial filter
        self._filter_models()

        # UI Layout
        self.list_x = 50
        self.list_y = 100
        self.list_width = 500
        self.list_height = 400
        self.item_height = 35

        self.sidebar_x = 600
        self.sidebar_y = 100

        # Status message system
        self.status_message = None
        self.status_type = None
        self.status_start_time = 0
        self.status_duration = 3000
        self.loading_angle = 0

        self.hovered_button = None
        self.pressed_button = None
        self.hovered_model_idx = -1

        # Dropdown state
        self.active_dropdown = None  # Which dropdown is currently open
        self.dropdown_hover_idx = -1  # Which item in dropdown is hovered

    def _scan_models(self):
        """Scan both new runs directory and legacy models directory for .zip files."""
        models = []

        # Scan new unified runs directory structure
        # Structure: ./runs/{algo}/style{style}/{run_name}/checkpoints/ and /final/
        runs_dir = config.RUNS_DIR
        if os.path.exists(runs_dir):
            for algo in self.algos:
                algo_dir = os.path.join(runs_dir, algo)
                if not os.path.exists(algo_dir):
                    continue

                for style in [1, 2]:
                    style_dir = os.path.join(algo_dir, f"style{style}")
                    if not os.path.exists(style_dir):
                        continue

                    # Iterate through run directories
                    for run_name in os.listdir(style_dir):
                        run_path = os.path.join(style_dir, run_name)
                        if not os.path.isdir(run_path):
                            continue

                        # Check both checkpoints/ and final/ subdirectories
                        for subdir in ["checkpoints", "final"]:
                            subdir_path = os.path.join(run_path, subdir)
                            if not os.path.exists(subdir_path):
                                continue

                            for f in os.listdir(subdir_path):
                                if f.endswith(".zip"):
                                    full_path = os.path.join(subdir_path, f)
                                    mtime = os.path.getmtime(full_path)
                                    models.append(
                                        {
                                            "name": f[:-4],
                                            "path": full_path,
                                            "algo": algo,
                                            "style": style,
                                            "mtime": mtime,
                                        }
                                    )

        # Also scan legacy models directory for backward compatibility
        legacy_models_dir = config.MODEL_SAVE_DIR
        if os.path.exists(legacy_models_dir):
            for algo in self.algos:
                for style in [1, 2]:
                    style_dir = os.path.join(
                        legacy_models_dir, algo, f"style{style}")
                    if not os.path.exists(style_dir):
                        continue
                    for f in os.listdir(style_dir):
                        if f.endswith(".zip"):
                            full_path = os.path.join(style_dir, f)
                            mtime = os.path.getmtime(full_path)
                            models.append(
                                {
                                    "name": f[:-4],
                                    "path": full_path,
                                    "algo": algo,
                                    "style": style,
                                    "mtime": mtime,
                                }
                            )

        models.sort(key=lambda x: x["mtime"], reverse=True)
        return models

    def _filter_models(self):
        """Filter models based on selected algorithm and style."""
        if self.show_all_models:
            self.models = self.all_models[:]
        else:
            selected_algo = self.algos[self.selected_algo_idx]
            selected_style = self.styles[self.selected_style_idx]
            self.models = [
                m
                for m in self.all_models
                if m["algo"] == selected_algo and m["style"] == selected_style
            ]

        # Reset selection when filter changes
        self.selected_model_idx = 0
        self.scroll_offset = 0

    def refresh_models(self):
        """Rescan models directory and reapply filters."""
        self.all_models = self._scan_models()
        self._filter_models()
        self.set_status(
            f"Refreshed: Found {len(self.all_models)} models", "info", duration=2000
        )

    def update(self, events):
        """Handle menu events."""
        if self.status_type == "loading":
            self.loading_angle = (self.loading_angle + 10) % 360

        if self.status_message and self.status_duration is not None:
            current_time = pygame.time.get_ticks()
            if current_time - self.status_start_time > self.status_duration:
                self.clear_status()

        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    x, y = event.pos

                    # Check if clicking outside dropdown to close it
                    if self.active_dropdown:
                        dropdown_handled = self._handle_dropdown_click(
                            event.pos)
                        if not dropdown_handled:
                            self.active_dropdown = None
                        continue  # Don't handle other clicks while dropdown is open

                    if self.sidebar_x <= x <= self.sidebar_x + 150 and 500 <= y <= 550:
                        self.pressed_button = "start"

                    self.handle_sidebar_clicks(event.pos)
                    action = self._handle_click(event.pos)
                    if action is not None:
                        return action
                elif event.button == 4:
                    self.scroll_offset = max(0, self.scroll_offset - 1)
                elif event.button == 5:
                    self.scroll_offset = min(
                        max(0, len(self.models) - 10), self.scroll_offset + 1
                    )

            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.pressed_button = None

            if event.type == pygame.MOUSEMOTION:
                self._update_hover_state(event.pos)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.selected_model_idx = max(
                        0, self.selected_model_idx - 1)
                    if self.selected_model_idx < self.scroll_offset:
                        self.scroll_offset = self.selected_model_idx
                elif event.key == pygame.K_DOWN:
                    self.selected_model_idx = min(
                        len(self.models) - 1, self.selected_model_idx + 1
                    )
                    if self.selected_model_idx >= self.scroll_offset + 10:
                        self.scroll_offset = self.selected_model_idx - 9
                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
                    is_human = (
                        self.gameplay_modes[self.selected_mode_idx] == "Human Player"
                    )
                    if not is_human and len(self.models) == 0:
                        self.set_status(
                            "No models available.", "warning", duration=4000
                        )
                        return None
                    return "start"
                elif event.key == pygame.K_ESCAPE:
                    if self.active_dropdown:
                        self.active_dropdown = None
                    else:
                        return "quit"

        return None

    def _handle_click(self, pos):
        """Handle mouse clicks on UI elements."""
        x, y = pos
        if self.list_x <= x <= self.list_x + self.list_width:
            item_idx = (y - self.list_y) // self.item_height + \
                self.scroll_offset
            if 0 <= item_idx < len(self.models):
                if self.list_y <= y <= self.list_y + self.list_height:
                    self.selected_model_idx = item_idx

        if self.sidebar_x <= x <= self.sidebar_x + 150 and 500 <= y <= 550:
            is_human = self.gameplay_modes[self.selected_mode_idx] == "Human Player"
            if not is_human and len(self.models) == 0:
                self.set_status("No models available.",
                                "warning", duration=4000)
                return None
            return "start"

    def render(self):
        """Draw the menu."""
        self.screen.fill(config.COLOR_BACKGROUND)

        title = self.title_font.render(
            "Arena Evaluation Menu", True, (255, 255, 255))
        self.screen.blit(title, (50, 30))

        is_human = self.gameplay_modes[self.selected_mode_idx] == "Human Player"

        if not is_human and len(self.models) == 0:
            warning_text = self.font.render(
                "No models found", True, (255, 180, 50))
            self.screen.blit(warning_text, (50, 520))

        pygame.draw.rect(
            self.screen,
            (30, 30, 40),
            (self.list_x, self.list_y, self.list_width, self.list_height),
        )
        pygame.draw.rect(
            self.screen,
            (100, 100, 150),
            (self.list_x, self.list_y, self.list_width, self.list_height),
            2,
        )

        if not is_human:
            visible_count = 11
            for i in range(min(len(self.models), visible_count)):
                idx = i + self.scroll_offset
                if idx >= len(self.models):
                    break

                is_selected = idx == self.selected_model_idx
                is_hovered = idx == self.hovered_model_idx

                if is_selected:
                    bg_color = (60, 60, 100)
                    text_color = (255, 255, 255)
                elif is_hovered:
                    bg_color = (50, 50, 80)
                    text_color = (220, 220, 220)
                else:
                    bg_color = None
                    text_color = (150, 150, 150)

                if bg_color:
                    pygame.draw.rect(
                        self.screen,
                        bg_color,
                        (
                            self.list_x + 2,
                            self.list_y + i * self.item_height + 2,
                            self.list_width - 4,
                            self.item_height - 2,
                        ),
                    )

                model = self.models[idx]
                display_text = (
                    f"[{model['algo'].upper()}|S{model['style']}] {model['name']}"
                )
                max_text_width = (
                    self.list_width - 40
                )  # Account for padding and scrollbar
                if self.font.size(display_text)[0] > max_text_width:
                    # Dynamically truncate to fit within available width
                    while (
                        self.font.size(
                            display_text + "...")[0] > max_text_width
                        and len(display_text) > 10
                    ):
                        display_text = display_text[:-1]
                    display_text = display_text.rstrip() + "..."
                text = self.font.render(display_text, True, text_color)
                self.screen.blit(
                    text, (self.list_x + 10, self.list_y +
                           i * self.item_height + 5)
                )

            if len(self.models) > visible_count:
                scrollbar_x = self.list_x + self.list_width - 15
                scrollbar_y = self.list_y + 5
                scrollbar_height = self.list_height - 10
                pygame.draw.rect(
                    self.screen,
                    (40, 40, 50),
                    (scrollbar_x, scrollbar_y, 10, scrollbar_height),
                )

                thumb_height = max(
                    20, scrollbar_height * visible_count // len(self.models)
                )
                thumb_y = scrollbar_y + (
                    scrollbar_height - thumb_height
                ) * self.scroll_offset // max(1, len(self.models) - visible_count)
                pygame.draw.rect(
                    self.screen,
                    (100, 100, 150),
                    (scrollbar_x, int(thumb_y), 10, int(thumb_height)),
                )
        else:
            # Human Player Mode - Show helpful text in the list area
            help_title = self.font.render(
                "Human Player Mode Selected", True, (100, 200, 255)
            )
            self.screen.blit(help_title, (self.list_x + 20, self.list_y + 30))

            controls_y = self.list_y + 80
            lines = [
                "Manual control enabled.",
                "Select Control Style on the right.",
                "",
                "Style 1 (Rot/Thrust):",
                "  W / UP      - Thrust",
                "  A / LEFT    - Rotate Left",
                "  D / RIGHT   - Rotate Right",
                "  SPACE       - Shoot",
                "",
                "Style 2 (Directional):",
                "  WASD / ARROWS - Move",
                "  SPACE         - Shoot",
            ]
            for line in lines:
                color = (
                    (200, 200, 200) if not line.startswith(
                        "  ") else (150, 150, 150)
                )
                if ":" in line:
                    color = (255, 255, 100)
                text = self.small_font.render(line, True, color)
                self.screen.blit(text, (self.list_x + 40, controls_y))
                controls_y += 25

        y = self.sidebar_y
        self._draw_dropdown_option(
            self.sidebar_x,
            y,
            "Gameplay Mode:",
            self.gameplay_modes,
            self.selected_mode_idx,
            "mode",
        )
        y += 60

        self._draw_dropdown_option(
            self.sidebar_x,
            y,
            "Control Style:",
            [f"Style {s}" for s in self.styles],
            self.selected_style_idx,
            "style",
        )
        y += 60

        if not is_human:
            self._draw_dropdown_option(
                self.sidebar_x,
                y,
                "Algorithm:",
                [algo.upper() for algo in self.algos],
                self.selected_algo_idx,
                "algo",
            )
            y += 60

            self._draw_dropdown_option(
                self.sidebar_x,
                y,
                "Deterministic:",
                ["Yes", "No"],
                0 if self.deterministic else 1,
                "deterministic",
            )
            y += 60

            # Curriculum stage dropdown
            self._draw_dropdown_option(
                self.sidebar_x,
                y,
                "Curriculum:",
                self.curriculum_options,
                self.selected_curriculum_idx,
                "curriculum",
            )
            y += 60

            self._draw_dropdown_option(
                self.sidebar_x,
                y,
                "Display Mode:",
                ["All Models", "Filtered"],
                0 if self.show_all_models else 1,
                "show_all",
            )
            y += 50

            # Draw Refresh Button
            refresh_btn_rect = pygame.Rect(self.sidebar_x, y, 150, 40)
            if self.hovered_button == "refresh":
                btn_color = (70, 130, 200)
                border_color = (255, 255, 255)
            else:
                btn_color = (50, 100, 150)
                border_color = (150, 200, 255)
            pygame.draw.rect(self.screen, btn_color, refresh_btn_rect)
            pygame.draw.rect(self.screen, border_color, refresh_btn_rect, 2)
            refresh_text = self.font.render("↻ Refresh", True, (255, 255, 255))
            self.screen.blit(refresh_text, (self.sidebar_x + 25, y + 8))
            y += 60

            # Model count indicator
            count_text = f"Models: {len(self.models)}/{len(self.all_models)}"
            count_surf = self.small_font.render(
                count_text, True, (150, 200, 255))
            self.screen.blit(count_surf, (self.sidebar_x, y))
            y += 40

        start_btn_rect = pygame.Rect(self.sidebar_x, 500, 150, 50)
        button_enabled = is_human or len(self.models) > 0
        if not button_enabled:
            button_color = (30, 30, 30)
            border_color = (80, 80, 80)
            text_color = (100, 100, 100)
        elif self.pressed_button == "start":
            button_color = (40, 120, 40)
            border_color = (200, 200, 200)
            text_color = (220, 220, 220)
        elif self.hovered_button == "start":
            button_color = (70, 200, 70)
            border_color = (255, 255, 255)
            text_color = (255, 255, 255)
        else:
            button_color = (50, 150, 50)
            border_color = (255, 255, 255)
            text_color = (255, 255, 255)

        pygame.draw.rect(self.screen, button_color, start_btn_rect)
        pygame.draw.rect(self.screen, border_color, start_btn_rect, 2)
        start_text = self.font.render("START", True, text_color)
        self.screen.blit(start_text, (self.sidebar_x + 40, 515))

        if self.status_type == "loading":
            self._draw_loading_spinner(self.sidebar_x + 160, 525, radius=12)

        instr = self.small_font.render(
            "Arrow Keys/Mouse to select. ENTER to start. ESC to quit.",
            True,
            (150, 150, 150),
        )
        self.screen.blit(instr, (50, 560))

        if self.status_message:
            self._draw_status_banner()

        # Draw tooltip for hovered model (drawn last so it appears on top)
        if self.hovered_model_idx >= 0 and self.hovered_model_idx < len(self.models):
            self._draw_model_tooltip()

        # Draw active dropdown on top of everything
        if self.active_dropdown:
            self._draw_active_dropdown()

    def _draw_dropdown_option(self, x, y, label, options, selected_idx, dropdown_id):
        """Draw a dropdown option with label and current value."""
        is_open = self.active_dropdown == dropdown_id
        is_hovered = self.hovered_button == dropdown_id

        # Store dropdown bounds for click detection
        dropdown_width = 240
        dropdown_height = 45

        # Background
        if is_open:
            bg_color = (50, 50, 90)
            border_color = (100, 200, 255)
        elif is_hovered:
            bg_color = (40, 40, 70)
            border_color = (100, 150, 200)
        else:
            bg_color = (30, 30, 50)
            border_color = (80, 80, 120)

        bg_rect = pygame.Rect(x - 5, y - 5, dropdown_width, dropdown_height)
        pygame.draw.rect(self.screen, bg_color, bg_rect)
        pygame.draw.rect(self.screen, border_color, bg_rect, 2)

        # Label
        label_surf = self.small_font.render(label, True, (180, 180, 180))
        self.screen.blit(label_surf, (x, y))

        # Selected value
        value_text = options[selected_idx]
        val_surf = self.font.render(value_text, True, (255, 255, 100))
        self.screen.blit(val_surf, (x, y + 20))

        # Dropdown arrow
        arrow = "▼" if not is_open else "▲"
        arrow_surf = self.small_font.render(arrow, True, (150, 150, 200))
        self.screen.blit(arrow_surf, (x + dropdown_width - 25, y + 15))

    def _draw_active_dropdown(self):
        """Draw the dropdown menu for the currently active dropdown."""
        if not self.active_dropdown:
            return

        # Determine which dropdown and get its options
        dropdown_data = self._get_dropdown_data(self.active_dropdown)
        if not dropdown_data:
            return

        options, selected_idx, y_pos = dropdown_data

        dropdown_width = 240
        item_height = 30
        dropdown_height = len(options) * item_height

        # Position dropdown below the selector
        dropdown_x = self.sidebar_x - 5
        dropdown_y = y_pos + 45

        # Draw dropdown background
        dropdown_rect = pygame.Rect(
            dropdown_x, dropdown_y, dropdown_width, dropdown_height)
        pygame.draw.rect(self.screen, (25, 25, 45), dropdown_rect)
        pygame.draw.rect(self.screen, (100, 150, 255), dropdown_rect, 2)

        # Draw each option
        for i, option in enumerate(options):
            item_y = dropdown_y + i * item_height
            item_rect = pygame.Rect(
                dropdown_x, item_y, dropdown_width, item_height)

            # Highlight selected item
            if i == selected_idx:
                pygame.draw.rect(self.screen, (60, 60, 100), item_rect)
            # Highlight hovered item
            elif i == self.dropdown_hover_idx:
                pygame.draw.rect(self.screen, (45, 45, 80), item_rect)

            # Draw option text
            color = (255, 255, 255) if i == selected_idx else (200, 200, 200)
            text_surf = self.small_font.render(option, True, color)
            self.screen.blit(text_surf, (dropdown_x + 10, item_y + 7))

            # Draw checkmark for selected item
            if i == selected_idx:
                check_surf = self.small_font.render("✓", True, (100, 255, 100))
                self.screen.blit(
                    check_surf, (dropdown_x + dropdown_width - 25, item_y + 7))

    def _get_dropdown_data(self, dropdown_id):
        """Get data for a specific dropdown."""
        is_human = self.gameplay_modes[self.selected_mode_idx] == "Human Player"

        # Calculate y position based on dropdown order
        y_base = self.sidebar_y

        if dropdown_id == "mode":
            return (self.gameplay_modes, self.selected_mode_idx, y_base)
        elif dropdown_id == "style":
            return ([f"Style {s}" for s in self.styles], self.selected_style_idx, y_base + 60)
        elif dropdown_id == "algo" and not is_human:
            return ([algo.upper() for algo in self.algos], self.selected_algo_idx, y_base + 120)
        elif dropdown_id == "deterministic" and not is_human:
            idx = 0 if self.deterministic else 1
            return (["Yes", "No"], idx, y_base + 180)
        elif dropdown_id == "curriculum" and not is_human:
            return (self.curriculum_options, self.selected_curriculum_idx, y_base + 240)
        elif dropdown_id == "show_all" and not is_human:
            idx = 0 if self.show_all_models else 1
            return (["All Models", "Filtered"], idx, y_base + 300)

        return None

    def _handle_dropdown_click(self, pos):
        """Handle clicks on dropdown menu items. Returns True if click was in dropdown."""
        if not self.active_dropdown:
            return False

        dropdown_data = self._get_dropdown_data(self.active_dropdown)
        if not dropdown_data:
            return False

        options, selected_idx, y_pos = dropdown_data

        dropdown_width = 240
        item_height = 30
        dropdown_height = len(options) * item_height

        dropdown_x = self.sidebar_x - 5
        dropdown_y = y_pos + 45

        x, y = pos

        # Check if click is within dropdown bounds
        if dropdown_x <= x <= dropdown_x + dropdown_width and dropdown_y <= y <= dropdown_y + dropdown_height:
            # Calculate which item was clicked
            item_idx = (y - dropdown_y) // item_height
            if 0 <= item_idx < len(options):
                # Update selection based on dropdown type
                self._update_selection(self.active_dropdown, item_idx)
                self.active_dropdown = None
                return True

        # Click outside dropdown
        return False

    def _update_selection(self, dropdown_id, item_idx):
        """Update the selected value for a dropdown."""
        if dropdown_id == "mode":
            self.selected_mode_idx = item_idx
            self._filter_models()
        elif dropdown_id == "style":
            self.selected_style_idx = item_idx
            self._filter_models()
        elif dropdown_id == "algo":
            self.selected_algo_idx = item_idx
            self._filter_models()
        elif dropdown_id == "deterministic":
            self.deterministic = (item_idx == 0)
        elif dropdown_id == "curriculum":
            self.selected_curriculum_idx = item_idx
        elif dropdown_id == "show_all":
            self.show_all_models = (item_idx == 0)
            self._filter_models()

    def _draw_option_toggle(self, x, y, label, value, current, total, hovered=False):
        """Draw an option toggle (deprecated, kept for compatibility)."""
        if hovered:
            bg_rect = pygame.Rect(x - 5, y - 5, 250, 45)
            pygame.draw.rect(self.screen, (40, 40, 60), bg_rect)
            pygame.draw.rect(self.screen, (100, 150, 200), bg_rect, 2)

        label_surf = self.small_font.render(label, True, (180, 180, 180))
        self.screen.blit(label_surf, (x, y))
        val_surf = self.font.render(value, True, (255, 255, 0))
        self.screen.blit(val_surf, (x, y + 20))

    def _draw_status_banner(self):
        """Draw status banner."""
        banner_height = 60
        banner_y = config.SCREEN_HEIGHT - banner_height - 10
        banner_x = 50
        banner_width = config.SCREEN_WIDTH - 100

        pygame.draw.rect(
            self.screen, (30, 30, 80), (banner_x, banner_y,
                                        banner_width, banner_height)
        )
        pygame.draw.rect(
            self.screen,
            (150, 150, 255),
            (banner_x, banner_y, banner_width, banner_height),
            2,
        )

        text_surf = self.font.render(
            self.status_message, True, (255, 255, 255))
        self.screen.blit(text_surf, (banner_x + 20, banner_y + 15))

    def _draw_loading_spinner(self, x, y, radius=15):
        """Draw animated loading spinner."""
        import math

        pygame.draw.circle(self.screen, (50, 50, 80),
                           (int(x), int(y)), radius, 2)
        arc_length = 120
        start_angle = math.radians(self.loading_angle)
        end_angle = math.radians(self.loading_angle + arc_length)
        num_segments = 10
        for i in range(num_segments):
            angle1 = start_angle + (end_angle - start_angle) * i / num_segments
            angle2 = start_angle + \
                (end_angle - start_angle) * (i + 1) / num_segments
            x1 = x + radius * math.cos(angle1)
            y1 = y + radius * math.sin(angle1)
            x2 = x + radius * math.cos(angle2)
            y2 = y + radius * math.sin(angle2)
            pygame.draw.line(
                self.screen, (100, 200, 255), (int(
                    x1), int(y1)), (int(x2), int(y2)), 3
            )

    def _draw_model_tooltip(self):
        """Draw tooltip with full model name when hovering."""
        model = self.models[self.hovered_model_idx]
        full_text = f"[{model['algo'].upper()}|S{model['style']}] {model['name']}"

        # Get mouse position for tooltip placement
        mouse_x, mouse_y = pygame.mouse.get_pos()

        # Calculate tooltip dimensions
        padding = 10
        text_surf = self.small_font.render(full_text, True, (255, 255, 255))
        tooltip_width = text_surf.get_width() + padding * 2
        tooltip_height = text_surf.get_height() + padding * 2

        # Position tooltip near mouse, but keep it on screen
        tooltip_x = mouse_x + 15
        tooltip_y = mouse_y - tooltip_height - 5

        # Adjust if tooltip would go off-screen
        if tooltip_x + tooltip_width > config.SCREEN_WIDTH:
            tooltip_x = config.SCREEN_WIDTH - tooltip_width - 5
        if tooltip_y < 0:
            tooltip_y = mouse_y + 20

        # Draw tooltip background with border
        tooltip_rect = pygame.Rect(
            tooltip_x, tooltip_y, tooltip_width, tooltip_height)
        pygame.draw.rect(self.screen, (20, 20, 35), tooltip_rect)
        pygame.draw.rect(self.screen, (100, 150, 255), tooltip_rect, 2)

        # Draw text
        self.screen.blit(text_surf, (tooltip_x + padding, tooltip_y + padding))

    def get_selection(self):
        is_human = self.gameplay_modes[self.selected_mode_idx] == "Human Player"
        if is_human:
            return {
                "mode": "Human Player",
                "style": self.styles[self.selected_style_idx],
            }

        if not self.models:
            return None
        model = self.models[self.selected_model_idx]

        # Determine curriculum stage
        curriculum_option = self.curriculum_options[self.selected_curriculum_idx]
        if curriculum_option == "None":
            curriculum_stage = None
            auto_curriculum = False
        elif curriculum_option == "Auto":
            curriculum_stage = None
            auto_curriculum = True
        else:
            curriculum_stage = int(curriculum_option)
            auto_curriculum = False

        return {
            "mode": "Model",
            "model": model["path"],
            "algo": model["algo"],
            "style": model["style"],
            "deterministic": self.deterministic,
            "curriculum_stage": curriculum_stage,
            "auto_curriculum": auto_curriculum,
        }

    def handle_sidebar_clicks(self, pos):
        x, y = pos
        if not (self.sidebar_x - 5 <= x <= self.sidebar_x + 250):
            return
        y_rel = y - self.sidebar_y

        is_human = self.gameplay_modes[self.selected_mode_idx] == "Human Player"

        # Open dropdowns instead of cycling values
        if -5 <= y_rel <= 40:
            self.active_dropdown = "mode"
        elif 55 <= y_rel <= 100:
            self.active_dropdown = "style"
        # Only handle these if in Model mode
        elif not is_human:
            if 115 <= y_rel <= 160:
                self.active_dropdown = "algo"
            elif 175 <= y_rel <= 220:
                self.active_dropdown = "deterministic"
            elif 235 <= y_rel <= 280:
                self.active_dropdown = "curriculum"
            elif 295 <= y_rel <= 340:
                self.active_dropdown = "show_all"
            elif 350 <= y_rel <= 390:
                self.refresh_models()
        else:
            # In human mode, refresh button is at 120
            if 120 <= y_rel <= 160:
                self.refresh_models()

    def set_status(self, message, status_type="info", duration=3000):
        self.status_message = message
        self.status_type = status_type
        self.status_start_time = pygame.time.get_ticks()
        self.status_duration = duration
        self.loading_angle = 0

    def clear_status(self):
        self.status_message = None
        self.status_type = None

    def _update_hover_state(self, pos):
        x, y = pos
        self.hovered_button = None
        self.hovered_model_idx = -1
        self.dropdown_hover_idx = -1

        is_human = self.gameplay_modes[self.selected_mode_idx] == "Human Player"

        # Check if hovering over active dropdown items
        if self.active_dropdown:
            dropdown_data = self._get_dropdown_data(self.active_dropdown)
            if dropdown_data:
                options, selected_idx, y_pos = dropdown_data
                dropdown_width = 240
                item_height = 30
                dropdown_x = self.sidebar_x - 5
                dropdown_y = y_pos + 45

                if dropdown_x <= x <= dropdown_x + dropdown_width:
                    if dropdown_y <= y <= dropdown_y + len(options) * item_height:
                        self.dropdown_hover_idx = (
                            y - dropdown_y) // item_height

        if not is_human and self.list_x <= x <= self.list_x + self.list_width:
            if self.list_y <= y <= self.list_y + self.list_height:
                item_idx = (y - self.list_y) // self.item_height + \
                    self.scroll_offset
                if 0 <= item_idx < len(self.models):
                    self.hovered_model_idx = item_idx

        if self.sidebar_x <= x <= self.sidebar_x + 150 and 500 <= y <= 550:
            if is_human or len(self.models) > 0:
                self.hovered_button = "start"

        if self.sidebar_x - 5 <= x <= self.sidebar_x + 240:
            y_rel = y - self.sidebar_y
            if -5 <= y_rel <= 40:
                self.hovered_button = "mode"
            elif 55 <= y_rel <= 100:
                self.hovered_button = "style"

            # Only handle these if in Model mode
            if not is_human:
                if 115 <= y_rel <= 160:
                    self.hovered_button = "algo"
                elif 175 <= y_rel <= 220:
                    self.hovered_button = "deterministic"
                elif 235 <= y_rel <= 280:
                    self.hovered_button = "curriculum"
                elif 295 <= y_rel <= 340:
                    self.hovered_button = "show_all"
                elif 350 <= y_rel <= 390:
                    self.hovered_button = "refresh"
