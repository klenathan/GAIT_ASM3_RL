"""
In-game menu for evaluation configuration.
"""

import pygame
import os
from arena.core import config
from arena.training.registry import AlgorithmRegistry
from arena.training.algorithms import dqn, ppo, ppo_lstm, a2c

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
        
        self.models = self._scan_models()
        self.selected_model_idx = 0
        self.scroll_offset = 0
        
        self.styles = [1, 2]
        self.selected_style_idx = 0
        
        self.deterministic = True
        
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
        
    def _scan_models(self):
        """Scan models directory for .zip files."""
        models = []
        if not os.path.exists(self.models_dir):
            return []
        
        for algo in self.algos:
            for style in [1, 2]:
                style_dir = os.path.join(self.models_dir, algo, f"style{style}")
                if not os.path.exists(style_dir):
                    continue
                for f in os.listdir(style_dir):
                    if f.endswith(".zip"):
                        full_path = os.path.join(style_dir, f)
                        mtime = os.path.getmtime(full_path)
                        models.append({
                            "name": f[:-4],
                            "path": full_path,
                            "algo": algo,
                            "style": style,
                            "mtime": mtime,
                        })
        
        models.sort(key=lambda x: x["mtime"], reverse=True)
        return models
    
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
                    if self.sidebar_x <= x <= self.sidebar_x + 150 and 500 <= y <= 550:
                        self.pressed_button = "start"
                    
                    self.handle_sidebar_clicks(event.pos)
                    action = self._handle_click(event.pos)
                    if action is not None:
                        return action
                elif event.button == 4:
                    self.scroll_offset = max(0, self.scroll_offset - 1)
                elif event.button == 5:
                    self.scroll_offset = min(max(0, len(self.models) - 10), self.scroll_offset + 1)
            
            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.pressed_button = None
            
            if event.type == pygame.MOUSEMOTION:
                self._update_hover_state(event.pos)
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.selected_model_idx = max(0, self.selected_model_idx - 1)
                    if self.selected_model_idx < self.scroll_offset:
                        self.scroll_offset = self.selected_model_idx
                elif event.key == pygame.K_DOWN:
                    self.selected_model_idx = min(len(self.models) - 1, self.selected_model_idx + 1)
                    if self.selected_model_idx >= self.scroll_offset + 10:
                        self.scroll_offset = self.selected_model_idx - 9
                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
                    if len(self.models) == 0:
                        self.set_status("No models available.", "warning", duration=4000)
                        return None
                    return "start"
                elif event.key == pygame.K_ESCAPE:
                    return "quit"
        
        return None

    def _handle_click(self, pos):
        """Handle mouse clicks on UI elements."""
        x, y = pos
        if self.list_x <= x <= self.list_x + self.list_width:
            item_idx = (y - self.list_y) // self.item_height + self.scroll_offset
            if 0 <= item_idx < len(self.models):
                if self.list_y <= y <= self.list_y + self.list_height:
                    self.selected_model_idx = item_idx
        
        if self.sidebar_x <= x <= self.sidebar_x + 150 and 500 <= y <= 550:
            if len(self.models) == 0:
                self.set_status("No models available.", "warning", duration=4000)
                return None
            return "start"

    def render(self):
        """Draw the menu."""
        self.screen.fill(config.COLOR_BACKGROUND)
        
        title = self.title_font.render("Arena Evaluation Menu", True, (255, 255, 255))
        self.screen.blit(title, (50, 30))
        
        if len(self.models) == 0:
            warning_text = self.font.render("No models found", True, (255, 180, 50))
            self.screen.blit(warning_text, (50, 520))
        
        pygame.draw.rect(self.screen, (30, 30, 40), (self.list_x, self.list_y, self.list_width, self.list_height))
        pygame.draw.rect(self.screen, (100, 100, 150), (self.list_x, self.list_y, self.list_width, self.list_height), 2)
        
        visible_count = 11
        for i in range(min(len(self.models), visible_count)):
            idx = i + self.scroll_offset
            if idx >= len(self.models): break
            
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
                pygame.draw.rect(self.screen, bg_color, 
                               (self.list_x + 2, self.list_y + i * self.item_height + 2, 
                                self.list_width - 4, self.item_height - 2))
            
            model = self.models[idx]
            display_text = f"[{model['algo'].upper()}|S{model['style']}] {model['name']}"
            if self.font.size(display_text)[0] > self.list_width - 40:
                display_text = display_text[:45] + "..."
            text = self.font.render(display_text, True, text_color)
            self.screen.blit(text, (self.list_x + 10, self.list_y + i * self.item_height + 5))
        
        if len(self.models) > visible_count:
            scrollbar_x = self.list_x + self.list_width - 15
            scrollbar_y = self.list_y + 5
            scrollbar_height = self.list_height - 10
            pygame.draw.rect(self.screen, (40, 40, 50), (scrollbar_x, scrollbar_y, 10, scrollbar_height))
            
            thumb_height = max(20, scrollbar_height * visible_count // len(self.models))
            thumb_y = scrollbar_y + (scrollbar_height - thumb_height) * self.scroll_offset // max(1, len(self.models) - visible_count)
            pygame.draw.rect(self.screen, (100, 100, 150), (scrollbar_x, int(thumb_y), 10, int(thumb_height)))
            
        y = self.sidebar_y
        self._draw_option_toggle(self.sidebar_x, y, "Algorithm:", self.algos[self.selected_algo_idx], 
                                self.selected_algo_idx, len(self.algos), hovered=(self.hovered_button == "algo"))
        y += 60
        self._draw_option_toggle(self.sidebar_x, y, "Control Style:", f"Style {self.styles[self.selected_style_idx]}", 
                                self.selected_style_idx, len(self.styles), hovered=(self.hovered_button == "style"))
        y += 60
        det_text = "Yes" if self.deterministic else "No"
        self._draw_option_toggle(self.sidebar_x, y, "Deterministic:", det_text, 
                                0 if self.deterministic else 1, 2, hovered=(self.hovered_button == "deterministic"))
        y += 80
        
        start_btn_rect = pygame.Rect(self.sidebar_x, 500, 150, 50)
        button_enabled = len(self.models) > 0
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
        
        instr = self.small_font.render("Arrow Keys/Mouse to select. ENTER to start. ESC to quit.", True, (150, 150, 150))
        self.screen.blit(instr, (50, 560))
        
        if self.status_message:
            self._draw_status_banner()

    def _draw_option_toggle(self, x, y, label, value, current, total, hovered=False):
        """Draw an option toggle."""
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
        
        pygame.draw.rect(self.screen, (30, 30, 80), (banner_x, banner_y, banner_width, banner_height))
        pygame.draw.rect(self.screen, (150, 150, 255), (banner_x, banner_y, banner_width, banner_height), 2)
        
        text_surf = self.font.render(self.status_message, True, (255, 255, 255))
        self.screen.blit(text_surf, (banner_x + 20, banner_y + 15))
    
    def _draw_loading_spinner(self, x, y, radius=15):
        """Draw animated loading spinner."""
        import math
        pygame.draw.circle(self.screen, (50, 50, 80), (int(x), int(y)), radius, 2)
        arc_length = 120
        start_angle = math.radians(self.loading_angle)
        end_angle = math.radians(self.loading_angle + arc_length)
        num_segments = 10
        for i in range(num_segments):
            angle1 = start_angle + (end_angle - start_angle) * i / num_segments
            angle2 = start_angle + (end_angle - start_angle) * (i + 1) / num_segments
            x1 = x + radius * math.cos(angle1)
            y1 = y + radius * math.sin(angle1)
            x2 = x + radius * math.cos(angle2)
            y2 = y + radius * math.sin(angle2)
            pygame.draw.line(self.screen, (100, 200, 255), (int(x1), int(y1)), (int(x2), int(y2)), 3)

    def get_selection(self):
        if not self.models:
            return None
        model = self.models[self.selected_model_idx]
        return {
            "model": model["path"],
            "algo": model["algo"],
            "style": model["style"],
            "deterministic": self.deterministic
        }
    
    def handle_sidebar_clicks(self, pos):
        x, y = pos
        if not (self.sidebar_x <= x <= self.sidebar_x + 200):
            return
        y_rel = y - self.sidebar_y
        if 0 <= y_rel <= 40:
            self.selected_algo_idx = (self.selected_algo_idx + 1) % len(self.algos)
        elif 60 <= y_rel <= 100:
            self.selected_style_idx = (self.selected_style_idx + 1) % len(self.styles)
        elif 120 <= y_rel <= 160:
            self.deterministic = not self.deterministic
    
    def set_status(self, message, status_type='info', duration=3000):
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
        if self.list_x <= x <= self.list_x + self.list_width:
            if self.list_y <= y <= self.list_y + self.list_height:
                item_idx = (y - self.list_y) // self.item_height + self.scroll_offset
                if 0 <= item_idx < len(self.models):
                    self.hovered_model_idx = item_idx
        if self.sidebar_x <= x <= self.sidebar_x + 150 and 500 <= y <= 550:
            if len(self.models) > 0:
                self.hovered_button = "start"
        if self.sidebar_x <= x <= self.sidebar_x + 200:
            y_rel = y - self.sidebar_y
            if 0 <= y_rel <= 40:
                self.hovered_button = "algo"
            elif 60 <= y_rel <= 100:
                self.hovered_button = "style"
            elif 120 <= y_rel <= 160:
                self.hovered_button = "deterministic"
