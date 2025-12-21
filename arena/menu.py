import pygame
import os
from arena import config

class Menu:
    """In-game menu for evaluation configuration"""
    
    def __init__(self, screen):
        self.screen = screen
        self.font = pygame.font.Font(None, 32)
        self.title_font = pygame.font.Font(None, 48)
        self.small_font = pygame.font.Font(None, 24)
        
        self.models_dir = "models"
        self.models = self._scan_models()
        self.selected_model_idx = 0
        self.scroll_offset = 0
        
        self.algos = ["ppo", "dqn"]
        self.selected_algo_idx = 0
        
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
        self.status_type = None  # 'error', 'success', 'loading', 'warning', 'info'
        self.status_start_time = 0
        self.status_duration = 3000  # milliseconds
        self.loading_angle = 0  # For spinner animation
        
        # Button hover state
        self.hovered_button = None
        self.pressed_button = None  # Track which button is being pressed
        self.hovered_model_idx = -1  # Track which model is being hovered
        
    def _scan_models(self):
        """Scan models directory for .zip files, organized by algo/style, sorted by modification time"""
        models = []
        if not os.path.exists(self.models_dir):
            return []
        
        # Scan nested structure: models/{algo}/style{N}/
        for algo in ["ppo", "dqn"]:
            for style in [1, 2]:
                style_dir = os.path.join(self.models_dir, algo, f"style{style}")
                if not os.path.exists(style_dir):
                    continue
                for f in os.listdir(style_dir):
                    if f.endswith(".zip"):
                        full_path = os.path.join(style_dir, f)
                        mtime = os.path.getmtime(full_path)
                        models.append({
                            "name": f[:-4],  # Remove .zip extension
                            "path": full_path,
                            "algo": algo,
                            "style": style,
                            "mtime": mtime,
                        })
        
        # Sort by modification time descending (newest first)
        models.sort(key=lambda x: x["mtime"], reverse=True)
        return models
    
    def update(self, events):
        """Handle menu events"""
        # Update loading spinner animation
        if self.status_type == "loading":
            self.loading_angle = (self.loading_angle + 10) % 360
        
        # Check status timeout
        if self.status_message and self.status_duration is not None:
            current_time = pygame.time.get_ticks()
            if current_time - self.status_start_time > self.status_duration:
                self.clear_status()
        
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: # Left click
                    # Track pressed button for visual feedback
                    x, y = event.pos
                    if self.sidebar_x <= x <= self.sidebar_x + 150 and 500 <= y <= 550:
                        self.pressed_button = "start"
                    
                    # First handle sidebar toggles using the actual click position
                    self.handle_sidebar_clicks(event.pos)

                    # Then handle list/start clicks and propagate returned actions
                    action = self._handle_click(event.pos)
                    if action is not None:
                        return action
                elif event.button == 4: # Scroll up
                    self.scroll_offset = max(0, self.scroll_offset - 1)
                elif event.button == 5: # Scroll down
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
                        self.set_status("No models available. Place .zip model files in 'models/' directory.", 
                                      "warning", duration=4000)
                        return None
                    return "start"
                elif event.key == pygame.K_ESCAPE:
                    return "quit"
        
        return None

    def _handle_click(self, pos):
        """Handle mouse clicks on UI elements"""
        x, y = pos
        
        # Check model list clicks
        if self.list_x <= x <= self.list_x + self.list_width:
            item_idx = (y - self.list_y) // self.item_height + self.scroll_offset
            if 0 <= item_idx < len(self.models):
                if self.list_y <= y <= self.list_y + self.list_height:
                    self.selected_model_idx = item_idx
        
        # Check sidebar buttons
        # Start Button
        if self.sidebar_x <= x <= self.sidebar_x + 150 and 500 <= y <= 550:
            if len(self.models) == 0:
                self.set_status("No models available. Place .zip model files in 'models/' directory.", 
                              "warning", duration=4000)
                return None
            return "start" # Note: update() handles the return value

    def render(self):
        """Draw the menu"""
        self.screen.fill(config.COLOR_BACKGROUND)
        
        # Title
        title = self.title_font.render("Arena Evaluation Menu", True, (255, 255, 255))
        self.screen.blit(title, (50, 30))
        
        # Check if no models available and show warning
        if len(self.models) == 0:
            warning_text = self.font.render("No models found in 'models/' directory", True, config.COLOR_STATUS_WARNING)
            self.screen.blit(warning_text, (50, 520))
        
        # Model List Background
        pygame.draw.rect(self.screen, (30, 30, 40), (self.list_x, self.list_y, self.list_width, self.list_height))
        pygame.draw.rect(self.screen, (100, 100, 150), (self.list_x, self.list_y, self.list_width, self.list_height), 2)
        
        # Display Models
        visible_count = 11
        for i in range(min(len(self.models), visible_count)):
            idx = i + self.scroll_offset
            if idx >= len(self.models): break
            
            # Determine colors based on selection and hover state
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
            
            # Draw background
            if bg_color:
                pygame.draw.rect(self.screen, bg_color, 
                               (self.list_x + 2, self.list_y + i * self.item_height + 2, 
                                self.list_width - 4, self.item_height - 2))
            
            # Draw text with algo/style tag
            model = self.models[idx]
            display_text = f"[{model['algo'].upper()}|S{model['style']}] {model['name']}"
            # Truncate if too long
            if self.font.size(display_text)[0] > self.list_width - 40:
                display_text = display_text[:45] + "..."
            text = self.font.render(display_text, True, text_color)
            self.screen.blit(text, (self.list_x + 10, self.list_y + i * self.item_height + 5))
        
        # Draw scroll indicators if needed
        if len(self.models) > visible_count:
            # Scrollbar background
            scrollbar_x = self.list_x + self.list_width - 15
            scrollbar_y = self.list_y + 5
            scrollbar_height = self.list_height - 10
            pygame.draw.rect(self.screen, (40, 40, 50), (scrollbar_x, scrollbar_y, 10, scrollbar_height))
            
            # Scrollbar thumb
            thumb_height = max(20, scrollbar_height * visible_count // len(self.models))
            thumb_y = scrollbar_y + (scrollbar_height - thumb_height) * self.scroll_offset // max(1, len(self.models) - visible_count)
            pygame.draw.rect(self.screen, (100, 100, 150), (scrollbar_x, int(thumb_y), 10, int(thumb_height)))
            
            # Scroll indicators text
            if self.scroll_offset > 0:
                scroll_up_text = self.small_font.render("↑ More above", True, (150, 150, 200))
                self.screen.blit(scroll_up_text, (self.list_x + 10, self.list_y + 5))
            
            if self.scroll_offset + visible_count < len(self.models):
                scroll_down_text = self.small_font.render("↓ More below", True, (150, 150, 200))
                self.screen.blit(scroll_down_text, (self.list_x + 10, self.list_y + self.list_height - 25))
            
        # Sidebar Options
        y = self.sidebar_y
        
        # Algorithm selection
        self._draw_option_toggle(self.sidebar_x, y, "Algorithm:", self.algos[self.selected_algo_idx], 
                                self.selected_algo_idx, len(self.algos), hovered=(self.hovered_button == "algo"))
        y += 60
        
        # Style selection
        self._draw_option_toggle(self.sidebar_x, y, "Control Style:", f"Style {self.styles[self.selected_style_idx]}", 
                                self.selected_style_idx, len(self.styles), hovered=(self.hovered_button == "style"))
        y += 60
        
        # Deterministic toggle
        det_text = "Yes" if self.deterministic else "No"
        self._draw_option_toggle(self.sidebar_x, y, "Deterministic:", det_text, 
                                0 if self.deterministic else 1, 2, hovered=(self.hovered_button == "deterministic"))
        y += 80
        
        # Start Button
        start_btn_rect = pygame.Rect(self.sidebar_x, 500, 150, 50)
        
        # Determine button color and enabled state
        button_enabled = len(self.models) > 0
        if not button_enabled:
            button_color = (30, 30, 30)  # Grayed out
            border_color = (80, 80, 80)
            text_color = (100, 100, 100)
        elif self.pressed_button == "start":
            button_color = (40, 120, 40)  # Darker when pressed
            border_color = (200, 200, 200)
            text_color = (220, 220, 220)
        elif self.hovered_button == "start":
            button_color = (70, 200, 70)  # Brighter on hover
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
        
        # Draw loading spinner next to button if loading
        if self.status_type == "loading":
            self._draw_loading_spinner(self.sidebar_x + 160, 525, radius=12)
        
        # Instructions
        instr = self.small_font.render("Use Arrow Keys/Mouse to select model. ENTER to start. ESC to quit.", True, (150, 150, 150))
        self.screen.blit(instr, (50, 560))
        
        # Draw status banner
        if self.status_message:
            self._draw_status_banner()

    def _draw_option_toggle(self, x, y, label, value, current, total, hovered=False):
        """Draw an option toggle with hover effect"""
        # Draw background box if hovered
        if hovered:
            bg_rect = pygame.Rect(x - 5, y - 5, 250, 45)
            pygame.draw.rect(self.screen, (40, 40, 60), bg_rect)
            pygame.draw.rect(self.screen, (100, 150, 200), bg_rect, 2)
        
        label_color = (220, 220, 220) if hovered else (180, 180, 180)
        value_color = (255, 255, 100) if hovered else (255, 255, 0)
        
        label_surf = self.small_font.render(label, True, label_color)
        self.screen.blit(label_surf, (x, y))
        
        val_surf = self.font.render(value, True, value_color)
        self.screen.blit(val_surf, (x, y + 20))
        
        # Draw click indicator hint if hovered
        if hovered:
            hint_text = self.small_font.render("(click to change)", True, (150, 150, 200))
            self.screen.blit(hint_text, (x + 120, y + 25))
    
    def _draw_status_banner(self):
        """Draw status message banner at the bottom of screen"""
        if not self.status_message:
            return
        
        # Determine colors based on status type
        color_map = {
            'error': (config.COLOR_STATUS_ERROR, config.COLOR_STATUS_BG_ERROR),
            'success': (config.COLOR_STATUS_SUCCESS, config.COLOR_STATUS_BG_SUCCESS),
            'warning': (config.COLOR_STATUS_WARNING, config.COLOR_STATUS_BG_WARNING),
            'loading': (config.COLOR_STATUS_LOADING, config.COLOR_STATUS_BG_LOADING),
            'info': (config.COLOR_STATUS_INFO, config.COLOR_STATUS_BG_INFO),
        }
        
        text_color, bg_color = color_map.get(self.status_type, color_map['info'])
        
        # Calculate fade effect based on remaining time
        alpha = 255
        if self.status_duration is not None:
            elapsed = pygame.time.get_ticks() - self.status_start_time
            remaining = self.status_duration - elapsed
            # Fade out in last 500ms
            if remaining < 500:
                alpha = int(255 * (remaining / 500))
        
        # Banner dimensions
        banner_height = 60
        banner_y = config.SCREEN_HEIGHT - banner_height - 10
        banner_x = 50
        banner_width = config.SCREEN_WIDTH - 100
        
        # Create surface with alpha for fade effect
        banner_surface = pygame.Surface((banner_width, banner_height))
        banner_surface.set_alpha(alpha)
        banner_surface.fill(bg_color)
        
        # Draw banner background
        self.screen.blit(banner_surface, (banner_x, banner_y))
        
        # Draw border
        border_surface = pygame.Surface((banner_width, banner_height))
        border_surface.set_alpha(alpha)
        pygame.draw.rect(border_surface, text_color, (0, 0, banner_width, banner_height), 2)
        self.screen.blit(border_surface, (banner_x, banner_y))
        
        # Draw message text (wrap if too long)
        max_width = banner_width - 60
        message_lines = self._wrap_text(self.status_message, self.font, max_width)
        
        text_y = banner_y + 10
        for line in message_lines:
            text_surf = self.font.render(line, True, text_color)
            text_surf.set_alpha(alpha)
            text_x = banner_x + 30
            self.screen.blit(text_surf, (text_x, text_y))
            text_y += 25
        
        # Draw loading spinner if loading
        if self.status_type == "loading":
            self._draw_loading_spinner(banner_x + 10, banner_y + banner_height // 2, radius=10)
    
    def _draw_loading_spinner(self, x, y, radius=15):
        """Draw animated loading spinner"""
        # Draw rotating arc
        import math
        
        # Draw background circle
        pygame.draw.circle(self.screen, (50, 50, 80), (int(x), int(y)), radius, 2)
        
        # Draw rotating arc
        arc_length = 120  # degrees
        start_angle = math.radians(self.loading_angle)
        end_angle = math.radians(self.loading_angle + arc_length)
        
        # Draw arc using lines
        num_segments = 20
        for i in range(num_segments):
            angle1 = start_angle + (end_angle - start_angle) * i / num_segments
            angle2 = start_angle + (end_angle - start_angle) * (i + 1) / num_segments
            
            x1 = x + radius * math.cos(angle1)
            y1 = y + radius * math.sin(angle1)
            x2 = x + radius * math.cos(angle2)
            y2 = y + radius * math.sin(angle2)
            
            pygame.draw.line(self.screen, config.COLOR_STATUS_LOADING, 
                           (int(x1), int(y1)), (int(x2), int(y2)), 3)
    
    def _wrap_text(self, text, font, max_width):
        """Wrap text to fit within max_width"""
        words = text.split(' ')
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            if font.size(test_line)[0] <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines if lines else [text]
        
    def get_selection(self):
        """Return the current configuration"""
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
        """Additional click handling for sidebar options"""
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
        elif 400 <= y_rel <= 450: # Start button relative to sidebar_y is wrong, but handled in _handle_click
            pass
    
    def set_status(self, message, status_type='info', duration=3000):
        """
        Set a status message with auto-dismiss
        
        Args:
            message: Status message text
            status_type: Type of status ('error', 'success', 'loading', 'warning', 'info')
            duration: Duration in milliseconds (None for persistent, only cleared manually)
        """
        self.status_message = message
        self.status_type = status_type
        self.status_start_time = pygame.time.get_ticks()
        self.status_duration = duration
        self.loading_angle = 0
    
    def clear_status(self):
        """Clear current status message"""
        self.status_message = None
        self.status_type = None
        self.status_start_time = 0
        self.loading_angle = 0
    
    def _update_hover_state(self, pos):
        """Update which button is being hovered over"""
        x, y = pos
        self.hovered_button = None
        self.hovered_model_idx = -1
        
        # Check model list items
        if self.list_x <= x <= self.list_x + self.list_width:
            if self.list_y <= y <= self.list_y + self.list_height:
                item_idx = (y - self.list_y) // self.item_height + self.scroll_offset
                if 0 <= item_idx < len(self.models):
                    self.hovered_model_idx = item_idx
        
        # Check START button
        if self.sidebar_x <= x <= self.sidebar_x + 150 and 500 <= y <= 550:
            if len(self.models) > 0:  # Only highlight if models are available
                self.hovered_button = "start"
        
        # Check sidebar toggles
        if self.sidebar_x <= x <= self.sidebar_x + 200:
            y_rel = y - self.sidebar_y
            if 0 <= y_rel <= 40:
                self.hovered_button = "algo"
            elif 60 <= y_rel <= 100:
                self.hovered_button = "style"
            elif 120 <= y_rel <= 160:
                self.hovered_button = "deterministic"
