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
        
    def _scan_models(self):
        """Scan models directory for .zip files"""
        if not os.path.exists(self.models_dir):
            return []
        
        models = [f for f in os.listdir(self.models_dir) if f.endswith(".zip")]
        # Remove extension for display
        models = [m[:-4] for m in models]
        return sorted(models)
    
    def update(self, events):
        """Handle menu events"""
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: # Left click
                    self._handle_click(event.pos)
                elif event.button == 4: # Scroll up
                    self.scroll_offset = max(0, self.scroll_offset - 1)
                elif event.button == 5: # Scroll down
                    self.scroll_offset = min(max(0, len(self.models) - 10), self.scroll_offset + 1)
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.selected_model_idx = max(0, self.selected_model_idx - 1)
                    if self.selected_model_idx < self.scroll_offset:
                        self.scroll_offset = self.selected_model_idx
                elif event.key == pygame.K_DOWN:
                    self.selected_model_idx = min(len(self.models) - 1, self.selected_model_idx + 1)
                    if self.selected_model_idx >= self.scroll_offset + 10:
                        self.scroll_offset = self.selected_model_idx - 9
                elif event.key == pygame.K_RETURN:
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
             return "start" # Note: update() handles the return value

    def render(self):
        """Draw the menu"""
        self.screen.fill(config.COLOR_BACKGROUND)
        
        # Title
        title = self.title_font.render("Arena Evaluation Menu", True, (255, 255, 255))
        self.screen.blit(title, (50, 30))
        
        # Model List Background
        pygame.draw.rect(self.screen, (30, 30, 40), (self.list_x, self.list_y, self.list_width, self.list_height))
        pygame.draw.rect(self.screen, (100, 100, 150), (self.list_x, self.list_y, self.list_width, self.list_height), 2)
        
        # Display Models
        visible_count = 11
        for i in range(min(len(self.models), visible_count)):
            idx = i + self.scroll_offset
            if idx >= len(self.models): break
            
            color = (255, 255, 255) if idx == self.selected_model_idx else (150, 150, 150)
            if idx == self.selected_model_idx:
                pygame.draw.rect(self.screen, (60, 60, 100), (self.list_x + 2, self.list_y + i * self.item_height + 2, self.list_width - 4, self.item_height - 2))
            
            text = self.font.render(self.models[idx], True, color)
            self.screen.blit(text, (self.list_x + 10, self.list_y + i * self.item_height + 5))
            
        # Sidebar Options
        y = self.sidebar_y
        
        # Algorithm selection
        self._draw_option_toggle(self.sidebar_x, y, "Algorithm:", self.algos[self.selected_algo_idx], self.selected_algo_idx, len(self.algos))
        y += 60
        
        # Style selection
        self._draw_option_toggle(self.sidebar_x, y, "Control Style:", f"Style {self.styles[self.selected_style_idx]}", self.selected_style_idx, len(self.styles))
        y += 60
        
        # Deterministic toggle
        det_text = "Yes" if self.deterministic else "No"
        self._draw_option_toggle(self.sidebar_x, y, "Deterministic:", det_text, 0 if self.deterministic else 1, 2)
        y += 80
        
        # Start Button
        start_btn_rect = pygame.Rect(self.sidebar_x, 500, 150, 50)
        pygame.draw.rect(self.screen, (50, 150, 50), start_btn_rect)
        pygame.draw.rect(self.screen, (255, 255, 255), start_btn_rect, 2)
        start_text = self.font.render("START", True, (255, 255, 255))
        self.screen.blit(start_text, (self.sidebar_x + 40, 515))
        
        # Instructions
        instr = self.small_font.render("Use Arrow Keys/Mouse to select model. ENTER to start. ESC to quit.", True, (150, 150, 150))
        self.screen.blit(instr, (50, 560))

    def _draw_option_toggle(self, x, y, label, value, current, total):
        label_surf = self.small_font.render(label, True, (180, 180, 180))
        self.screen.blit(label_surf, (x, y))
        
        val_surf = self.font.render(value, True, (255, 255, 0))
        self.screen.blit(val_surf, (x, y + 20))
        
    def get_selection(self):
        """Return the current configuration"""
        if not self.models:
            return None
        
        model_path = os.path.join(self.models_dir, self.models[self.selected_model_idx] + ".zip")
        return {
            "model": model_path,
            "algo": self.algos[self.selected_algo_idx],
            "style": self.styles[self.selected_style_idx],
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
