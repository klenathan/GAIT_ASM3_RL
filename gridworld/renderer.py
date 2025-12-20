import pygame
from gridworld.config import *

class Renderer:
    def __init__(self, title="Gridworld RL"):
        pygame.init()
        self.width = GRID_WIDTH * TILE_SIZE
        self.height = GRID_HEIGHT * TILE_SIZE
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(title)
        self.font = pygame.font.SysFont('Arial', 18)
        self.clock = pygame.time.Clock()

    def draw(self, env, episode=0, reward=0):
        self.screen.fill(COLOR_BG)
        
        # Draw Grid
        for r in range(GRID_HEIGHT):
            for c in range(GRID_WIDTH):
                rect = pygame.Rect(c * TILE_SIZE, r * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                pygame.draw.rect(self.screen, COLOR_GRID, rect, 1)
                
        # Draw Entities
        for r, c in env.rocks:
            self._draw_rect(r, c, COLOR_ROCK)
        for r, c in env.fires:
            self._draw_rect(r, c, COLOR_FIRE)
        for r, c in env.apples:
            if (r, c) not in env.collected_apples:
                self._draw_circle(r, c, COLOR_APPLE)
        for r, c in env.chests:
            if (r, c) in env.opened_chests:
                self._draw_rect(r, c, COLOR_BG) # Empty looking
                pygame.draw.rect(self.screen, COLOR_CHEST, (c*TILE_SIZE+10, r*TILE_SIZE+10, TILE_SIZE-20, TILE_SIZE-20), 2)
            else:
                self._draw_rect(r, c, COLOR_CHEST)
        for r, c in env.keys:
            if not env.has_key: # Key disappears when picked up (simplified visual)
                self._draw_circle(r, c, COLOR_KEY, radius=10)
        
        # Draw Monsters
        for m in env.monsters:
            self._draw_circle(m[0], m[1], COLOR_MONSTER, radius=15)
            
        # Draw Agent
        self._draw_circle(env.agent_pos[0], env.agent_pos[1], COLOR_AGENT)
        
        # Draw Info
        info_text = f"Ep: {episode} | Rew: {reward:.1f}"
        text_surf = self.font.render(info_text, True, (255, 255, 255))
        self.screen.blit(text_surf, (10, 10))

        pygame.display.flip()

    def _draw_rect(self, r, c, color):
        rect = pygame.Rect(c * TILE_SIZE + 2, r * TILE_SIZE + 2, TILE_SIZE - 4, TILE_SIZE - 4)
        pygame.draw.rect(self.screen, color, rect)

    def _draw_circle(self, r, c, color, radius=None):
        cx = c * TILE_SIZE + TILE_SIZE // 2
        cy = r * TILE_SIZE + TILE_SIZE // 2
        if radius is None:
            radius = TILE_SIZE // 2 - 4
        pygame.draw.circle(self.screen, color, (cx, cy), radius)
    
    def process_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True
