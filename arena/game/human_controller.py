"""
Handle human player keyboard input for the Arena environment.
"""

import pygame

class HumanController:
    """Translates keyboard input to game actions based on control style."""
    
    def __init__(self, style=1):
        self.style = style
        
    def get_action(self, events):
        """
        Processes events and returns the action index.
        Returns -1 if ESC is pressed (signal to return to menu).
        """
        keys = pygame.key.get_pressed()
        
        # Style 1: Rotation + Thrust
        # Actions: 0: Idle, 1: Thrust, 2: Rotate Left, 3: Rotate Right, 4: Shoot
        if self.style == 1:
            if keys[pygame.K_w] or keys[pygame.K_UP]:
                return 1
            if keys[pygame.K_a] or keys[pygame.K_LEFT]:
                return 2
            if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                return 3
            if keys[pygame.K_SPACE]:
                return 4
            return 0
            
        # Style 2: Directional Movement
        # Actions: 0: Idle, 1: Up, 2: Down, 3: Left, 4: Right, 5: Shoot
        else:
            if keys[pygame.K_w] or keys[pygame.K_UP]:
                return 1
            if keys[pygame.K_s] or keys[pygame.K_DOWN]:
                return 2
            if keys[pygame.K_a] or keys[pygame.K_LEFT]:
                return 3
            if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                return 4
            if keys[pygame.K_SPACE]:
                return 5
            return 0

