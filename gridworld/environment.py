import numpy as np
import random
from gridworld.config import *

class GridWorld:
    def __init__(self, level_idx=0):
        self.level_idx = level_idx
        self.width = GRID_WIDTH
        self.height = GRID_HEIGHT
        self.layout = LEVELS[level_idx]
        self.reset()

    def reset(self):
        self.agent_pos = [0, 0]
        self.has_key = False
        self.collected_apples = [] # List of (r, c)
        self.opened_chests = [] # List of (r, c)
        self.visit_counts = {} # For intrinsic reward (s -> count)
        
        # Parse layout
        self.rocks = []
        self.fires = []
        self.apples = []
        self.chests = []
        self.keys = []
        self.monsters = []
        
        for r, row in enumerate(self.layout):
            for c, char in enumerate(row):
                if char == 'R': self.rocks.append((r, c))
                elif char == 'F': self.fires.append((r, c))
                elif char == 'A': self.apples.append((r, c))
                elif char == 'C': self.chests.append((r, c))
                elif char == 'K': self.keys.append((r, c))
                elif char == 'M': self.monsters.append([r, c]) # Mutable list for movement
        
        # Check start pos collision? Assuming (0,0) is safe usually.
        self.done = False
        return self.get_state()

    def get_state(self):
        # State: (agent_r, agent_c, has_key, tuple(sorted_apples_collected), tuple(sorted_chests_opened))
        # This might be too large for Q-table if not careful.
        # For simplified levels, maybe just agent pos is enough?
        # Specification says "Task 1: learn shortest-path".
        
        # Minimal state for basic Q-learning (Level 0, 1):
        if self.level_idx <= 1:
            return tuple(self.agent_pos)
        
        # For levels with objectives, we need to track them.
        # To keep state space manageable, we can use binary flags for specific known items if quantities are small.
        # Or just tuple of sorted remaining items.
        
        # Frozen dict or tuple for hashability
        return (tuple(self.agent_pos), 
                self.has_key, 
                tuple(sorted(self.collected_apples)), 
                tuple(sorted(self.opened_chests)))

    def step(self, action):
        # Actions: 0=Up, 1=Down, 2=Left, 3=Right
        if self.done:
            return self.get_state(), 0, True, {}

        # Intrinsic reward (Level 6)
        state_key = tuple(self.agent_pos)
        self.visit_counts[state_key] = self.visit_counts.get(state_key, 0) + 1
        
        reward = REWARD_STEP
        
        # Move agent
        dr, dc = 0, 0
        if action == 0: dr = -1
        elif action == 1: dr = 1
        elif action == 2: dc = -1
        elif action == 3: dc = 1
        
        next_r = self.agent_pos[0] + dr
        next_c = self.agent_pos[1] + dc
        
        # Boundary check
        if 0 <= next_r < self.height and 0 <= next_c < self.width:
            # Rock check
            if (next_r, next_c) not in self.rocks:
                self.agent_pos = [next_r, next_c]

        # Check hazards
        r, c = self.agent_pos[0], self.agent_pos[1]
        if (r, c) in self.fires:
            return self.get_state(), REWARD_DEATH, True, {"cause": "fire"}
        
        for m in self.monsters:
            if m[0] == r and m[1] == c:
                return self.get_state(), REWARD_DEATH, True, {"cause": "monster"}

        # Collectibles
        if (r, c) in self.apples and (r, c) not in self.collected_apples:
            reward += REWARD_APPLE
            self.collected_apples.append((r, c))
            
        if (r, c) in self.keys and not self.has_key:
            self.has_key = True
            # Maybe small reward for key? Spec says "no reward but allow opening chests"
            
        if (r, c) in self.chests and (r, c) not in self.opened_chests:
            if self.has_key:
                reward += REWARD_CHEST
                self.opened_chests.append((r, c))
            else:
                # Can't open without key, maybe bumping sound or message?
                pass

        # Monster movement (stochastic)
        if self.monsters:
            self.move_monsters()
            # Check collision again after monster move
            for m in self.monsters:
                if m[0] == self.agent_pos[0] and m[1] == self.agent_pos[1]:
                    return self.get_state(), REWARD_DEATH, True, {"cause": "monster"}

        # Check win condition
        # Win if all apples collected and (if chests exist) all chests opened
        all_apples = len(self.collected_apples) == len(self.apples)
        all_chests = len(self.opened_chests) == len(self.chests)
        
        if all_apples and all_chests:
            reward += REWARD_WIN
            self.done = True
            
        return self.get_state(), reward, self.done, {}

    def move_monsters(self):
        # 40% chance to move each monster
        for m in self.monsters:
            if random.random() < 0.4:
                # Simple random move
                move = random.choice([(0,1), (0,-1), (1,0), (-1,0)])
                nr, nc = m[0] + move[0], m[1] + move[1]
                
                # Check bounds and rocks
                if 0 <= nr < self.height and 0 <= nc < self.width:
                    if (nr, nc) not in self.rocks: # Monsters don't walk into rocks
                        # Update monster pos
                        m[0], m[1] = nr, nc
