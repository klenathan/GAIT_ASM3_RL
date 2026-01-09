# Grid Dimensions
GRID_WIDTH = 10
GRID_HEIGHT = 10
TILE_SIZE = 60

# Colors
COLOR_BG = (30, 30, 30)
COLOR_GRID = (50, 50, 50)
COLOR_AGENT = (0, 200, 255)
COLOR_ROCK = (100, 100, 100)
COLOR_FIRE = (255, 50, 0)
COLOR_APPLE = (0, 255, 0)
COLOR_CHEST = (255, 215, 0)
COLOR_KEY = (0, 0, 255)
COLOR_MONSTER = (200, 0, 200)

# Learning Parameters
ALPHA = 0.1  # Learning rate
GAMMA = 0.99  # Discount factor
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995

# Rewards
REWARD_STEP = -0.01
REWARD_APPLE = 1.0
REWARD_CHEST = 2.0
REWARD_DEATH = -10.0
REWARD_WIN = 10.0  # Bonus for completing all objectives

# Level Layouts
# 0: Empty, 1: Rocks, 2: Fire, 3: Apple, 4: Chest, 5: Key, 6: Monster
# Start position is usually (0, 0) or defined separately

LEVELS = []

LEVEL_0 = [
    "..........",
    "..........",
    "..........",
    "..........",
    "..........",
    "..........",
    "..........",
    "..........",
    "..........",
    "........A.",
]

LEVEL_1 = [
    "..........",
    "..........",
    ".RRRRRRR.R",
    ".R.......R",
    ".R.......R",
    ".R.RRRRRR.",
    ".R........",
    ".RRRRRRR..",
    ".FFFFFFR..",
    "........A.",
]

LEVEL_1_BACKUP = [
    "..........",
    "..........",
    ".RRRRRRRR.",
    "..........",
    "..........",
    ".RRRRRRRR.",
    "..........",
    "..........",
    ".FFFFFFFF.",
    "........A.",
]

# Level 2: Multiple apples, a key, a chest
LEVEL_2 = [
    "..........",
    ".R.R.R.R..",
    ".R.R.R.R..",
    "..A.......",
    "....K.....",
    "..........",
    ".RRRRRRRR.",
    ".A........",
    ".........C",
    "........A.",
]

# Level 3: Same as Level 2 for now, or slightly harder
LEVEL_3 = [
    "..........",
    "..........",
    ".RR.RR.RR.",
    ".K......A.",
    ".RR.RR.RR.",
    "..........",
    ".RR.RR.RR.",
    ".A......C.",
    ".RR.RR.RR.",
    "........A.",
]

# Level 4: Monster introduced (M)
# M should be placed where it can move.
LEVEL_4 = [
    "..........",
    "..........",
    "..M.......",
    "..........",
    "..........",
    ".......M..",
    "..........",
    "..........",
    "..........",
    "........A.",
]

# Level 5: More monsters or harder layout
LEVEL_5 = [
    "..........",
    ".R.R...R..",
    ".R...M.R..",
    ".R.R...R..",
    "..........",
    ".R.R...R..",
    ".R.M...R..",
    ".R.R...R..",
    "..........",
    "........A.",
]

# Level 6: Intrinsic reward - Sparse reward or deceptive
# Same as Level 0 or empty to test exploration, but let's use a layout that requires exploration
LEVEL_6 = [
    "..........",
    ".RRRRRRRR.",
    "..........",
    ".RRRRRRRR.",
    "..........",
    ".RRRRRRRR.",
    "..........",
    ".RRRRRRRR.",
    "..........",
    "........A.",
]

LEVELS = [LEVEL_0, LEVEL_1, LEVEL_2, LEVEL_3, LEVEL_4, LEVEL_5, LEVEL_6]
