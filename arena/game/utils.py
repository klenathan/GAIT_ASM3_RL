"""
Math and physics utilities for Arena game entities.
"""

import math
import numpy as np

def distance(pos1, pos2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

def angle_to_point(from_pos, to_pos):
    """Calculate angle from one point to another in radians."""
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]
    return math.atan2(dy, dx)

def relative_angle(from_angle, to_angle):
    """Calculate relative angle between two angles (-pi to pi)."""
    diff = to_angle - from_angle
    # Normalize to -pi to pi
    while diff > math.pi:
        diff -= 2 * math.pi
    while diff < -math.pi:
        diff += 2 * math.pi
    return diff

def normalize_angle(angle):
    """Normalize angle to 0-1 range from -pi to pi."""
    return (angle + math.pi) / (2 * math.pi)

def clamp(value, min_val, max_val):
    """Clamp value between min and max."""
    return max(min_val, min(max_val, value))

def check_collision(pos1, radius1, pos2, radius2):
    """Check if two circles collide."""
    return distance(pos1, pos2) < (radius1 + radius2)

def keep_in_bounds(pos, min_x, max_x, min_y, max_y, radius):
    """Keep position within bounds, accounting for radius."""
    x = clamp(pos[0], min_x + radius, max_x - radius)
    y = clamp(pos[1], min_y + radius, max_y - radius)
    return np.array([x, y], dtype=np.float32)

def vector_from_angle(angle, magnitude=1.0):
    """Create a 2D vector from an angle and magnitude."""
    return np.array([
        math.cos(angle) * magnitude,
        math.sin(angle) * magnitude
    ], dtype=np.float32)

def magnitude(vector):
    """Calculate magnitude of a 2D vector."""
    return np.sqrt(vector[0]**2 + vector[1]**2)

def normalize_vector(vector):
    """Normalize a 2D vector to unit length."""
    mag = magnitude(vector)
    if mag == 0:
        return np.array([0, 0], dtype=np.float32)
    return vector / mag

def limit_magnitude(vector, max_mag):
    """Limit the magnitude of a vector."""
    mag = magnitude(vector)
    if mag > max_mag:
        return normalize_vector(vector) * max_mag
    return vector
