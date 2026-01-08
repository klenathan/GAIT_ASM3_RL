"""
Sound Manager for Arena game.
Handles all sound effect loading and playback using pygame.mixer.
"""

import pygame
import os
from typing import Dict, Optional
import warnings

# Suppress pkg_resources deprecation warning from pygame
warnings.filterwarnings(
    "ignore", message=".*pkg_resources is deprecated.*", category=UserWarning)


class SoundManager:
    """Manages sound effects for the Arena game."""

    def __init__(self, enabled: bool = True, sound_dir: str = "arena/sound", volume: float = 0.7, debug: bool = False):
        """
        Initialize the sound manager.

        Args:
            enabled: Whether sound is enabled (disable for headless training)
            sound_dir: Directory containing sound files
            volume: Master volume (0.0 to 1.0)
            debug: Print debug messages for sound playback
        """
        self.enabled = enabled
        self.sound_dir = sound_dir
        self.master_volume = volume
        self.sounds: Dict[str, Optional[pygame.mixer.Sound]] = {}
        self.debug = debug

        if not self.enabled:
            return

        # Initialize pygame mixer (or use existing initialization)
        try:
            # Check if mixer is already initialized
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            self._load_sounds()
        except Exception as e:
            print(f"Warning: Failed to initialize audio system: {e}")
            self.enabled = False

    def _load_sounds(self):
        """Load all sound files from the sound directory."""
        # Map game events to available sound files
        # You have: laserShoot.wav, hitHurt.wav, explosion.wav, pickupCoin.wav
        sound_files = {
            'player_shoot': 'laserShoot.wav',
            # 'enemy_shoot': 'laserShoot.wav',  # Reuse same shoot sound
            'enemy_hit': 'hitHurt.wav',
            'enemy_destroyed': 'explosion.wav',
            'spawner_hit': 'hitHurt.wav',
            'spawner_destroyed': 'explosion.wav',
            'player_hit': 'hitHurt.wav',
            'player_death': 'explosion.wav',
            'heal': 'pickupCoin.wav',
            # 'enemy_spawn': 'pickupCoin.wav',  # Can reuse for spawn notification
            'phase_complete': 'pickupCoin.wav',  # Achievement sound
            'victory': 'pickupCoin.wav',  # Victory fanfare
        }

        for sound_name, filename in sound_files.items():
            sound_path = os.path.join(self.sound_dir, filename)
            try:
                if os.path.exists(sound_path):
                    sound = pygame.mixer.Sound(sound_path)
                    sound.set_volume(self.master_volume)
                    self.sounds[sound_name] = sound
                else:
                    # Don't warn for missing optional sounds, just skip them
                    self.sounds[sound_name] = None
            except Exception as e:
                print(f"Warning: Could not load sound '{filename}': {e}")
                self.sounds[sound_name] = None

    def play(self, sound_name: str, volume_multiplier: float = 1.0):
        """
        Play a sound effect.

        Args:
            sound_name: Name of the sound to play
            volume_multiplier: Multiplier for this specific sound (0.0 to 1.0)
        """
        if not self.enabled:
            if self.debug:
                print(f"[Sound] Skipped '{sound_name}' (disabled)")
            return

        sound = self.sounds.get(sound_name)
        if sound:
            try:
                # Set volume on the sound itself before playing
                volume = self.master_volume * volume_multiplier
                sound.set_volume(volume)
                # Play the sound (returns channel or None if no channels available)
                channel = sound.play()
                if self.debug:
                    print(f"[Sound] Playing '{sound_name}' at volume {volume:.2f} (channel={channel})")
            except Exception as e:
                # Print error for debugging but don't crash
                print(f"Warning: Could not play sound '{sound_name}': {e}")
        elif self.debug:
            print(f"[Sound] Sound '{sound_name}' not found in loaded sounds")

    def set_volume(self, volume: float):
        """
        Set master volume for all sounds.

        Args:
            volume: Volume level (0.0 to 1.0)
        """
        self.master_volume = max(0.0, min(1.0, volume))
        if self.enabled:
            for sound in self.sounds.values():
                if sound:
                    sound.set_volume(self.master_volume)

    def stop_all(self):
        """Stop all currently playing sounds."""
        if self.enabled:
            pygame.mixer.stop()

    def cleanup(self):
        """Clean up audio resources."""
        if self.enabled:
            try:
                pygame.mixer.quit()
            except:
                pass
