"""
Final verification: Play sounds WITH visual confirmation.
This will show you when sounds should be playing.
"""

import pygame
from arena.audio import SoundManager
import time

print("=" * 70)
print("FINAL SOUND VERIFICATION TEST")
print("=" * 70)
print("\nThis test will play sounds with visual confirmation.")
print("Watch for '🔊 PLAYING NOW' messages and listen carefully.\n")

# Initialize pygame first
pygame.init()

# Create sound manager
sm = SoundManager(enabled=True, sound_dir="arena/sound", volume=1.0)

if not sm.enabled:
    print("❌ ERROR: Sound manager failed to initialize!")
    exit(1)

print(f"✓ Sound system initialized")
print(f"✓ Pygame mixer: {pygame.mixer.get_init()}")
print(f"✓ Loaded sounds: {sum(1 for s in sm.sounds.values() if s)}/12")

# Test each unique sound file
test_sounds = [
    ('laserShoot.wav', ['player_shoot', 'enemy_shoot']),
    ('hitHurt.wav', ['player_hit', 'enemy_hit', 'spawner_hit']),
    ('explosion.wav', ['enemy_destroyed', 'spawner_destroyed', 'player_death']),
    ('pickupCoin.wav', ['heal', 'enemy_spawn', 'phase_complete', 'victory']),
]

print("\n" + "=" * 70)
print("Starting sound playback test...")
print("PUT ON HEADPHONES or TURN UP VOLUME NOW!")
print("=" * 70)

input("\nPress ENTER when ready to start the test...")

for filename, sound_names in test_sounds:
    print(f"\n--- Testing: {filename} ---")
    print(f"Used for: {', '.join(sound_names)}")
    
    for i in range(3):
        print(f"\n  🔊 PLAYING NOW - Listen! (attempt {i+1}/3)")
        sm.play(sound_names[0], volume_multiplier=1.0)
        time.sleep(1.2)  # Wait between plays
    
    print(f"  ✓ {filename} test complete")
    time.sleep(0.5)

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
print("\n📊 Results:")
print("  ✓ All sound playback calls executed successfully")
print("\nDid you hear the sounds?")
print("  YES → Sound system is working perfectly! 🎉")
print("  NO  → Check system audio settings (see SOUND_TROUBLESHOOTING.md)")
print("\nTo test during gameplay:")
print("  uv run python -m arena.evaluate")
print("=" * 70)

sm.cleanup()
