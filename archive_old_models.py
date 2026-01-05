#!/usr/bin/env python3
"""
Script to archive old SB3 model files after PufferLib migration.

This moves all .zip model files to an 'archived_sb3_models' directory
for safekeeping without deleting them.
"""

import shutil
from pathlib import Path


def archive_old_models():
    """Archive old SB3 .zip model files."""
    runs_dir = Path("runs")
    archive_dir = Path("archived_sb3_models")
    
    if not runs_dir.exists():
        print("No runs directory found")
        return
    
    # Find all .zip files
    zip_files = list(runs_dir.rglob("*.zip"))
    
    if not zip_files:
        print("No .zip model files found")
        return
    
    print(f"Found {len(zip_files)} old SB3 model files")
    
    # Create archive directory
    archive_dir.mkdir(exist_ok=True)
    
    # Move files
    for zip_file in zip_files:
        relative_path = zip_file.relative_to(runs_dir)
        dest_path = archive_dir / relative_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Archiving: {zip_file} -> {dest_path}")
        shutil.move(str(zip_file), str(dest_path))
    
    print(f"\n✓ Archived {len(zip_files)} files to {archive_dir}")
    print("  These models are SB3 format and incompatible with PufferLib")
    print("  You can delete the archive directory once you verify new training works")


if __name__ == "__main__":
    import sys
    
    print("=" * 70)
    print("Archive Old SB3 Models")
    print("=" * 70)
    print("\nThis will move all .zip model files to 'archived_sb3_models/' directory.")
    print("These models cannot be used with the new PufferLib system.")
    print()
    
    response = input("Continue? [y/N]: ")
    if response.lower() != 'y':
        print("Aborted")
        sys.exit(0)
    
    archive_old_models()
