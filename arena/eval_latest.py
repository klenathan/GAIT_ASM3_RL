"""
Quick utility to evaluate the latest trained model.
Finds the most recent model in runs/ and evaluates it.

Usage:
    python -m arena.eval_latest --episodes 1000
    python -m arena.eval_latest --algo ppo --style 1 --episodes 500
"""

import argparse
import os
from pathlib import Path
import subprocess
import sys

# Import algorithms to register them - needed for AlgorithmRegistry
from arena.training.algorithms import dqn, ppo, ppo_lstm, ppo_dict, a2c  # noqa: F401


def find_latest_model(algo: str = None, style: int = None) -> str:
    """Find the most recently modified model file."""
    runs_dir = Path("runs")
    
    if not runs_dir.exists():
        raise FileNotFoundError("runs/ directory not found")
    
    # Build search pattern
    if algo and style:
        search_pattern = f"{algo}/style{style}/**/*.zip"
    elif algo:
        search_pattern = f"{algo}/**/*.zip"
    elif style:
        search_pattern = f"*/style{style}/**/*.zip"
    else:
        search_pattern = "**/*.zip"
    
    # Find all model files
    model_files = list(runs_dir.glob(search_pattern))
    
    if not model_files:
        raise FileNotFoundError(f"No models found matching pattern: {search_pattern}")
    
    # Sort by modification time (most recent first)
    model_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    
    return str(model_files[0])


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the latest trained model",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--algo', type=str, default=None, 
                       help='Algorithm to search for (ppo, ppo_lstm, etc.)')
    parser.add_argument('--style', type=int, default=None, choices=[1, 2],
                       help='Control style to search for')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of episodes to evaluate (default: 100)')
    parser.add_argument('--deterministic', action='store_true', default=True,
                       help='Use deterministic policy (default)')
    parser.add_argument('--stochastic', action='store_true',
                       help='Use stochastic policy')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file')
    parser.add_argument('--final-only', action='store_true',
                       help='Only search in final/ directories')
    parser.add_argument('--checkpoint-only', action='store_true',
                       help='Only search in checkpoints/ directories')
    
    # Curriculum options
    parser.add_argument('--curriculum-stage', type=int, default=None, choices=[0, 1, 2, 3, 4, 5],
                       help='Curriculum stage (0-5). None=full difficulty.')
    parser.add_argument('--auto-curriculum', action='store_true',
                       help='Auto-detect curriculum stage from training state file')
    
    args = parser.parse_args()
    
    try:
        print("Searching for latest model...")
        model_path = find_latest_model(args.algo, args.style)
        print(f"✓ Found: {model_path}")
        
        # Filter by directory type if requested
        if args.final_only and 'final' not in model_path:
            print("Searching for latest final model...")
            runs_dir = Path("runs")
            pattern = f"{'**/' if not args.algo else f'{args.algo}/**/'}" + "**/final/*.zip"
            final_models = list(runs_dir.glob(pattern))
            if final_models:
                final_models.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                model_path = str(final_models[0])
                print(f"✓ Found final model: {model_path}")
            else:
                print("No final models found, using latest model anyway")
        
        if args.checkpoint_only and 'checkpoints' not in model_path:
            print("Searching for latest checkpoint model...")
            runs_dir = Path("runs")
            pattern = f"{'**/' if not args.algo else f'{args.algo}/**/'}" + "**/checkpoints/*.zip"
            checkpoint_models = list(runs_dir.glob(pattern))
            if checkpoint_models:
                checkpoint_models.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                model_path = str(checkpoint_models[0])
                print(f"✓ Found checkpoint model: {model_path}")
            else:
                print("No checkpoint models found, using latest model anyway")
        
        # Detect style from path if not specified
        style = args.style
        if style is None:
            if 'style1' in model_path:
                style = 1
            elif 'style2' in model_path:
                style = 2
            else:
                style = 1  # Default
        
        # Build eval command
        cmd = [
            sys.executable, '-m', 'arena.eval_headless',
            '--model', model_path,
            '--episodes', str(args.episodes),
            '--style', str(style)
        ]
        
        if args.stochastic:
            cmd.append('--stochastic')
        
        if args.output:
            cmd.extend(['--output', args.output])
        
        # Pass curriculum options
        if args.curriculum_stage is not None:
            cmd.extend(['--curriculum-stage', str(args.curriculum_stage)])
        
        if args.auto_curriculum:
            cmd.append('--auto-curriculum')
        
        print(f"\nRunning evaluation with {args.episodes} episodes...")
        print(f"Command: {' '.join(cmd)}\n")
        
        # Run evaluation
        subprocess.run(cmd)
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

