"""
Batch evaluation and comparison tool.
Automatically finds and compares all models in a run directory.

Usage:
    # Compare all PPO style1 models
    python -m arena.eval_compare_all --algo ppo --style 1 --episodes 500
    
    # Compare all final models across all algorithms
    python -m arena.eval_compare_all --final-only --episodes 1000
    
    # Compare specific run's checkpoints
    python -m arena.eval_compare_all --run-dir runs/ppo/style1/ppo_style1_20251225_175203 --episodes 500
"""

import argparse
import sys
from pathlib import Path
from typing import List
import subprocess

# Import algorithms to register them - needed for AlgorithmRegistry
from arena.training.algorithms import dqn, ppo, ppo_lstm, ppo_dict, a2c  # noqa: F401


def find_models(
    algo: str = None,
    style: int = None,
    run_dir: str = None,
    final_only: bool = False,
    checkpoint_only: bool = False
) -> List[str]:
    """Find models matching criteria."""
    
    if run_dir:
        # Search in specific run directory
        search_path = Path(run_dir)
    else:
        # Search in runs directory
        search_path = Path("runs")
        if algo:
            search_path = search_path / algo
        if style:
            search_path = search_path / f"style{style}"
    
    if not search_path.exists():
        return []
    
    # Build search pattern
    if final_only:
        pattern = "**/final/*.zip"
    elif checkpoint_only:
        pattern = "**/checkpoints/*.zip"
    else:
        pattern = "**/*.zip"
    
    models = list(search_path.glob(pattern))
    return [str(m) for m in sorted(models, key=lambda p: p.stat().st_mtime)]


def group_models_by_run(model_paths: List[str]) -> dict:
    """Group models by their run directory."""
    groups = {}
    for path in model_paths:
        # Extract run directory (3 levels up from model file)
        parts = Path(path).parts
        # Find the run directory (contains timestamp)
        run_name = None
        for i, part in enumerate(parts):
            if '_2025' in part or '_2024' in part:  # Run directory has timestamp
                run_name = part
                break
        
        if run_name:
            if run_name not in groups:
                groups[run_name] = []
            groups[run_name].append(path)
    
    return groups


def main():
    parser = argparse.ArgumentParser(
        description="Batch evaluation and comparison of models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Selection criteria
    parser.add_argument('--algo', type=str, default=None,
                       help='Filter by algorithm (ppo, ppo_lstm, etc.)')
    parser.add_argument('--style', type=int, default=None, choices=[1, 2],
                       help='Filter by control style')
    parser.add_argument('--run-dir', type=str, default=None,
                       help='Evaluate specific run directory')
    
    # Model type filters
    model_type = parser.add_mutually_exclusive_group()
    model_type.add_argument('--final-only', action='store_true',
                           help='Only evaluate final models')
    model_type.add_argument('--checkpoint-only', action='store_true',
                           help='Only evaluate checkpoint models')
    
    # Evaluation parameters
    parser.add_argument('--episodes', type=int, default=100,
                       help='Episodes per model (default: 100)')
    parser.add_argument('--stochastic', action='store_true',
                       help='Use stochastic policy')
    
    # Output options
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file')
    parser.add_argument('--group-by-run', action='store_true',
                       help='Group comparison by training run')
    parser.add_argument('--top-n', type=int, default=None,
                       help='Only evaluate top N most recent models')
    
    args = parser.parse_args()
    
    # Find models
    print("Searching for models...")
    models = find_models(
        algo=args.algo,
        style=args.style,
        run_dir=args.run_dir,
        final_only=args.final_only,
        checkpoint_only=args.checkpoint_only
    )
    
    if not models:
        print("No models found matching criteria")
        sys.exit(1)
    
    print(f"✓ Found {len(models)} models")
    
    # Filter to top N if requested
    if args.top_n:
        models = models[-args.top_n:]  # Most recent N
        print(f"  Evaluating {len(models)} most recent models")
    
    # Group by run if requested
    if args.group_by_run:
        groups = group_models_by_run(models)
        print(f"  Grouped into {len(groups)} training runs")
        
        # Evaluate each group
        for run_name, run_models in groups.items():
            print(f"\n{'='*80}")
            print(f"Evaluating run: {run_name}")
            print(f"{'='*80}")
            
            output_file = f"eval_{run_name}.json" if not args.output else args.output
            
            cmd = [
                sys.executable, '-m', 'arena.eval_headless',
                '--models', *run_models,
                '--episodes', str(args.episodes),
                '--compare',
                '--output', output_file
            ]
            
            if args.stochastic:
                cmd.append('--stochastic')
            
            subprocess.run(cmd)
            print(f"✓ Results saved to {output_file}")
    else:
        # Evaluate all together
        print(f"\nEvaluating {len(models)} models...")
        
        # Detect style from first model if not specified
        style = args.style
        if style is None:
            if 'style1' in models[0]:
                style = 1
            elif 'style2' in models[0]:
                style = 2
            else:
                style = 1
        
        cmd = [
            sys.executable, '-m', 'arena.eval_headless',
            '--models', *models,
            '--episodes', str(args.episodes),
            '--style', str(style),
            '--compare'
        ]
        
        if args.stochastic:
            cmd.append('--stochastic')
        
        if args.output:
            cmd.extend(['--output', args.output])
        
        print(f"Running evaluation...")
        subprocess.run(cmd)


if __name__ == "__main__":
    main()

