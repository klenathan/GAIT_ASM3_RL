"""
Headless evaluation script for trained models - PufferLib/PyTorch version.
Runs multiple episodes without rendering for benchmarking.
"""

import argparse
from pathlib import Path
from arena.evaluation.evaluator import Evaluator, find_latest_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate trained Arena models headlessly",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--model", type=str, default=None, 
                       help="Path to model checkpoint (.pt file). Use 'latest' to auto-find")
    parser.add_argument("--style", type=int, default=1, choices=[1, 2],
                       help="Control style (used with 'latest')")
    parser.add_argument("--episodes", type=int, default=100, 
                       help="Number of episodes to evaluate")
    parser.add_argument("--deterministic", action="store_true", default=True,
                       help="Use deterministic policy (argmax)")
    parser.add_argument("--stochastic", action="store_true",
                       help="Use stochastic policy (sampling)")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cpu", "cuda", "mps"], 
                       help="Device to use for inference")
    parser.add_argument("--runs-dir", type=str, default="./runs",
                       help="Base runs directory")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Find model path
    if args.model is None or args.model.lower() == "latest":
        print(f"Finding latest checkpoint for style {args.style}...")
        model_path = find_latest_checkpoint(args.runs_dir, args.style)
        if model_path is None:
            print(f"No checkpoints found for style {args.style} in {args.runs_dir}")
            return
        print(f"Using: {model_path}")
    else:
        model_path = args.model
        if not Path(model_path).exists():
            print(f"Model not found: {model_path}")
            return
    
    # Create evaluator
    evaluator = Evaluator(requested_device=args.device)
    
    # Load model
    if not evaluator.load_checkpoint(model_path):
        print("Failed to load model")
        return
    
    # Run evaluation
    print(f"\nRunning headless evaluation ({args.episodes} episodes)...")
    print(f"Policy: {'Deterministic' if (args.deterministic and not args.stochastic) else 'Stochastic'}")
    print("-" * 60)
    
    results = evaluator.evaluate_multiple_episodes(
        num_episodes=args.episodes,
        deterministic=(args.deterministic and not args.stochastic),
        render=False,
        stochastic=args.stochastic,
    )
    
    # Save results
    import json
    output_file = Path(model_path).parent / "eval_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    evaluator.close()


if __name__ == "__main__":
    main()

