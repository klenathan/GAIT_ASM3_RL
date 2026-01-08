"""
Main entry point for evaluating trained models in the Arena environment.
Uses the modular Evaluator and UI components.
"""

import argparse
from arena.evaluation.evaluator import Evaluator

def main():
    parser = argparse.ArgumentParser(description="Deep RL Arena - Interactive Evaluation")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()
    
    print(f"Starting evaluation on device: {args.device}")
    
    evaluator = Evaluator(requested_device=args.device)
    evaluator.main_loop()

if __name__ == "__main__":
    main()
