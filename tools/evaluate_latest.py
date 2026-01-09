import os
import re
import argparse
import subprocess
from pathlib import Path
from collections import defaultdict


def get_latest_runs(runs_dir: Path):
    """
    Finds the latest run for each (algo, level, intrinsic) combination.
    Run directory format: {algo}_level{level}{suffix}_{timestamp}
    """
    runs = defaultdict(list)

    if not runs_dir.exists():
        print(f"Directory {runs_dir} does not exist.")
        return []

    # Pattern to match: algo_levelX[_intrinsic]_timestamp
    pattern = re.compile(r"^(q_learning|sarsa)_level(\d+)(_intrinsic)?_(\d+)$")

    for run_path in runs_dir.iterdir():
        if not run_path.is_dir():
            continue

        match = pattern.match(run_path.name)
        if match:
            algo, level, intrinsic, timestamp = match.groups()
            key = (algo, int(level), bool(intrinsic))
            runs[key].append((int(timestamp), run_path))

    latest_runs = []
    for key, timestamped_runs in runs.items():
        # Sort by timestamp descending
        timestamped_runs.sort(key=lambda x: x[0], reverse=True)
        latest_runs.append((key, timestamped_runs[0][1]))

    return latest_runs


def evaluate_runs(latest_runs, episodes, max_steps):
    """
    Runs gridworld.evaluate for each latest run.
    """
    for (algo, level, intrinsic), run_path in sorted(latest_runs):
        final_dir = run_path / "final"
        if not final_dir.exists():
            print(f"Skipping {run_path.name}: final/ directory not found.")
            continue

        pkl_files = list(final_dir.glob("*.pkl"))
        if not pkl_files:
            print(f"Skipping {run_path.name}: No .pkl files in final/ directory.")
            continue

        # Usually there's only one .pkl in final/
        model_path = pkl_files[0]

        print(f"\n{'=' * 60}")
        print(f"Evaluating: {run_path.name}")
        print(f"Algo: {algo}, Level: {level}, Intrinsic: {intrinsic}")
        print(f"Model: {model_path}")
        print(f"{'=' * 60}")

        cmd = [
            "python",
            "-m",
            "gridworld.evaluate",
            "--level",
            str(level),
            "--model",
            str(model_path),
            "--mode",
            "headless",
            "--episodes",
            str(episodes),
            "--max_steps",
            str(max_steps),
        ]

        if intrinsic:
            cmd.append("--intrinsic")

        subprocess.run(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate latest models in headless mode."
    )
    parser.add_argument(
        "--episodes", type=int, default=100, help="Number of episodes to evaluate"
    )
    parser.add_argument(
        "--max_steps", type=int, default=100, help="Max steps per episode"
    )
    parser.add_argument(
        "--runs_dir", type=str, default="runs/gridworld", help="Path to runs directory"
    )

    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    latest_runs = get_latest_runs(runs_dir)

    if not latest_runs:
        print("No valid runs found to evaluate.")
    else:
        print(f"Found {len(latest_runs)} latest runs. Starting evaluation...")
        evaluate_runs(latest_runs, args.episodes, args.max_steps)
