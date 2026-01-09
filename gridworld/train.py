import argparse
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# ANSI color codes
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
BLUE = "\033[34m"


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


# Experiments config
DEFAULT_COMMANDS = [
    (
        "Task 1: Level 0 Q-Learning",
        "python -m gridworld.main --level 0 --algo q_learning --episodes 8000 --no_render  --save_model q_level0.pkl",
    ),
    (
        "Task 2: Level 1 Q-Learning",
        "python -m gridworld.main --level 1 --algo q_learning --episodes 20000 --no_render  --save_model q_level1.pkl",
    ),
    (
        "Task 2: Level 1 SARSA",
        "python -m gridworld.main --level 1 --algo sarsa --episodes 20000 --no_render  --save_model sarsa_level1.pkl",
    ),
    (
        "Task 3: Level 2 Q-Learning",
        "python -m gridworld.main --level 2 --algo q_learning --episodes 20000 --no_render  --save_model q_level2.pkl",
    ),
    (
        "Task 4: Level 3 SARSA",
        "python -m gridworld.main --level 3 --algo sarsa --episodes 20000 --no_render  --save_model sarsa_level3.pkl",
    ),
    (
        "Task 4: Level 3 Q-Learning",
        "python -m gridworld.main --level 3 --algo q_learning --episodes 30000 --no_render  --save_model q_level3.pkl",
    ),
    (
        "Task 5: Level 4 Q-Learning",
        "python -m gridworld.main --level 4 --algo q_learning --episodes 300000 --no_render  --save_model q_level4.pkl",
    ),
    (
        "Task 6: Level 5 SARSA",
        "python -m gridworld.main --level 5 --algo sarsa --episodes 300000 --no_render  --save_model sarsa_level5.pkl",
    ),
    (
        "Task 6: Level 5 Q-Learning",
        "python -m gridworld.main --level 5 --algo q_learning --episodes 300000 --no_render  --save_model q_level5.pkl",
    ),
    (
        "Task 7: Level 6 Q-Learning (Intrinsic)",
        "python -m gridworld.main --level 6 --algo q_learning --episodes 8000 --no_render  --intrinsic --save_model q_level6_intrinsic.pkl",
    ),
    (
        "Task 7: Level 6 Q-Learning (Baseline)",
        "python -m gridworld.main --level 6 --algo q_learning --episodes 8000 --no_render  --save_model q_level6_baseline.pkl",
    ),
]


def run_command(name: str, cmd: str, cwd: str) -> tuple[str, int, str]:
    started = time.time()
    proc = subprocess.run(
        cmd,
        shell=True,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    duration_s = time.time() - started

    output = (proc.stdout or "").strip()
    if output:
        output = f"{output}\n"

    summary = f"exit={proc.returncode} time={duration_s:.1f}s"
    return name, proc.returncode, f"{summary}\n{output}"


def _supports_ansi() -> bool:
    if not sys.stdout.isatty():
        return False
    if os.environ.get("NO_COLOR"):
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Batch GridWorld training runner")
    parser.add_argument(
        "--jobs",
        type=int,
        default=max(1, (os.cpu_count() or 2) // 2),
        help="Max concurrent training jobs",
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Run all tasks even if some fail",
    )
    args = parser.parse_args()

    if shutil.which("python") is None and shutil.which("python3") is None:
        raise SystemExit("Python executable not found in PATH")

    ansi = _supports_ansi()
    repo_root = _repo_root()

    ran = 0
    failures: list[tuple[str, int, str]] = []

    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = [
            executor.submit(run_command, name, cmd, repo_root)
            for name, cmd in DEFAULT_COMMANDS
        ]

        for fut in as_completed(futures):
            name, returncode, output = fut.result()
            ran += 1

            if returncode == 0:
                if ansi:
                    print(f"[{GREEN}OK{RESET}] {name}")
                else:
                    print(f"[OK] {name}")
            else:
                failures.append((name, returncode, output))
                if ansi:
                    print(f"[{RED}FAIL{RESET}] {name} (exit {returncode})")
                else:
                    print(f"[FAIL] {name} (exit {returncode})")

                if not args.continue_on_error:
                    break

    if failures:
        print("\nFailures")
        for name, returncode, output in failures:
            print(f"\n{name} (exit {returncode})")
            print(output)

        raise SystemExit(1)

    print(f"\nCompleted {ran}/{len(DEFAULT_COMMANDS)} tasks successfully.")


if __name__ == "__main__":
    main()
