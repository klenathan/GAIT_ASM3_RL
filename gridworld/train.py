import subprocess
import threading
import time
import sys

# ANSI color codes
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"
BLUE = "\033[34m"

# Experiments config
commands = [
    (
        "Task 1: Level 0 Q-Learning",
        "python3 main.py --level 0 --algo q_learning --episodes 100000 --no_render --save_model q_level0.pkl",
    ),
    (
        "Task 2: Level 1 Q-Learning",
        "python3 main.py --level 1 --algo q_learning --episodes 1000000 --no_render --save_model q_level1.pkl",
    ),
    (
        "Task 2: Level 1 SARSA",
        "python3 main.py --level 1 --algo sarsa --episodes 1000000 --no_render --save_model sarsa_level1.pkl",
    ),
    (
        "Task 3: Level 2 Q-Learning",
        "python3 main.py --level 2 --algo q_learning --episodes 4000000 --no_render --save_model q_level2.pkl",
    ),
    (
        "Task 3: Level 3 SARSA",
        "python3 main.py --level 3 --algo sarsa --episodes 4000000 --no_render --save_model sarsa_level3.pkl",
    ),
    (
        "Task 3: Level 3 Q-Learning",
        "python3 main.py --level 3 --algo q_learning --episodes 4000000 --no_render --save_model q_level3.pkl",
    ),
    (
        "Task 4: Level 4 Q-Learning",
        "python3 main.py --level 4 --algo q_learning --episodes 10000000 --no_render --save_model q_level4.pkl",
    ),
    (
        "Task 4: Level 5 SARSA",
        "python3 main.py --level 5 --algo sarsa --episodes 10000000 --no_render --save_model sarsa_level5.pkl",
    ),
    (
        "Task 5: Level 5 Q-Learning",
        "python3 main.py --level 5 --algo q_learning --episodes 10000000 --no_render --save_model q_level5.pkl",
    ),
    (
        "Task 5: Level 6 Q-Learning (Intrinsic)",
        "python3 main.py --level 6 --algo q_learning --episodes 10000000 --no_render --intrinsic --save_model q_level6_intrinsic.pkl",
    ),
    (
        "Task 5: Level 6 Q-Learning (Baseline)",
        "python3 main.py --level 6 --algo q_learning --episodes 10000000 --no_render --save_model q_level6_baseline.pkl",
    ),
]


# Track status: 'PENDING', 'RUNNING', 'DONE', 'ERROR'
task_status = {name: "PENDING" for name, _ in commands}
task_lock = threading.Lock()


def run_command(name, cmd):
    with task_lock:
        task_status[name] = "RUNNING"

    try:
        # Run command, suppressing stdout/stderr
        subprocess.run(
            cmd,
            shell=True,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        with task_lock:
            task_status[name] = "DONE"
    except subprocess.CalledProcessError:
        with task_lock:
            task_status[name] = "ERROR"


def print_status():
    # Clear screen sequences could be used, but simple carriage return is safer for log history
    # Or just print the full table repeatedly with clear screen

    # Move cursor up N lines
    n_lines = len(commands) + 2
    # ANSI escape code to move up N lines: \033[nA
    # But for cleaner simple output, let's just reprint the table if we can clear console
    # or just use \r for single line. For multi-line, clearing is better.

    sys.stdout.write(f"\033[{n_lines}A")  # Move up

    print(f"{BOLD}Training Progress:{RESET}")
    for name, _ in commands:
        status = task_status[name]
        if status == "PENDING":
            color = BLUE
            icon = "..."
        elif status == "RUNNING":
            color = YELLOW
            icon = ">>>"
        elif status == "DONE":
            color = GREEN
            icon = "✔"
        else:  # ERROR
            color = RED
            icon = "✘"

        # Fixed width padding
        print(f"  [{color}{icon}{RESET}] {name:<40} {color}{status}{RESET}")
    sys.stdout.flush()


def main():
    threads = []

    # Start threads
    for name, cmd in commands:
        t = threading.Thread(target=run_command, args=(name, cmd))
        threads.append(t)
        t.start()

    # Monitoring loop
    print("\n" * (len(commands) + 2))  # Make space

    while any(t.is_alive() for t in threads):
        print_status()
        time.sleep(0.5)

    # Final update
    print_status()
    print("\n\nAll tasks completed.")

    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
