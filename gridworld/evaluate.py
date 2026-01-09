import argparse
import os
import time
from dataclasses import dataclass

from gridworld.agent import QLearningAgent
from gridworld.environment import GridWorld
from gridworld.renderer import Renderer


@dataclass
class EvalResult:
    episodes: int
    wins: int
    deaths: int
    timeouts: int
    mean_return: float
    mean_steps: float


def _load_agent(model_path: str):
    # Models are stored as pickled q_table dicts (see BaseAgent.save/load).
    # For evaluation we only need greedy action selection from the Q-table.
    actions = [0, 1, 2, 3]

    # The Q-table format is the same for Q-learning and SARSA, so either class works.
    agent = QLearningAgent(actions)
    agent.load(model_path)

    # Pure evaluation: no learning, greedy policy.
    agent.epsilon = 0.0
    agent.epsilon_end = 0.0
    agent.alpha = 0.0

    return agent


def evaluate_once(env: GridWorld, agent, max_steps: int):
    state = env.reset()
    total_reward = 0.0

    steps = 0
    done = False
    info = {}
    while not done and steps < max_steps:
        steps += 1
        action = agent.choose_action(state)
        state, reward, done, info = env.step(action)
        total_reward += float(reward)

    # Prefer explicit terminal signal from env.step() and `info`, since `env.done`
    # is only set on win in the current environment implementation.
    if not done:
        outcome = "timeout"
    elif info.get("cause") in {"fire", "monster"}:
        outcome = "death"
    elif getattr(env, "done", False):
        outcome = "win"
    else:
        # Fallback: treat unknown terminal as death.
        outcome = "death"

    return total_reward, steps, outcome


def evaluate_headless(
    level: int, model_path: str, episodes: int, max_steps: int, intrinsic: bool
):
    env = GridWorld(level_idx=level, use_intrinsic_reward=intrinsic)
    agent = _load_agent(model_path)

    returns = []
    steps_list = []
    wins = deaths = timeouts = 0

    for _ in range(episodes):
        ep_return, ep_steps, outcome = evaluate_once(env, agent, max_steps=max_steps)
        returns.append(ep_return)
        steps_list.append(ep_steps)

        if outcome == "win":
            wins += 1
        elif outcome == "death":
            deaths += 1
        else:
            timeouts += 1

    mean_return = sum(returns) / len(returns) if returns else 0.0
    mean_steps = sum(steps_list) / len(steps_list) if steps_list else 0.0

    return EvalResult(
        episodes=episodes,
        wins=wins,
        deaths=deaths,
        timeouts=timeouts,
        mean_return=mean_return,
        mean_steps=mean_steps,
    )


def evaluate_ui(
    level: int, model_path: str, max_steps: int, render_delay: float, intrinsic: bool
):
    env = GridWorld(level_idx=level, use_intrinsic_reward=intrinsic)
    agent = _load_agent(model_path)

    renderer = Renderer(title=f"Gridworld Eval - Level {level}")

    try:
        state = env.reset()
        total_reward = 0.0
        steps = 0
        done = False

        while not done and steps < max_steps:
            steps += 1
            if not renderer.process_events():
                return 0

            renderer.draw(env, episode=0, reward=int(total_reward))
            time.sleep(render_delay)

            action = agent.choose_action(state)
            state, reward, done, info = env.step(action)
            total_reward += float(reward)

        # Keep the final frame visible briefly.
        renderer.draw(env, episode=0, reward=int(total_reward))
        time.sleep(max(0.5, render_delay))

        return 0
    finally:
        renderer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--model", type=str, required=True, help="Path to .pkl model")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["ui", "headless"],
        default="ui",
        help="Evaluation mode",
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--render_delay", type=float, default=0.05)
    parser.add_argument(
        "--intrinsic",
        action="store_true",
        help="Enable intrinsic reward in the environment (for Level 6 runs)",
    )

    args = parser.parse_args()

    model_path = os.path.abspath(args.model)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    if args.mode == "ui":
        raise SystemExit(
            evaluate_ui(
                level=args.level,
                model_path=model_path,
                max_steps=args.max_steps,
                render_delay=args.render_delay,
                intrinsic=args.intrinsic,
            )
        )

    result = evaluate_headless(
        level=args.level,
        model_path=model_path,
        episodes=args.episodes,
        max_steps=args.max_steps,
        intrinsic=args.intrinsic,
    )

    win_rate = result.wins / result.episodes if result.episodes else 0.0

    print("Evaluation results")
    print(f"- episodes: {result.episodes}")
    print(f"- wins: {result.wins} (win_rate={win_rate:.3f})")
    print(f"- deaths: {result.deaths}")
    print(f"- timeouts: {result.timeouts}")
    print(f"- mean_return: {result.mean_return:.3f}")
    print(f"- mean_steps: {result.mean_steps:.2f}")


if __name__ == "__main__":
    main()
