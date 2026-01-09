import argparse
import time
import os

from gridworld.config import *
from gridworld.environment import GridWorld
from gridworld.agent import QLearningAgent, SARSAAgent
from gridworld.renderer import Renderer

try:
    from tqdm import trange
except ImportError:  # pragma: no cover
    trange = None

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None
    print("Warning: torch or tensorboard not found. TensorBoard logging disabled.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, default=0, help="Level index (0-6)")
    parser.add_argument(
        "--algo", type=str, default="q_learning", choices=["q_learning", "sarsa"]
    )
    parser.add_argument(
        "--episodes", type=int, default=500, help="Number of training episodes"
    )
    parser.add_argument(
        "--render_delay",
        type=float,
        default=0.01,
        help="Delay between steps in seconds",
    )
    parser.add_argument(
        "--no_render", action="store_true", help="Disable rendering for faster training"
    )
    parser.add_argument("--save_model", type=str, help="Filename to save the model to")
    parser.add_argument(
        "--load_model", type=str, help="Filename to load the model from"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode (no training). Prefer `python -m gridworld.evaluate` for evaluation.",
    )
    parser.add_argument(
        "--max_steps", type=int, default=100, help="Maximum steps per episode"
    )
    # Repo-root runs directory. GridWorld artifacts go under `runs/gridworld/`.
    parser.add_argument(
        "--runs_dir",
        type=str,
        default=os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)),
            "runs",
            "gridworld",
        ),
        help="Base directory for GridWorld runs",
    )
    parser.add_argument(
        "--intrinsic", action="store_true", help="Enable intrinsic reward (Level 6)"
    )

    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=1000,
        help="Interval (in episodes) to save model checkpoints",
    )
    parser.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable tqdm progress bar",
    )
    args = parser.parse_args()

    # Init structure
    env = GridWorld(level_idx=args.level, use_intrinsic_reward=args.intrinsic)
    actions = [0, 1, 2, 3]  # Up, Down, Left, Right

    # Calculate linear epsilon decay to reach epsilon_end by the end of training
    # or a fraction of it. Let's say we want to reach end by 90% of episodes to allow some exploitation at end.
    # But for "Basic Q-Learning" usually linear over whole period.
    linear_decay = (EPSILON_START - EPSILON_END) / args.episodes

    if args.algo == "q_learning":
        agent = QLearningAgent(
            actions, ALPHA, GAMMA, EPSILON_START, EPSILON_END, linear_decay
        )
    else:
        agent = SARSAAgent(
            actions, ALPHA, GAMMA, EPSILON_START, EPSILON_END, linear_decay
        )

    # Run directory layout
    #
    # runs/gridworld/
    #   <run_name>/
    #     logs/         (tensorboard events)
    #     final/        (best/final model)
    #     checkpoints/  (periodic checkpoints)
    # Note: `repo_root` is kept for convenience but all artifacts use `args.runs_dir`.
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    gridworld_runs_dir = args.runs_dir
    os.makedirs(gridworld_runs_dir, exist_ok=True)

    suffix = "_intrinsic" if args.intrinsic else ""
    run_name = f"{args.algo}_level{args.level}{suffix}_{int(time.time())}"
    run_dir = os.path.join(gridworld_runs_dir, run_name)

    logs_dir = os.path.join(run_dir, "logs")
    final_dir = os.path.join(run_dir, "final")
    checkpoints_dir = os.path.join(run_dir, "checkpoints")

    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    if args.load_model:
        model_path = os.path.join(final_dir, args.load_model)
        if os.path.exists(model_path):
            agent.load(model_path)
        else:
            print(f"Model file not found: {model_path}")
            return

    if args.test:
        agent.epsilon = 0.0
        agent.epsilon_end = 0.0
        agent.alpha = 0.0
        print("Test mode: Epsilon and Alpha set to 0")

    renderer = None
    if not args.no_render:
        renderer = Renderer(title=f"Gridworld - Level {args.level} - {args.algo}")

    writer = None
    if SummaryWriter and not args.test:
        # Write tensorboard events under `<run_dir>/logs/`.
        writer = SummaryWriter(logs_dir)
        print(f"Logging to TensorBoard: {logs_dir}")
    elif not SummaryWriter and not args.test:
        print("TensorBoard logging skipped (module not found).")

    episode_rewards = []

    best_avg_reward = -float("inf")
    last_ckpt_path = None
    total_timesteps = 0

    try:
        current_window_rewards = []

        if args.no_progress:
            iterator = range(args.episodes)
        else:
            if trange is None:
                raise ImportError(
                    "tqdm is required for progress bars. Install it or pass --no_progress."
                )
            iterator = trange(args.episodes, desc="Training", unit="ep")

        for ep in iterator:
            state = env.reset()
            action = agent.choose_action(state)

            total_reward = 0
            done = False

            steps = 0
            while not done and steps < args.max_steps:
                steps += 1
                total_timesteps += 1
                if renderer:
                    if not renderer.process_events():
                        return  # Quit
                    renderer.draw(env, ep, total_reward)
                    time.sleep(args.render_delay)

                next_state, reward, done, info = env.step(action)

                if args.algo == "sarsa":
                    next_action = agent.choose_action(next_state)
                    update_info = agent.update(
                        state, action, reward, next_state, next_action
                    )
                    action = next_action
                else:  # Q-learning
                    update_info = agent.update(state, action, reward, next_state)
                    action = agent.choose_action(next_state)

                if writer and update_info:
                    if "td_error" in update_info:
                        writer.add_scalar(
                            "train/td_error",
                            abs(update_info["td_error"]),
                            total_timesteps,
                        )
                    if args.algo == "q_learning" and "q_max" in update_info:
                        writer.add_scalar(
                            "train/q_max",
                            update_info["q_max"],
                            total_timesteps,
                        )
                    # Note: Q-learning chooses next action greedily for update logic (inside update),
                    # but effectively for the NEXT step in the environment, we usually re-select based on policy.
                    # Wait, standard Q-learning loop:
                    # Choose A from S using policy
                    # Take action A, observe R, S'
                    # Update Q(S,A) using max Q(S',a)
                    # S <- S'
                    # Loop
                    # In my loop above I already chose 'action' before loop for the first time.
                    # So:
                    # 1. env.step(action) -> next_state
                    # 2. agent.update(...)
                    # 3. state = next_state
                    # 4. action = agent.choose_action(state)
                    # Matches standard flow. The 'next_action' in Q-learning case inside loop is just for the next iteration.

                state = next_state
                total_reward += reward

            agent.decay_epsilon()
            episode_rewards.append(total_reward)

            # Log to TensorBoard (per-episode)
            if writer:
                writer.add_scalar("rollout/ep_rew_mean", total_reward, ep)
                writer.add_scalar("rollout/ep_len_mean", steps, ep)
                writer.add_scalar("train/epsilon", agent.epsilon, ep)
                writer.add_scalar("train/episode_steps", steps, ep)
                if (ep + 1) % 10 == 0:
                    writer.flush()

            # Track best model based on moving average of last 10 episodes
            current_window_rewards.append(total_reward)
            if len(current_window_rewards) > 10:
                current_window_rewards.pop(0)

            avg_reward = sum(current_window_rewards) / len(current_window_rewards)

            # Save best model logic
            if args.save_model and not args.test:
                # We save if average reward improves
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    save_path = os.path.join(final_dir, args.save_model)
                    agent.save(save_path, verbose=False)
                    # print(f"New best average reward: {best_avg_reward:.2f}. Model saved.")

                # Checkpoint logic
                if (ep + 1) % args.checkpoint_interval == 0:
                    base_name, ext = os.path.splitext(args.save_model)
                    ckpt_name = f"{base_name}_ep{ep + 1}{ext}"
                    ckpt_path = os.path.join(checkpoints_dir, ckpt_name)
                    agent.save(ckpt_path, verbose=False)

                    print(f"Checkpoint saved: {ckpt_path}")
                    final_path = os.path.join(final_dir, args.save_model)
                    if os.path.exists(final_path):
                        print(f"Final model (best so far): {final_path}")

                    # Cleanup old checkpoint
                    if last_ckpt_path and os.path.exists(last_ckpt_path):
                        try:
                            os.remove(last_ckpt_path)
                        except OSError as e:
                            print(f"Error deleting old checkpoint: {e}")

                    last_ckpt_path = ckpt_path
                    # print(f"Checkpoint saved: {ckpt_path}")

            # print(f"Episode {ep}: Reward {total_reward:.2f}, Epsilon {agent.epsilon:.2f}")

    except KeyboardInterrupt:
        print("Training interrupted.")
    finally:
        if renderer:
            renderer.close()

        if writer:
            writer.close()

    print(f"Run directory: {run_dir}")
    if writer:
        print(f"TensorBoard logs: {logs_dir}")
    if args.save_model:
        print(f"Final model dir: {final_dir}")
        print(f"Checkpoints dir: {checkpoints_dir}")
    print("Training finished.")

    # We no longer save unconditionally at the end to avoid overwriting the best model
    # if args.save_model:
    #     save_path = os.path.join(models_dir, args.save_model)
    #     agent.save(save_path)


if __name__ == "__main__":
    main()
