import argparse
import time
import os
import matplotlib.pyplot as plt
from config import *
from environment import GridWorld
from agent import QLearningAgent, SARSAAgent
from renderer import Renderer

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
    parser.add_argument("--test", action="store_true", help="Test mode (no training)")
    parser.add_argument(
        "--max_steps", type=int, default=100, help="Maximum steps per episode"
    )
    parser.add_argument(
        "--tensorboard_log",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "runs"),
        help="TensorBoard log directory",
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

    # Models directory
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)

    if args.load_model:
        model_path = os.path.join(models_dir, args.load_model)
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
        log_dir = os.path.join(
            args.tensorboard_log, f"{args.algo}_level{args.level}_{int(time.time())}"
        )
        writer = SummaryWriter(log_dir)
        print(f"Logging to TensorBoard: {log_dir}")
    elif not SummaryWriter and not args.test:
        print("TensorBoard logging skipped (module not found).")

    episode_rewards = []

    best_avg_reward = -float("inf")
    last_ckpt_path = None

    try:
        current_window_rewards = []
        for ep in range(args.episodes):
            state = env.reset()
            action = agent.choose_action(state)

            total_reward = 0
            done = False

            steps = 0
            while not done and steps < args.max_steps:
                steps += 1
                if renderer:
                    if not renderer.process_events():
                        return  # Quit
                    renderer.draw(env, ep, total_reward)
                    time.sleep(args.render_delay)

                next_state, reward, done, info = env.step(action)

                if args.algo == "sarsa":
                    next_action = agent.choose_action(next_state)
                    agent.update(state, action, reward, next_state, next_action)
                    action = next_action
                else:  # Q-learning
                    agent.update(state, action, reward, next_state)
                    action = agent.choose_action(next_state)
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
                    save_path = os.path.join(models_dir, args.save_model)
                    agent.save(save_path)
                    # print(f"New best average reward: {best_avg_reward:.2f}. Model saved.")

                # Checkpoint logic
                if (ep + 1) % args.checkpoint_interval == 0:
                    base_name, ext = os.path.splitext(args.save_model)
                    ckpt_name = f"{base_name}_ep{ep+1}{ext}"
                    ckpt_path = os.path.join(models_dir, ckpt_name)
                    agent.save(ckpt_path)

                    # Cleanup old checkpoint
                    if last_ckpt_path and os.path.exists(last_ckpt_path):
                        try:
                            os.remove(last_ckpt_path)
                        except OSError as e:
                            print(f"Error deleting old checkpoint: {e}")

                    last_ckpt_path = ckpt_path
                    # print(f"Checkpoint saved: {ckpt_path}")

            # print(f"Episode {ep}: Reward {total_reward:.2f}, Epsilon {agent.epsilon:.2f}")

            if writer:
                writer.add_scalar("rollout/ep_rew_mean", total_reward, ep)
                writer.add_scalar("train/epsilon", agent.epsilon, ep)
                writer.add_scalar("rollout/ep_len_mean", steps, ep)

    except KeyboardInterrupt:
        print("Training interrupted.")

    if renderer:
        renderer.close()

    if writer:
        writer.close()

    # Construct unique filename components
    suffix = ""
    if args.intrinsic:
        suffix = "_intrinsic"

    # Plotting
    plt.plot(episode_rewards)
    plt.title(f"Training Curve - Level {args.level} - {args.algo}{suffix}")
    plt.xlabel("Episode")
    plt.xscale("log")
    plt.ylabel("Total Reward")

    plot_filename = f"outcomes/training_level_{args.level}_{args.algo}{suffix}.png"
    plt.savefig(plot_filename)
    print(f"Training finished. Plot saved to {plot_filename}")

    # We no longer save unconditionally at the end to avoid overwriting the best model
    # if args.save_model:
    #     save_path = os.path.join(models_dir, args.save_model)
    #     agent.save(save_path)


if __name__ == "__main__":
    main()
