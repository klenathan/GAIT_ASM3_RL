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
    args = parser.parse_args()

    # Init structure
    env = GridWorld(level_idx=args.level)
    actions = [0, 1, 2, 3]  # Up, Down, Left, Right

    if args.algo == "q_learning":
        agent = QLearningAgent(
            actions, ALPHA, GAMMA, EPSILON_START, EPSILON_END, EPSILON_DECAY
        )
    else:
        agent = SARSAAgent(
            actions, ALPHA, GAMMA, EPSILON_START, EPSILON_END, EPSILON_DECAY
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

    try:
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

    # Plotting is useful but maybe optional for CLI run?
    # Let's save a plot
    plt.plot(episode_rewards)
    plt.title(f"Training Curve - Level {args.level} - {args.algo}")
    plt.xlabel("Episode")
    plt.xscale("log")
    plt.ylabel("Total Reward")
    plt.savefig(f"outcomes/training_level_{args.level}_{args.algo}.png")
    print(f"Training finished. Plot saved to outcomes/")

    if args.save_model:
        save_path = os.path.join(models_dir, args.save_model)
        agent.save(save_path)


if __name__ == "__main__":
    main()
