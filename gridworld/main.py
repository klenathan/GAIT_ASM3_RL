import argparse
import time
import matplotlib.pyplot as plt
import pygame
from gridworld.config import *
from gridworld.environment import GridWorld
from gridworld.agent import QLearningAgent, SARSAAgent
from gridworld.renderer import Renderer


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

    renderer = None
    if not args.no_render:
        renderer = Renderer(title=f"Gridworld - Level {args.level} - {args.algo}")

    episode_rewards = []

    try:
        for ep in range(args.episodes):
            state = env.reset()
            action = agent.choose_action(state)

            total_reward = 0
            done = False

            while not done:
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

    except KeyboardInterrupt:
        print("Training interrupted.")

    if not args.no_render:
        pygame.quit()

    # Plotting is useful but maybe optional for CLI run?
    # Let's save a plot
    plt.plot(episode_rewards)
    plt.title(f"Training Curve - Level {args.level} - {args.algo}")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.savefig(f"outcomes/training_level_{args.level}_{args.algo}.png")
    print(f"Training finished. Plot saved to outcomes/")


if __name__ == "__main__":
    main()
