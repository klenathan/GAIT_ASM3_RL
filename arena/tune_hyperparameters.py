"""
Hyperparameter tuning using Optuna for Deep RL Arena
Optimizes neural network architecture and training hyperparameters
"""

import argparse
import os
from datetime import datetime
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback

from arena.environment import ArenaEnv
from arena import config


def make_env(control_style):
    """Create environment factory"""
    def _init():
        return ArenaEnv(control_style=control_style, render_mode=None)
    return _init


def optimize_dqn(trial, control_style, n_timesteps=50_000):
    """
    Optimize DQN hyperparameters
    
    Trial will sample:
    - Learning rate
    - Network architecture
    - Buffer size
    - Batch size
    - Gamma (discount factor)
    - Exploration parameters
    """
    
    # Sample hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    buffer_size = trial.suggest_categorical("buffer_size", [50_000, 100_000, 200_000])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    gamma = trial.suggest_float("gamma", 0.95, 0.999)
    exploration_fraction = trial.suggest_float("exploration_fraction", 0.1, 0.5)
    exploration_final_eps = trial.suggest_float("exploration_final_eps", 0.01, 0.1)
    target_update_interval = trial.suggest_categorical("target_update_interval", [500, 1000, 2000])
    
    # Sample network architecture
    n_layers = trial.suggest_int("n_layers", 2, 4)
    layer_sizes = []
    for i in range(n_layers):
        size = trial.suggest_categorical(f"layer_{i}_size", [64, 128, 256, 512])
        layer_sizes.append(size)
    
    # Create environment
    env = DummyVecEnv([make_env(control_style)])
    eval_env = DummyVecEnv([make_env(control_style)])
    
    # Create model
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        gamma=gamma,
        exploration_fraction=exploration_fraction,
        exploration_final_eps=exploration_final_eps,
        target_update_interval=target_update_interval,
        policy_kwargs=dict(net_arch=layer_sizes),
        verbose=0,
        device="auto",
    )
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        n_eval_episodes=5,
        eval_freq=5000,
        deterministic=True,
        render=False,
        verbose=0,
    )
    
    try:
        # Train
        model.learn(total_timesteps=n_timesteps, callback=eval_callback)
        
        # Evaluate
        mean_reward = eval_callback.best_mean_reward
        
    except Exception as e:
        print(f"Trial failed with error: {e}")
        mean_reward = -float('inf')
    finally:
        env.close()
        eval_env.close()
    
    return mean_reward


def optimize_ppo(trial, control_style, n_timesteps=50_000):
    """
    Optimize PPO hyperparameters
    
    Trial will sample:
    - Learning rate
    - Network architecture
    - n_steps
    - Batch size
    - n_epochs
    - Gamma
    - GAE lambda
    - Clip range
    - Entropy coefficient
    """
    
    # Sample hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    n_epochs = trial.suggest_int("n_epochs", 5, 15)
    gamma = trial.suggest_float("gamma", 0.95, 0.999)
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.99)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.3)
    ent_coef = trial.suggest_float("ent_coef", 0.0, 0.1)
    
    # Sample network architecture
    n_layers = trial.suggest_int("n_layers", 2, 4)
    layer_sizes = []
    for i in range(n_layers):
        size = trial.suggest_categorical(f"layer_{i}_size", [64, 128, 256, 512])
        layer_sizes.append(size)
    
    # Create environment
    env = DummyVecEnv([make_env(control_style)])
    eval_env = DummyVecEnv([make_env(control_style)])
    
    # Create model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        policy_kwargs=dict(
            net_arch=[dict(pi=layer_sizes, vf=layer_sizes)],
            ortho_init=True,
        ),
        verbose=0,
        device="auto",
    )
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        n_eval_episodes=5,
        eval_freq=5000,
        deterministic=True,
        render=False,
        verbose=0,
    )
    
    try:
        # Train
        model.learn(total_timesteps=n_timesteps, callback=eval_callback)
        
        # Evaluate
        mean_reward = eval_callback.best_mean_reward
        
    except Exception as e:
        print(f"Trial failed with error: {e}")
        mean_reward = -float('inf')
    finally:
        env.close()
        eval_env.close()
    
    return mean_reward


def tune_hyperparameters(args):
    """Run hyperparameter optimization"""
    
    print("=" * 60)
    print("Hyperparameter Optimization with Optuna")
    print("=" * 60)
    print(f"Algorithm: {args.algo.upper()}")
    print(f"Control Style: {args.style}")
    print(f"Number of Trials: {args.n_trials}")
    print(f"Timesteps per Trial: {args.timesteps}")
    print("=" * 60)
    
    # Create study
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_name = f"{args.algo}_style{args.style}_{timestamp}"
    
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=TPESampler(n_startup_trials=10),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=0),
    )
    
    # Optimization objective
    def objective(trial):
        if args.algo == "dqn":
            return optimize_dqn(trial, args.style, args.timesteps)
        else:
            return optimize_ppo(trial, args.style, args.timesteps)
    
    # Run optimization
    print("\nStarting optimization...")
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)
    
    # Print results
    print("\n" + "=" * 60)
    print("Optimization Complete!")
    print("=" * 60)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best reward: {study.best_value:.2f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("=" * 60)
    
    # Save study
    study_path = os.path.join(config.MODEL_SAVE_DIR, f"{study_name}.pkl")
    optuna.logging.get_logger("optuna").info(f"Saving study to {study_path}")
    
    # Optionally save best params to a file
    best_params_path = os.path.join(config.MODEL_SAVE_DIR, f"{study_name}_best_params.txt")
    with open(best_params_path, 'w') as f:
        f.write(f"Best Trial: {study.best_trial.number}\n")
        f.write(f"Best Reward: {study.best_value:.2f}\n\n")
        f.write("Best Hyperparameters:\n")
        for key, value in study.best_params.items():
            f.write(f"{key}: {value}\n")
    
    print(f"\nBest parameters saved to: {best_params_path}")
    
    return study


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for Deep RL Arena")
    
    parser.add_argument(
        "--algo",
        type=str,
        choices=["dqn", "ppo"],
        required=True,
        help="RL algorithm to optimize"
    )
    
    parser.add_argument(
        "--style",
        type=int,
        choices=[1, 2],
        required=True,
        help="Control style: 1=rotation/thrust, 2=directional"
    )
    
    parser.add_argument(
        "--n_trials",
        type=int,
        default=50,
        help="Number of optimization trials"
    )
    
    parser.add_argument(
        "--timesteps",
        type=int,
        default=50_000,
        help="Timesteps per trial (shorter for faster tuning)"
    )
    
    args = parser.parse_args()
    
    # Create model directory
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    
    # Run tuning
    study = tune_hyperparameters(args)


if __name__ == "__main__":
    main()
