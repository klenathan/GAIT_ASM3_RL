"""
Optuna optimization script for Arena RL.
Optimizes hyperparameters to maximize curriculum progression.
Supports all registered algorithms via CLI arguments.
"""

import argparse
import os
import sys
import time
from typing import Dict, Any, Type, Optional

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from tqdm.auto import tqdm

from arena.core.config import TrainerConfig
from arena.core.device import DeviceManager
from arena.training.base import BaseTrainer
from arena.training.registry import AlgorithmRegistry

# Import algorithms to ensure they register themselves
import arena.training.algorithms


class OptunaCallback(BaseCallback):
    """
    Reports curriculum progress to Optuna for pruning.

    Metric:
        Primary: Curriculum Stage Index (int)
        Secondary: Progress within stage (fractional, estimated from win rate)

    We report a composite score: stage_index + (win_rate * 0.99)
    """

    def __init__(self, trial: optuna.Trial, curriculum_manager, verbose=0):
        super().__init__(verbose)
        self.trial = trial
        self.curriculum_manager = curriculum_manager
        self.eval_freq = 2048  # Check for pruning every N steps
        self.best_score = -float("inf")

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # Calculate composite score
            stage_idx = self.curriculum_manager.current_stage_index

            # Get recent win rate for fractional progress
            wins = (
                self.curriculum_manager.metrics.wins[-100:]
                if self.curriculum_manager.metrics.wins
                else [0]
            )
            win_rate = sum(wins) / len(wins) if wins else 0.0

            score = stage_idx + (win_rate * 0.99)

            self.best_score = max(self.best_score, score)

            # Report to Optuna
            self.trial.report(score, self.num_timesteps)

            # Prune if necessary
            if self.trial.should_prune():
                return False

        return True


class TqdmCallback(BaseCallback):
    """
    Displays a progress bar for the training process.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.pbar = None

    def _on_training_start(self) -> None:
        self.pbar = tqdm(total=self.model._total_timesteps, desc="Training")

    def _on_step(self) -> bool:
        if self.pbar:
            self.pbar.update(self.model.n_envs)
        return True

    def _on_training_end(self) -> None:
        if self.pbar:
            self.pbar.close()
            self.pbar = None


class OptunaTrainerMixin:
    """
    Mixin to inject Optuna functionality into any BaseTrainer subclass.
    """

    def __init__(
        self,
        config: TrainerConfig,
        trial: optuna.Trial,
        sampled_hparams: Dict[str, Any],
    ):
        self.trial = trial
        self.sampled_hparams = sampled_hparams
        # Initialize BaseTrainer (this mixin must be used with a class inheriting from BaseTrainer)
        super().__init__(config)

    def get_hyperparameters(self) -> Dict[str, Any]:
        """Override to return sampled parameters."""
        # Use simple validation for batch size from BaseTrainer logic if needed,
        # but primarily return the sampled dict.
        return self.sampled_hparams

    def setup_callbacks(self, hparams: Dict[str, Any]) -> None:
        """Inject OptunaCallback."""
        super().setup_callbacks(hparams)

        callbacks_to_add = []

        # Add Tqdm Callback
        callbacks_to_add.append(TqdmCallback())

        if self.curriculum_manager and self.curriculum_manager.enabled:
            optuna_callback = OptunaCallback(
                self.trial, self.curriculum_manager, verbose=1
            )
            callbacks_to_add.append(optuna_callback)
        else:
            print(
                "Warning: Curriculum manager not disabled/missing, Optuna pruning will be limited."
            )

        # Add to existing callback list
        existing_callbacks = self.callbacks.callbacks
        self.callbacks = CallbackList(existing_callbacks + callbacks_to_add)


def make_optuna_trainer_class(base_cls: Type[BaseTrainer]) -> Type:
    """Dynamically create a trainer class that supports Optuna."""

    class DynamicOptunaTrainer(OptunaTrainerMixin, base_cls):
        pass

    DynamicOptunaTrainer.__name__ = f"Optuna{base_cls.__name__}"
    return DynamicOptunaTrainer


def sample_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sample hyperparameters for PPO."""
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_categorical("n_steps", [1024, 2048, 4096, 8192])
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    ent_coef = trial.suggest_float("ent_coef", 0.0001, 0.1, log=True)
    gamma = trial.suggest_categorical("gamma", [0.98, 0.99, 0.995, 0.999])
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.90, 0.95, 0.98, 0.99])
    max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.3, 0.5, 0.7, 1.0])

    return {
        "learning_rate": learning_rate,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "ent_coef": ent_coef,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "max_grad_norm": max_grad_norm,
        "n_epochs": 10,
        "clip_range": 0.2,
        "target_kl": 0.015,
        "verbose": 0,
    }


def sample_dqn_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sample hyperparameters for DQN."""
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    buffer_size = trial.suggest_categorical("buffer_size", [10000, 50000, 100000])
    gamma = trial.suggest_categorical("gamma", [0.98, 0.99, 0.995, 0.999])
    train_freq = trial.suggest_categorical("train_freq", [1, 4, 8, 16])
    gradient_steps = trial.suggest_categorical("gradient_steps", [1, 2, 4])
    exploration_fraction = trial.suggest_float("exploration_fraction", 0.1, 0.5)

    return {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "gamma": gamma,
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
        "exploration_fraction": exploration_fraction,
        "exploration_final_eps": 0.05,
        "target_update_interval": 1000,
        "learning_starts": 1000,
        "verbose": 0,
    }


def sample_a2c_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sample hyperparameters for A2C."""
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_categorical(
        "n_steps", [5, 10, 20, 50, 100]
    )  # A2C usually has small n_steps
    gamma = trial.suggest_categorical("gamma", [0.98, 0.99, 0.995, 0.999])
    ent_coef = trial.suggest_float("ent_coef", 0.0001, 0.1, log=True)
    vf_coef = trial.suggest_float("vf_coef", 0.1, 1.0)
    max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.3, 0.5, 0.7, 1.0])

    return {
        "learning_rate": learning_rate,
        "n_steps": n_steps,
        "gamma": gamma,
        "ent_coef": ent_coef,
        "vf_coef": vf_coef,
        "max_grad_norm": max_grad_norm,
        "gae_lambda": 1.0,  # A2C usually doesn't use GAE but SB3 implementation might
        "verbose": 0,
    }


def sample_hyperparameters(trial: optuna.Trial, algo: str) -> Dict[str, Any]:
    """Dispatcher for sampling hyperparameters."""
    if algo == "ppo":
        return sample_ppo_params(trial)
    elif algo == "dqn":
        return sample_dqn_params(trial)
    elif algo == "a2c":
        return sample_a2c_params(trial)
    elif "ppo" in algo:  # Fallback for variants like ppo_lstm
        return sample_ppo_params(trial)
    else:
        # Generic fallback or error
        print(
            f"Warning: No specific sampler for {algo}, using default PPO-like params."
        )
        return sample_ppo_params(trial)


def objective(trial: optuna.Trial):
    # Retrieve global args (passed via global variable for simplicity in this script structure)
    args = globals().get("ARGS")

    algo = args.algo
    style = args.style

    # 1. Sample Hyperparameters
    hparams = sample_hyperparameters(trial, algo)

    # Sample architecture if supported
    # Note: This updates config, not hparams directly passed to algo constructor usually,
    # but BaseTrainer uses config to build policy_kwargs.
    # We need to handle this.
    net_arch_type = trial.suggest_categorical("net_arch", ["tiny", "small", "medium"])
    if net_arch_type == "tiny":
        net_arch = dict(pi=[64, 64], vf=[64, 64])
    elif net_arch_type == "small":
        net_arch = dict(pi=[128, 128], vf=[128, 128])
    else:  # medium
        net_arch = dict(pi=[256, 256], vf=[256, 256])

    # 2. Configure Trainer
    run_name = f"optuna_{algo}_style{style}_trial_{trial.number}"

    # Determine timesteps
    if args.steps is not None:
        timesteps = args.steps
    elif args.test:
        timesteps = 2048
    else:
        timesteps = 200_000

    config = TrainerConfig(
        algo=algo,
        style=style,
        device=args.device,
        num_envs=0,  # Auto-detect
        render=False,
        progress_bar=False,
        total_timesteps=timesteps,
    )

    # Apply architecture to config
    # Note: BaseTrainer expects these in specific config fields depending on algo
    # e.g. ppo_net_arch, a2c_net_arch.
    # We'll set them all to be safe or map dynamically.
    if "ppo" in algo:
        config.ppo_net_arch = net_arch
    elif "a2c" in algo:
        config.a2c_net_arch = net_arch

    # 3. Instantiate Dynamic Trainer
    try:
        base_cls = AlgorithmRegistry.get(algo)
    except ValueError as e:
        print(e)
        raise

    OptunaTrainerClass = make_optuna_trainer_class(base_cls)
    trainer = OptunaTrainerClass(config, trial, hparams)

    # 4. Run Training
    try:
        model = trainer.train()
    except optuna.exceptions.TrialPruned:
        raise
    except Exception as e:
        print(f"Trial failed with error: {e}")
        import traceback

        traceback.print_exc()
        return -float("inf")

    # 5. Return Score
    if trainer.curriculum_manager and trainer.curriculum_manager.enabled:
        stage_idx = trainer.curriculum_manager.current_stage_index
        wins = (
            trainer.curriculum_manager.metrics.wins[-100:]
            if trainer.curriculum_manager.metrics.wins
            else [0]
        )
        win_rate = sum(wins) / len(wins) if wins else 0.0
        final_score = stage_idx + (win_rate * 0.99)
    else:
        # Fallback if no curriculum (e.g. plain RL)
        # Use mean episode reward
        if trainer.model and hasattr(trainer, "callbacks"):
            # Find ArenaCallback
            arena_cb = next(
                (
                    cb
                    for cb in trainer.callbacks.callbacks
                    if isinstance(cb, type(trainer.callbacks.callbacks[1]))
                ),
                None,
            )
            # This is fragile, better to check class name or just assume return 0
            final_score = 0

    return final_score


def main():
    parser = argparse.ArgumentParser(description="Arena RL Optuna Optimization")
    parser.add_argument(
        "--algo", type=str, default="ppo", help="Algorithm to optimize (ppo, a2c, dqn)"
    )
    parser.add_argument(
        "--style", type=int, default=1, choices=[1, 2], help="Control style"
    )
    parser.add_argument("--n-trials", type=int, default=50, help="Number of trials")
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Study name (default: auto-generated)",
    )
    parser.add_argument(
        "--storage", type=str, default="sqlite:///arena_optuna.db", help="Storage URL"
    )
    parser.add_argument(
        "--test", action="store_true", help="Run in test mode (fast, minimal steps)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Total timesteps for training (default: 200000, or 2048 if --test)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cpu, cuda, mps)",
    )
    args = parser.parse_args()

    # Generate study name if not provided
    if args.study_name is None:
        args.study_name = f"optuna_{args.algo}_style{args.style}"

    # Set globals for objective
    global ARGS
    ARGS = args

    # Create study
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction="maximize",
        sampler=TPESampler(n_startup_trials=10, multivariate=True),
        pruner=MedianPruner(
            n_startup_trials=5, n_warmup_steps=10000, interval_steps=2048
        ),
    )

    print(f"Starting optimization for {args.algo} (Style {args.style})")
    print(f"Trials: {args.n_trials}, Study: {args.study_name}")

    study.optimize(objective, n_trials=args.n_trials, n_jobs=1)

    print("\nOptimization complete!")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value: {study.best_value}")
    print("Best hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")


if __name__ == "__main__":
    main()
