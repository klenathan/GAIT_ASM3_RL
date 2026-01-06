"""
Imitation Learning (Behavioral Cloning) module for Deep RL Arena.
Enables recording human demonstrations and pretraining agents from expert data.
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class Demonstration:
    """A single demonstration trajectory."""

    observations: List[np.ndarray]
    actions: List[int]
    rewards: List[float]
    dones: List[bool]
    info: Dict = field(default_factory=dict)

    def __len__(self):
        return len(self.observations)

    def to_dict(self):
        """Convert to serializable dictionary."""
        return {
            "observations": [
                obs.tolist() if isinstance(obs, np.ndarray) else obs
                for obs in self.observations
            ],
            "actions": self.actions,
            "rewards": self.rewards,
            "dones": self.dones,
            "info": self.info,
        }

    @classmethod
    def from_dict(cls, data: Dict):
        """Create from dictionary."""
        return cls(
            observations=[
                np.array(obs, dtype=np.float32) for obs in data["observations"]
            ],
            actions=data["actions"],
            rewards=data["rewards"],
            dones=data["dones"],
            info=data.get("info", {}),
        )


@dataclass
class DemonstrationBuffer:
    """Buffer for storing multiple demonstrations."""

    demonstrations: List[Demonstration] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def add_demonstration(self, demo: Demonstration):
        """Add a demonstration to the buffer."""
        self.demonstrations.append(demo)

    def __len__(self):
        return len(self.demonstrations)

    def get_total_transitions(self) -> int:
        """Get total number of state-action pairs across all demonstrations."""
        return sum(len(demo) for demo in self.demonstrations)

    def get_statistics(self) -> Dict:
        """Get statistics about the demonstrations."""
        if not self.demonstrations:
            return {}

        total_rewards = [sum(demo.rewards) for demo in self.demonstrations]
        lengths = [len(demo) for demo in self.demonstrations]
        wins = [demo.info.get("win", False) for demo in self.demonstrations]

        return {
            "num_demonstrations": len(self.demonstrations),
            "total_transitions": self.get_total_transitions(),
            "avg_return": np.mean(total_rewards),
            "std_return": np.std(total_rewards),
            "min_return": np.min(total_rewards),
            "max_return": np.max(total_rewards),
            "avg_length": np.mean(lengths),
            "win_rate": np.mean(wins) if wins else 0.0,
            "wins": sum(wins),
        }

    def save(self, filepath: str):
        """Save buffer to file."""
        data = {
            "demonstrations": [demo.to_dict() for demo in self.demonstrations],
            "metadata": self.metadata,
        }

        # Create directory if needed
        os.makedirs(
            os.path.dirname(filepath) if os.path.dirname(filepath) else ".",
            exist_ok=True,
        )

        with open(filepath, "wb") as f:
            pickle.dump(data, f)

        print(f"Saved {len(self.demonstrations)} demonstrations to {filepath}")

        # Save metadata as JSON for easy viewing
        metadata_path = filepath.replace(".pkl", "_metadata.json")
        stats = self.get_statistics()
        stats.update(self.metadata)
        with open(metadata_path, "w") as f:
            json.dump(stats, f, indent=2)

    @classmethod
    def load(cls, filepath: str):
        """Load buffer from file."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        buffer = cls()
        buffer.demonstrations = [
            Demonstration.from_dict(d) for d in data["demonstrations"]
        ]
        buffer.metadata = data.get("metadata", {})

        print(f"Loaded {len(buffer.demonstrations)} demonstrations from {filepath}")
        stats = buffer.get_statistics()
        print(f"  Total transitions: {stats['total_transitions']}")
        print(f"  Avg return: {stats['avg_return']:.2f} ± {stats['std_return']:.2f}")
        print(f"  Win rate: {stats['win_rate'] * 100:.1f}%")

        return buffer


class DemonstrationRecorder:
    """Records human demonstrations during gameplay."""

    def __init__(self, control_style: int = 1):
        self.control_style = control_style
        self.current_demo = None
        self.buffer = DemonstrationBuffer()
        self.is_recording = False

    def start_episode(self):
        """Start recording a new episode."""
        self.current_demo = Demonstration(
            observations=[],
            actions=[],
            rewards=[],
            dones=[],
            info={"control_style": self.control_style},
        )
        self.is_recording = True

    def record_step(self, obs: np.ndarray, action: int, reward: float, done: bool):
        """Record a single step."""
        if not self.is_recording or self.current_demo is None:
            return

        self.current_demo.observations.append(obs.copy())
        self.current_demo.actions.append(action)
        self.current_demo.rewards.append(reward)
        self.current_demo.dones.append(done)

    def end_episode(self, info: Optional[Dict] = None):
        """End the current episode and add to buffer."""
        if not self.is_recording or self.current_demo is None:
            return

        if info is not None:
            self.current_demo.info.update(info)

        # Only add if episode has data
        if len(self.current_demo) > 0:
            self.buffer.add_demonstration(self.current_demo)
            total_reward = sum(self.current_demo.rewards)
            win = self.current_demo.info.get("win", False)
            print(
                f"Episode recorded: {len(self.current_demo)} steps, "
                f"reward: {total_reward:.1f}, win: {win}"
            )

        self.current_demo = None
        self.is_recording = False

    def save_demonstrations(self, filepath: Optional[str] = None) -> str:
        """Save recorded demonstrations to file."""
        if filepath is None:
            # Generate default filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs("./demonstrations", exist_ok=True)
            filepath = (
                f"./demonstrations/demo_style{self.control_style}_{timestamp}.pkl"
            )

        self.buffer.metadata.update(
            {
                "control_style": self.control_style,
                "timestamp": datetime.now().isoformat(),
            }
        )

        self.buffer.save(filepath)
        return filepath


class ImitationDataset(Dataset):
    """PyTorch Dataset for imitation learning."""

    def __init__(self, buffer: DemonstrationBuffer):
        self.observations = []
        self.actions = []

        # Flatten all demonstrations into state-action pairs
        for demo in buffer.demonstrations:
            for obs, action in zip(demo.observations, demo.actions):
                self.observations.append(obs)
                self.actions.append(action)

        self.observations = np.array(self.observations, dtype=np.float32)
        self.actions = np.array(self.actions, dtype=np.int64)

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx]


class BehavioralCloningTrainer:
    """Trains a policy network via behavioral cloning from demonstrations."""

    def __init__(
        self,
        policy_network: nn.Module,
        demo_buffer: DemonstrationBuffer,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        device: str = "cpu",
    ):
        self.policy = policy_network
        self.buffer = demo_buffer
        self.batch_size = batch_size
        self.device = device

        self.policy.to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        self.dataset = ImitationDataset(demo_buffer)
        self.dataloader = DataLoader(
            self.dataset, batch_size=batch_size, shuffle=True, drop_last=False
        )

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch over the dataset."""
        self.policy.train()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0

        for obs_batch, action_batch in self.dataloader:
            obs_batch = obs_batch.to(self.device)
            action_batch = action_batch.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            logits = self.policy(obs_batch)
            loss = self.criterion(logits, action_batch)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Compute accuracy
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == action_batch).float().mean()

            total_loss += loss.item()
            total_accuracy += accuracy.item()
            num_batches += 1

        return {
            "loss": total_loss / num_batches,
            "accuracy": total_accuracy / num_batches,
        }

    def train(self, num_epochs: int = 10, verbose: bool = True) -> Dict:
        """Train the policy for multiple epochs."""
        history = {"loss": [], "accuracy": []}

        if verbose:
            print(f"Starting behavioral cloning training...")
            print(f"  Dataset size: {len(self.dataset)} transitions")
            print(f"  Batch size: {self.batch_size}")
            print(f"  Epochs: {num_epochs}")
            print(f"  Device: {self.device}")

        for epoch in range(num_epochs):
            metrics = self.train_epoch()
            history["loss"].append(metrics["loss"])
            history["accuracy"].append(metrics["accuracy"])

            if verbose:
                print(
                    f"Epoch {epoch + 1}/{num_epochs}: "
                    f"Loss={metrics['loss']:.4f}, "
                    f"Acc={metrics['accuracy'] * 100:.2f}%"
                )

        if verbose:
            print("Behavioral cloning training complete!")

        return history

    def save_policy(self, filepath: str):
        """Save the trained policy."""
        torch.save(self.policy.state_dict(), filepath)
        print(f"Saved BC policy to {filepath}")

    def load_policy(self, filepath: str):
        """Load a trained policy."""
        self.policy.load_state_dict(torch.load(filepath, map_location=self.device))
        print(f"Loaded BC policy from {filepath}")


def pretrain_from_demonstrations(
    model,
    demo_path: str,
    num_epochs: int = 10,
    learning_rate: float = 1e-3,
    batch_size: int = 64,
    device: str = "cpu",
    verbose: bool = True,
) -> Dict:
    """
    Pretrain an SB3 model's policy network using behavioral cloning.

    Args:
        model: SB3 model instance (PPO, DQN, A2C, etc.)
        demo_path: Path to saved demonstrations (.pkl file)
        num_epochs: Number of training epochs
        learning_rate: Learning rate for BC training
        batch_size: Batch size for BC training
        device: Device to train on ('cpu', 'cuda', 'mps')
        verbose: Whether to print training progress

    Returns:
        Training history dict with loss and accuracy
    """
    # Load demonstrations
    buffer = DemonstrationBuffer.load(demo_path)

    if len(buffer) == 0:
        raise ValueError("No demonstrations found in buffer!")

    # Extract policy network from SB3 model
    # Different model types have different architectures
    policy_network = None

    # Try to get the policy network (works for PPO, A2C)
    if hasattr(model.policy, "action_net"):
        # MlpPolicy: has separate actor (action_net) and critic (value_net)
        # We'll create a wrapper that uses the feature extractor + action net
        policy_network = PolicyNetworkWrapper(model.policy)
    elif hasattr(model.policy, "q_net"):
        # DQN: has q_net
        policy_network = model.policy.q_net
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")

    # Create BC trainer
    trainer = BehavioralCloningTrainer(
        policy_network=policy_network,
        demo_buffer=buffer,
        learning_rate=learning_rate,
        batch_size=batch_size,
        device=device,
    )

    # Train
    history = trainer.train(num_epochs=num_epochs, verbose=verbose)

    return history


class PolicyNetworkWrapper(nn.Module):
    """Wrapper for SB3 policy network to provide standard forward interface."""

    def __init__(self, sb3_policy):
        super().__init__()
        self.policy = sb3_policy

    def forward(self, obs):
        """Forward pass: obs -> action logits."""
        # Extract features
        features = self.policy.extract_features(obs)

        # Get latent policy features (if using shared networks)
        if hasattr(self.policy, "mlp_extractor"):
            latent_pi, _ = self.policy.mlp_extractor(features)
        else:
            latent_pi = features

        # Get action logits
        logits = self.policy.action_net(latent_pi)

        return logits
