"""
LSTM Policy for PufferLib - Recurrent actor-critic with LSTM memory.
"""

import torch
import torch.nn as nn
import numpy as np
from gymnasium import spaces


class LSTMPolicy(nn.Module):
    """
    LSTM-based recurrent policy for partially observable environments.
    
    Uses PufferLib's encode/decode pattern:
    - encode_observations: Process obs through LSTM (used during rollouts)
    - decode_actions: Full forward pass (used during training)
    """
    
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Discrete,
        hidden_size: int = 256,
        num_layers: int = 1,
        input_size: int = 128,
        activation: str = "ReLU",
    ):
        super().__init__()
        
        self.observation_space = observation_space
        self.action_space = action_space
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Get observation and action dimensions
        self.obs_dim = int(np.prod(observation_space.shape))
        self.num_actions = action_space.n
        
        # Activation function
        activation_fn = getattr(nn, activation)
        
        # Input encoder (obs -> LSTM input)
        self.encoder = nn.Sequential(
            nn.Linear(self.obs_dim, input_size),
            activation_fn(),
        )
        
        # LSTM core
        # During rollouts: use LSTMCell for step-by-step processing
        # During training: use LSTM for batch processing (3x faster)
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        
        # Actor head (policy)
        self.actor = nn.Linear(hidden_size, self.num_actions)
        
        # Critic head (value function)
        self.critic = nn.Linear(hidden_size, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.orthogonal_(param, gain=1.0)
                else:
                    nn.init.orthogonal_(param, gain=np.sqrt(2))
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
        
        # Smaller initialization for policy head
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.constant_(self.actor.bias, 0.0)
    
    def encode_observations(self, observations, state):
        """
        Encode observations through LSTM (used during rollouts).
        
        Args:
            observations: Tensor of shape (num_envs, obs_dim)
            state: Tuple of (hidden, cell) each of shape (num_envs, hidden_size)
        
        Returns:
            hidden: New hidden state (num_envs, hidden_size)
            state: Tuple of (hidden, cell) for next step
        """
        # Encode observations
        encoded = self.encoder(observations)
        
        # Process through LSTM cell
        hidden, cell = state
        hidden, cell = self.lstm_cell(encoded, (hidden, cell))
        
        return hidden, (hidden, cell)
    
    def decode_actions(self, hidden):
        """
        Decode actions and values from hidden state.
        
        Args:
            hidden: Tensor of shape (batch_size, hidden_size)
        
        Returns:
            logits: Action logits (batch_size, num_actions)
            values: Value estimates (batch_size, 1)
        """
        logits = self.actor(hidden)
        values = self.critic(hidden)
        return logits, values
    
    def forward(self, observations, state=None, done=None):
        """
        Full forward pass for training (batch mode).
        
        Args:
            observations: Tensor of shape (batch_size, seq_len, obs_dim)
            state: Optional initial LSTM state
            done: Optional done flags for masking (batch_size, seq_len)
        
        Returns:
            logits: Action logits (batch_size, seq_len, num_actions)
            values: Value estimates (batch_size, seq_len)
            state: Final LSTM state
        """
        batch_size, seq_len, _ = observations.shape
        
        # Encode observations
        encoded = self.encoder(observations.view(-1, self.obs_dim))
        encoded = encoded.view(batch_size, seq_len, -1)
        
        # Process through LSTM (batch mode - faster for training)
        if state is not None:
            hidden, cell = state
            # Reshape for LSTM: (num_layers, batch_size, hidden_size)
            hidden = hidden.unsqueeze(0)
            cell = cell.unsqueeze(0)
            lstm_out, (hidden, cell) = self.lstm(encoded, (hidden, cell))
            hidden = hidden.squeeze(0)
            cell = cell.squeeze(0)
        else:
            lstm_out, (hidden, cell) = self.lstm(encoded)
            hidden = hidden[-1]  # Last layer
            cell = cell[-1]
        
        # Apply done mask if provided (reset hidden states)
        if done is not None:
            # Expand done mask to match hidden dimensions
            done_mask = done.unsqueeze(-1).float()
            lstm_out = lstm_out * (1 - done_mask)
        
        # Decode actions and values
        lstm_flat = lstm_out.view(-1, self.hidden_size)
        logits = self.actor(lstm_flat).view(batch_size, seq_len, self.num_actions)
        values = self.critic(lstm_flat).view(batch_size, seq_len)
        
        return logits, values, (hidden, cell)
    
    def get_action_and_value(self, observations, state, done=None, actions=None, deterministic=False):
        """
        Get action distribution and value estimate (for training/evaluation).
        
        Args:
            observations: Tensor of shape (batch_size, obs_dim)
            state: Tuple of (hidden, cell) LSTM states
            done: Optional done flags
            actions: Optional pre-selected actions for log prob calculation
            deterministic: If True, select argmax action instead of sampling
        
        Returns:
            actions: Selected actions
            log_probs: Log probabilities of selected actions
            entropy: Entropy of action distribution
            values: Value estimates
            state: Updated LSTM state
        """
        # Encode through LSTM
        hidden, state = self.encode_observations(observations, state)
        
        # Decode actions and values
        logits, values = self.decode_actions(hidden)
        
        # Create categorical distribution
        dist = torch.distributions.Categorical(logits=logits)
        
        # Sample or take deterministic action
        if actions is None:
            if deterministic:
                actions = logits.argmax(dim=-1)
            else:
                actions = dist.sample()
        
        # Get log probabilities and entropy
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return actions, log_probs, entropy, values.squeeze(-1), state
    
    def init_state(self, batch_size, device):
        """Initialize LSTM hidden state."""
        hidden = torch.zeros(batch_size, self.hidden_size, device=device)
        cell = torch.zeros(batch_size, self.hidden_size, device=device)
        return (hidden, cell)
