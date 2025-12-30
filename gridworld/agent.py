import random
import numpy as np


class BaseAgent:
    def __init__(
        self,
        actions,
        alpha=0.1,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
    ):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.q_table = {}  # Map state -> [q_vals]

    def get_q(self, state, action):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))
        return self.q_table[state][action]

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)

        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))

        # Random tie-breaking
        q_vals = self.q_table[state]
        max_q = np.max(q_vals)
        ties = [i for i, q in enumerate(q_vals) if q == max_q]
        return random.choice(ties)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def update(self, state, action, reward, next_state, next_action=None):
        raise NotImplementedError

    def save(self, path):
        import pickle

        with open(path, "wb") as f:
            pickle.dump(self.q_table, f)
        print(f"Model saved to {path}")

    def load(self, path):
        import pickle

        with open(path, "rb") as f:
            self.q_table = pickle.load(f)
        print(f"Model loaded from {path}")


class QLearningAgent(BaseAgent):
    def update(self, state, action, reward, next_state, next_action=None):
        current_q = self.get_q(state, action)

        # Max Q for next state
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(len(self.actions))
        max_next_q = np.max(self.q_table[next_state])

        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q


class SARSAAgent(BaseAgent):
    def update(self, state, action, reward, next_state, next_action=None):
        current_q = self.get_q(state, action)

        # Q of next action (on-policy)
        next_q = self.get_q(next_state, next_action)

        new_q = current_q + self.alpha * (reward + self.gamma * next_q - current_q)
        self.q_table[state][action] = new_q
