"""
Deep Q-Network (DQN) agent for Wordle reinforcement learning.
This agent uses neural networks to learn optimal Wordle strategies.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import gymnasium as gym
from typing import Tuple, List

class DQN(nn.Module):
    """Deep Q-Network architecture for Wordle."""

    def __init__(self, output_size: int, input_size: int = 5*26 + 5*3 + 1, hidden_size: int = 256):
        super(DQN, self).__init__()

        # Input layer
        self.input_layer = nn.Linear(input_size, hidden_size)

        # Hidden layers
        self.hidden1 = nn.Linear(hidden_size, hidden_size)
        self.hidden2 = nn.Linear(hidden_size, hidden_size)

        # Output layer: one Q-value per word in the action space
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.output_layer(x)
        return x

class DQNAgent:
    """DQN Agent for playing Wordle."""

    def __init__(self,
                 action_space: int,
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 buffer_size: int = 10000,
                 batch_size: int = 32,
                 target_update_freq: int = 1000):

        self.action_space = action_space
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Q-network and target network
        self.q_network = DQN(output_size=action_space).to(self.device)
        self.target_network = DQN(output_size=action_space).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Hyperparameters
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon_start  # Exploration rate
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq

        # Replay buffer
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

        # Training counter
        self.steps_done = 0

    def _state_to_tensor(self, state: dict) -> torch.Tensor:
        """Convert state dictionary to tensor."""
        # Extract guess (5 letters)
        guess = state["guess"]
        # Extract feedback (5 values)
        feedback = state["feedback"]
        # Extract remaining attempts
        remaining = state["remaining_attempts"]

        # Convert to tensor
        tensor = torch.zeros(5*26 + 5*3 + 1)  # 136 elements

        # Convert guess letters to one-hot
        for i, letter_code in enumerate(guess):
            if letter_code < 26:
                tensor[i * 26 + letter_code] = 1

        # Convert feedback to one-hot
        for i, fb in enumerate(feedback):
            if fb < 3:
                tensor[5*26 + i * 3 + fb] = 1

        # Add remaining attempts
        tensor[-1] = remaining / 6.0  # Normalize to 0-1 range

        return tensor.float().to(self.device)

    def act(self, state: dict, training: bool = True) -> int:
        """Choose action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            # Exploration: random action
            return random.randrange(self.action_space)
        else:
            # Exploitation: best action according to Q-network
            with torch.no_grad():
                state_tensor = self._state_to_tensor(state)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()

    def remember(self, state: dict, action: int, reward: float, next_state: dict, done: bool):
        """Store experience in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        """Train the network on a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return

        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states_tensor = torch.stack([self._state_to_tensor(state) for state in states])
        actions_tensor = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float).to(self.device)
        next_states_tensor = torch.stack([self._state_to_tensor(state) for state in next_states])
        dones_tensor = torch.tensor(dones, dtype=torch.bool).to(self.device)

        # Compute current Q values
        current_q_values = self.q_network(states_tensor).gather(1, actions_tensor.unsqueeze(1))

        # Compute next Q values using target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states_tensor).max(1)[0]
            target_q_values = rewards_tensor + (self.gamma * next_q_values * ~dones_tensor)

        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

        # Update target network
        if self.steps_done % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        self.steps_done += 1

    def save_model(self, filepath: str):
        """Save the model weights."""
        torch.save(self.q_network.state_dict(), filepath)

    def load_model(self, filepath: str):
        """Load model weights."""
        self.q_network.load_state_dict(torch.load(filepath))
        self.target_network.load_state_dict(self.q_network.state_dict())