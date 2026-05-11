# dqn_agent.py

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DQNNetwork(nn.Module):
    """
    Simple neural network for Deep Q-learning.

    Input:
    - Flattened maze state

    Output:
    - 4 Q-values, one for each action:
      up, down, left, right
    """

    def __init__(self, state_size, action_size):
        super(DQNNetwork, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        return self.model(x)


class DQNAgent:
    """
    Deep Q-learning agent for maze navigation.
    """

    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=0.001,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        min_epsilon=0.05,
        memory_size=10000,
        batch_size=64,
        target_update_frequency=20
    ):
        self.state_size = state_size
        self.action_size = action_size

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency

        self.memory = deque(maxlen=memory_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, action_size).to(self.device)

        self.update_target_network()

        self.optimizer = optim.Adam(
            self.policy_network.parameters(),
            lr=self.learning_rate
        )

        self.loss_function = nn.SmoothL1Loss()

    def choose_action(self, state):
        """
        Uses epsilon-greedy action selection.
        """

        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.policy_network(state_tensor)

        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        """
        Stores experience in replay memory.
        """

        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        """
        Trains the neural network using a random batch from replay memory.
        """

        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)

        states = np.array([item[0] for item in batch], dtype=np.float32)
        actions = np.array([item[1] for item in batch], dtype=np.int64)
        rewards = np.array([item[2] for item in batch], dtype=np.float32)
        next_states = np.array([item[3] for item in batch], dtype=np.float32)
        dones = np.array([item[4] for item in batch], dtype=np.float32)

        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)

        current_q_values = self.policy_network(states_tensor)
        current_q_values = current_q_values.gather(
            1,
            actions_tensor.unsqueeze(1)
        ).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_network(next_states_tensor)
            max_next_q_values = torch.max(next_q_values, dim=1)[0]

            target_q_values = rewards_tensor + (
                self.discount_factor * max_next_q_values * (1 - dones_tensor)
            )

        loss = self.loss_function(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def decay_epsilon(self):
        """
        Reduces exploration over time.
        """

        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

        if self.epsilon < self.min_epsilon:
            self.epsilon = self.min_epsilon

    def update_target_network(self):
        """
        Copies policy network weights into target network.
        This improves training stability.
        """

        self.target_network.load_state_dict(self.policy_network.state_dict())