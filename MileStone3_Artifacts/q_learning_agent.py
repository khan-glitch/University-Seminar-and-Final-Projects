# q_learning_agent.py

import random
import numpy as np


class QLearningAgent:
    """
    Baseline Q-learning agent for maze navigation.
    """

    def __init__(
        self,
        rows,
        cols,
        num_actions=4,
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=1.0,
        epsilon_decay=0.995,
        min_epsilon=0.01
    ):
        self.rows = rows
        self.cols = cols
        self.num_actions = num_actions

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        # Epsilon controls exploration.
        # 1.0 means the agent explores a lot at the beginning.
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        # Q-table shape:
        # rows x cols x actions
        self.q_table = np.zeros((rows, cols, num_actions))

    def choose_action(self, state):
        """
        Chooses an action using epsilon-greedy strategy.

        Sometimes the agent explores randomly.
        Sometimes the agent uses what it has learned.
        """
        row, col = state

        # Exploration: choose random action
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)

        # Exploitation: choose best known action
        return np.argmax(self.q_table[row, col])

    def update_q_value(self, state, action, reward, next_state, done):
        """
        Updates the Q-table using the Q-learning formula.
        """
        row, col = state
        next_row, next_col = next_state

        current_q = self.q_table[row, col, action]

        if done:
            target_q = reward
        else:
            best_next_q = np.max(self.q_table[next_row, next_col])
            target_q = reward + self.discount_factor * best_next_q

        new_q = current_q + self.learning_rate * (target_q - current_q)

        self.q_table[row, col, action] = new_q

    def decay_epsilon(self):
        """
        Reduces exploration over time.
        """
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

        if self.epsilon < self.min_epsilon:
            self.epsilon = self.min_epsilon

    def get_best_action(self, state):
        """
        Returns the best learned action for a given state.
        """
        row, col = state
        return np.argmax(self.q_table[row, col])