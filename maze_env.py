# maze_env.py

import random
from collections import deque


class MazeEnv:
    """
    A grid-based maze environment for reinforcement learning.

    Cell meanings:
    0 = empty space
    1 = obstacle/wall
    2 = goal
    """

    ACTIONS = {
        0: (-1, 0),  # up
        1: (1, 0),   # down
        2: (0, -1),  # left
        3: (0, 1)    # right
    }

    DIFFICULTY_SETTINGS = {
        "easy": {
            "rows": 6,
            "cols": 6,
            "obstacle_density": 0.20,
            "max_steps": 100,
            "seed": 40
        },
        "medium": {
            "rows": 8,
            "cols": 8,
            "obstacle_density": 0.25,
            "max_steps": 150,
            "seed": 41
        },
        "hard": {
            "rows": 10,
            "cols": 10,
            "obstacle_density": 0.30,
            "max_steps": 200,
            "seed": 42
        }
    }

    def __init__(
        self,
        difficulty=None,
        rows=6,
        cols=6,
        obstacle_density=0.2,
        start_position=(0, 0),
        goal_position=None,
        max_steps=100,
        seed=None
    ):
        """
        Creates the maze environment.

        If difficulty is provided, the preset settings are used:
        easy, medium, or hard.

        If difficulty is not provided, custom values are used.
        """

        self.difficulty = difficulty

        if difficulty is not None:
            difficulty = difficulty.lower()

            if difficulty not in self.DIFFICULTY_SETTINGS:
                raise ValueError("Difficulty must be 'easy', 'medium', or 'hard'.")

            settings = self.DIFFICULTY_SETTINGS[difficulty]

            rows = settings["rows"]
            cols = settings["cols"]
            obstacle_density = settings["obstacle_density"]
            max_steps = settings["max_steps"]
            seed = settings["seed"]

            start_position = (0, 0)
            goal_position = (rows - 1, cols - 1)

        self.rows = rows
        self.cols = cols
        self.obstacle_density = obstacle_density
        self.start_position = start_position
        self.goal_position = goal_position or (rows - 1, cols - 1)
        self.max_steps = max_steps
        self.seed = seed

        if seed is not None:
            random.seed(seed)

        self.grid = self._generate_valid_maze()
        self.agent_position = self.start_position
        self.steps_taken = 0

    def reset(self):
        """
        Resets the environment back to the starting position.
        """
        self.agent_position = self.start_position
        self.steps_taken = 0
        return self.agent_position

    def step(self, action):
        """
        Moves the agent based on the selected action.

        Returns:
        next_state, reward, done
        """
        self.steps_taken += 1

        row_change, col_change = self.ACTIONS[action]
        current_row, current_col = self.agent_position

        new_position = (
            current_row + row_change,
            current_col + col_change
        )

        if not self._is_valid_position(new_position):
            reward = -10
            done = False
            return self.agent_position, reward, done

        self.agent_position = new_position

        if self.agent_position == self.goal_position:
            reward = 100
            done = True
        elif self.steps_taken >= self.max_steps:
            reward = -20
            done = True
        else:
            reward = -1
            done = False

        return self.agent_position, reward, done

    def render(self):
        """
        Prints the maze in the terminal.
        """
        print(f"Difficulty: {self.difficulty or 'custom'}")
        print(f"Size: {self.rows}x{self.cols}")
        print(f"Obstacle density: {self.obstacle_density}")
        print(f"Max steps: {self.max_steps}")
        print()

        for row in range(self.rows):
            line = ""

            for col in range(self.cols):
                position = (row, col)

                if position == self.agent_position:
                    line += "A "
                elif position == self.start_position:
                    line += "S "
                elif position == self.goal_position:
                    line += "G "
                elif self.grid[row][col] == 1:
                    line += "# "
                else:
                    line += ". "

            print(line)

        print()

    def get_state_size(self):
        """
        Returns the number of values needed to describe the state.
        Useful later for DQN.
        """
        return self.rows * self.cols

    def get_action_size(self):
        """
        Returns number of possible actions.
        """
        return len(self.ACTIONS)

    def _generate_valid_maze(self):
        """
        Generates a random maze that always has at least one path
        from start to goal.
        """
        while True:
            grid = [[0 for _ in range(self.cols)] for _ in range(self.rows)]

            for row in range(self.rows):
                for col in range(self.cols):
                    position = (row, col)

                    if position == self.start_position:
                        continue

                    if position == self.goal_position:
                        continue

                    if random.random() < self.obstacle_density:
                        grid[row][col] = 1

            goal_row, goal_col = self.goal_position
            grid[goal_row][goal_col] = 2

            if self._path_exists(grid):
                return grid

    def _is_valid_position(self, position):
        """
        Checks whether the position is inside the maze and not a wall.
        """
        row, col = position

        if row < 0 or row >= self.rows:
            return False

        if col < 0 or col >= self.cols:
            return False

        if self.grid[row][col] == 1:
            return False

        return True

    def _path_exists(self, grid):
        """
        Checks whether there is a possible path from start to goal.
        This prevents impossible mazes from being generated.
        """
        queue = deque([self.start_position])
        visited = set()

        while queue:
            position = queue.popleft()

            if position == self.goal_position:
                return True

            if position in visited:
                continue

            visited.add(position)

            row, col = position

            for row_change, col_change in self.ACTIONS.values():
                new_row = row + row_change
                new_col = col + col_change
                new_position = (new_row, new_col)

                if (
                    0 <= new_row < self.rows and
                    0 <= new_col < self.cols and
                    grid[new_row][new_col] != 1 and
                    new_position not in visited
                ):
                    queue.append(new_position)

        return False


# Runs when maze_env.py is executed directly.
# This is used to preview all map difficulty levels.
if __name__ == "__main__":
    for difficulty in ["easy", "medium", "hard"]:
        env = MazeEnv(difficulty=difficulty)
        env.reset()
        env.render()