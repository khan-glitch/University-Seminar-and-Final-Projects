# train_dqn.py

import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from maze_env import MazeEnv
from dqn_agent import DQNAgent


def encode_state(env):
    """
    Converts the maze into a flat numeric state for DQN.

    Values:
    0.0  = empty space
    -1.0 = obstacle
    0.5  = goal
    1.0  = agent position
    """

    state_grid = np.zeros((env.rows, env.cols), dtype=np.float32)

    for row in range(env.rows):
        for col in range(env.cols):
            if env.grid[row][col] == 1:
                state_grid[row][col] = -1.0
            elif (row, col) == env.goal_position:
                state_grid[row][col] = 0.5
            else:
                state_grid[row][col] = 0.0

    agent_row, agent_col = env.agent_position
    state_grid[agent_row][agent_col] = 1.0

    return state_grid.flatten()


def train_dqn(difficulty="easy", episodes=500):
    env = MazeEnv(difficulty=difficulty)

    agent = DQNAgent(
        state_size=env.get_state_size(),
        action_size=env.get_action_size()
    )

    rewards = []
    steps_list = []
    success_list = []
    epsilon_list = []
    invalid_moves_list = []
    loss_list = []

    start_time = time.time()

    for episode in range(1, episodes + 1):
        env.reset()
        state = encode_state(env)

        total_reward = 0
        steps = 0
        invalid_moves = 0
        success = 0
        done = False
        episode_losses = []

        while not done and steps < env.max_steps:
            old_position = env.agent_position

            action = agent.choose_action(state)

            next_position, reward, env_done = env.step(action)

            if reward == -10 and next_position == old_position:
                invalid_moves += 1

            steps += 1

            if steps >= env.max_steps and not env_done:
                done = True
                reward = -20
            else:
                done = env_done

            if env.agent_position == env.goal_position:
                success = 1

            next_state = encode_state(env)

            agent.remember(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done
            )

            loss = agent.replay()

            if loss is not None:
                episode_losses.append(loss)

            state = next_state
            total_reward += reward

        agent.decay_epsilon()

        if episode % agent.target_update_frequency == 0:
            agent.update_target_network()

        if len(episode_losses) > 0:
            average_loss = sum(episode_losses) / len(episode_losses)
        else:
            average_loss = 0

        rewards.append(total_reward)
        steps_list.append(steps)
        success_list.append(success)
        epsilon_list.append(agent.epsilon)
        invalid_moves_list.append(invalid_moves)
        loss_list.append(average_loss)

        if episode % 50 == 0:
            print(
                f"[{difficulty.upper()}] "
                f"Episode: {episode}, "
                f"Reward: {total_reward}, "
                f"Steps: {steps}, "
                f"Success: {success}, "
                f"Epsilon: {agent.epsilon:.3f}, "
                f"Loss: {average_loss:.4f}"
            )

    training_time = time.time() - start_time

    results = {
        "difficulty": difficulty,
        "algorithm": "DQN",
        "episodes": episodes,
        "rewards": rewards,
        "steps": steps_list,
        "success": success_list,
        "epsilon": epsilon_list,
        "invalid_moves": invalid_moves_list,
        "loss": loss_list,
        "training_time": training_time
    }

    return results, agent


def save_dqn_results(results):
    os.makedirs("results", exist_ok=True)

    difficulty = results["difficulty"]
    algorithm = results["algorithm"]

    df = pd.DataFrame({
        "episode": list(range(1, results["episodes"] + 1)),
        "reward": results["rewards"],
        "steps": results["steps"],
        "success": results["success"],
        "epsilon": results["epsilon"],
        "invalid_moves": results["invalid_moves"],
        "loss": results["loss"]
    })

    csv_path = f"results/{algorithm}_{difficulty}_training_data.csv"
    df.to_csv(csv_path, index=False)

    plt.figure()
    plt.plot(df["episode"], df["reward"])
    plt.title(f"{algorithm} Reward per Episode - {difficulty.capitalize()}")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.savefig(f"results/{algorithm}_{difficulty}_rewards.png")
    plt.close()

    plt.figure()
    plt.plot(df["episode"], df["steps"])
    plt.title(f"{algorithm} Steps per Episode - {difficulty.capitalize()}")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.savefig(f"results/{algorithm}_{difficulty}_steps.png")
    plt.close()

    df["success_rate_rolling"] = df["success"].rolling(window=50).mean() * 100

    plt.figure()
    plt.plot(df["episode"], df["success_rate_rolling"])
    plt.title(f"{algorithm} Success Rate Trend - {difficulty.capitalize()}")
    plt.xlabel("Episode")
    plt.ylabel("Success Rate (%)")
    plt.savefig(f"results/{algorithm}_{difficulty}_success_rate.png")
    plt.close()

    plt.figure()
    plt.plot(df["episode"], df["loss"])
    plt.title(f"{algorithm} Loss per Episode - {difficulty.capitalize()}")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.savefig(f"results/{algorithm}_{difficulty}_loss.png")
    plt.close()

    print(f"Saved results for {algorithm} on {difficulty} map.")


def summarize_results(results):
    last_100_success = results["success"][-100:]
    last_100_steps = results["steps"][-100:]
    last_100_rewards = results["rewards"][-100:]
    last_100_invalid = results["invalid_moves"][-100:]

    success_rate = sum(last_100_success) / len(last_100_success) * 100

    successful_steps = [
        step for step, success in zip(last_100_steps, last_100_success)
        if success == 1
    ]

    if len(successful_steps) > 0:
        avg_steps = sum(successful_steps) / len(successful_steps)
    else:
        avg_steps = None

    avg_reward = sum(last_100_rewards) / len(last_100_rewards)
    avg_invalid_moves = sum(last_100_invalid) / len(last_100_invalid)

    summary = {
        "difficulty": results["difficulty"],
        "algorithm": results["algorithm"],
        "success_rate_last_100": success_rate,
        "avg_steps_success_last_100": avg_steps,
        "avg_reward_last_100": avg_reward,
        "avg_invalid_moves_last_100": avg_invalid_moves,
        "training_time_seconds": results["training_time"]
    }

    return summary


if __name__ == "__main__":
    all_summaries = []

    for difficulty in ["easy", "medium", "hard"]:
        print(f"\nTraining DQN on {difficulty.upper()} map...")
        results, agent = train_dqn(difficulty=difficulty, episodes=500)
        save_dqn_results(results)

        summary = summarize_results(results)
        all_summaries.append(summary)

    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv("results/dqn_summary.csv", index=False)

    print("\nDQN summary:")
    print(summary_df)