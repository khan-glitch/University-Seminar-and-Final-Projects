# train_q_learning.py

import os
import time
import pandas as pd
import matplotlib.pyplot as plt

from maze_env import MazeEnv
from q_learning_agent import QLearningAgent


def train_q_learning(difficulty="easy", episodes=500):
    env = MazeEnv(difficulty=difficulty)

    agent = QLearningAgent(
        rows=env.rows,
        cols=env.cols,
        num_actions=env.get_action_size()
    )

    rewards = []
    steps_list = []
    success_list = []
    epsilon_list = []
    invalid_moves_list = []

    start_time = time.time()

    for episode in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0
        steps = 0
        invalid_moves = 0
        done = False
        success = 0

        while not done:
            action = agent.choose_action(state)

            next_state, reward, done = env.step(action)

            if reward == -10 and next_state == state:
                invalid_moves += 1

            agent.update_q_value(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done
            )

            state = next_state
            total_reward += reward
            steps += 1

            if state == env.goal_position:
                success = 1

        agent.decay_epsilon()

        rewards.append(total_reward)
        steps_list.append(steps)
        success_list.append(success)
        epsilon_list.append(agent.epsilon)
        invalid_moves_list.append(invalid_moves)

        if episode % 50 == 0:
            print(
                f"[{difficulty.upper()}] "
                f"Episode: {episode}, "
                f"Reward: {total_reward}, "
                f"Steps: {steps}, "
                f"Success: {success}, "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    training_time = time.time() - start_time

    results = {
        "difficulty": difficulty,
        "algorithm": "Q-learning",
        "episodes": episodes,
        "rewards": rewards,
        "steps": steps_list,
        "success": success_list,
        "epsilon": epsilon_list,
        "invalid_moves": invalid_moves_list,
        "training_time": training_time
    }

    return results, agent


def save_q_learning_results(results):
    os.makedirs("results", exist_ok=True)

    difficulty = results["difficulty"]
    algorithm = results["algorithm"]

    df = pd.DataFrame({
        "episode": list(range(1, results["episodes"] + 1)),
        "reward": results["rewards"],
        "steps": results["steps"],
        "success": results["success"],
        "epsilon": results["epsilon"],
        "invalid_moves": results["invalid_moves"]
    })

    csv_path = f"results/{algorithm}_{difficulty}_training_data.csv"
    df.to_csv(csv_path, index=False)

    # Reward graph
    plt.figure()
    plt.plot(df["episode"], df["reward"])
    plt.title(f"{algorithm} Reward per Episode - {difficulty.capitalize()}")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.savefig(f"results/{algorithm}_{difficulty}_rewards.png")
    plt.close()

    # Steps graph
    plt.figure()
    plt.plot(df["episode"], df["steps"])
    plt.title(f"{algorithm} Steps per Episode - {difficulty.capitalize()}")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.savefig(f"results/{algorithm}_{difficulty}_steps.png")
    plt.close()

    # Success rolling average graph
    df["success_rate_rolling"] = df["success"].rolling(window=50).mean() * 100

    plt.figure()
    plt.plot(df["episode"], df["success_rate_rolling"])
    plt.title(f"{algorithm} Success Rate Trend - {difficulty.capitalize()}")
    plt.xlabel("Episode")
    plt.ylabel("Success Rate (%)")
    plt.savefig(f"results/{algorithm}_{difficulty}_success_rate.png")
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
        print(f"\nTraining Q-learning on {difficulty.upper()} map...")
        results, agent = train_q_learning(difficulty=difficulty, episodes=500)
        save_q_learning_results(results)

        summary = summarize_results(results)
        all_summaries.append(summary)

    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv("results/q_learning_summary.csv", index=False)

    print("\nQ-learning summary:")
    print(summary_df)