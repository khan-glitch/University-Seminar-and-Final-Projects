# compare_results.py

import os
import pandas as pd
import matplotlib.pyplot as plt


def compare_results():
    os.makedirs("results", exist_ok=True)

    q_learning_path = "results/q_learning_summary.csv"
    dqn_path = "results/dqn_summary.csv"

    q_df = pd.read_csv(q_learning_path)
    dqn_df = pd.read_csv(dqn_path)

    combined_df = pd.concat([q_df, dqn_df], ignore_index=True)

    combined_df.to_csv("results/final_comparison.csv", index=False)

    print("\nFinal Comparison:")
    print(combined_df)

    create_bar_chart(
        combined_df,
        metric="success_rate_last_100",
        ylabel="Success Rate (%)",
        title="Success Rate Comparison",
        filename="results/comparison_success_rate.png"
    )

    create_bar_chart(
        combined_df,
        metric="avg_steps_success_last_100",
        ylabel="Average Steps to Goal",
        title="Average Steps Comparison",
        filename="results/comparison_avg_steps.png"
    )

    create_bar_chart(
        combined_df,
        metric="avg_reward_last_100",
        ylabel="Average Reward",
        title="Average Reward Comparison",
        filename="results/comparison_avg_reward.png"
    )

    create_bar_chart(
        combined_df,
        metric="avg_invalid_moves_last_100",
        ylabel="Average Invalid Moves",
        title="Invalid Moves Comparison",
        filename="results/comparison_invalid_moves.png"
    )

    create_bar_chart(
        combined_df,
        metric="training_time_seconds",
        ylabel="Training Time (seconds)",
        title="Training Time Comparison",
        filename="results/comparison_training_time.png"
    )

    print("\nComparison files saved in results folder.")


def create_bar_chart(df, metric, ylabel, title, filename):
    difficulties = ["easy", "medium", "hard"]
    algorithms = ["Q-learning", "DQN"]

    x_labels = []
    values = []

    for difficulty in difficulties:
        for algorithm in algorithms:
            row = df[
                (df["difficulty"] == difficulty) &
                (df["algorithm"] == algorithm)
            ]

            if not row.empty:
                x_labels.append(f"{difficulty}\n{algorithm}")
                values.append(row.iloc[0][metric])

    plt.figure(figsize=(10, 6))
    plt.bar(x_labels, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Difficulty and Algorithm")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    compare_results()