import numpy as np
import seaborn
import os
import yaml
import matplotlib.pyplot as plt
from tueplots import bundles


def plot_rewards(
        raw_data_path,
        env_name = "Hockey-One-v0",
        opponent_name = "strong opponent",
        final_avg_winrate = None):
    
    config_path = os.path.join(raw_data_path, "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}

    model_identifier = config["MODEL_IDENTIFIER"]

    #model_identifier = str(config.get("MODEL_IDENTIFIER", "")).strip().lower()

    episode_rewards_path = os.path.join(raw_data_path, f"ep_rew_data-{model_identifier}.npy")
    mv_avg_rewards_path = os.path.join(raw_data_path, f"mv_avg_rew_data-{model_identifier}.npy")

    episode_rewards = np.load(episode_rewards_path)
    mv_avg_rewards = np.load(mv_avg_rewards_path)
    mavg_window_size = len(mv_avg_rewards) / 100

     # ------ create reward plot -------

    # Create an x-axis that starts at self.mavg_window_size
    x_axis = np.arange(mavg_window_size, mavg_window_size + len(mv_avg_rewards))
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(episode_rewards, label="Episode Rewards", color="blue", linewidth=1.5)
    plt.plot(x_axis, mv_avg_rewards, label="Moving Average Episode Rewards", color="red", linestyle="--", alpha=0.7)
    
    plt.xlabel("Episode Number")
    plt.ylabel("Rewards per Episode")
    if env_name == "Hockey-One-v0":
        plt.title(f"Training against the {opponent_name}. Final average winrate: {final_avg_winrate:.2f}")
    elif env_name == "Pendulum-v1":
        plt.title(f"Solving the pendulum upswing problem")

    plt.savefig(os.path.join(raw_data_path, f"episode_rewards_nice_plot-{model_identifier}.svg"), dpi=300)
    plt.close()


if __name__ == "__main__":

    plt.rcParams.update(bundles.icml2022())
    plot_rewards(
        raw_data_path="/home/stud217/hockey007/experiments/2026-02-23_15-05-33_rainbow",
        final_avg_winrate=0.82)
