import numpy as np
import seaborn
import os
import yaml
import matplotlib.pyplot as plt
from tueplots import bundles

SAGE_GREEN = "#5a8840"
STORMY_TEAL = "#246A73"
PASTEL_PINK = "#F8C0C8"
PALE_SKY = "#C0D6DF"
RUBY_RED = "#A31621"


def plot_rewards(
        raw_data_path,
        env_name = "Hockey-One-v0",
        opponent_name = "strong opponent",
        final_avg_winrate = None):
    
    config_path = os.path.join(raw_data_path, "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}

    model_identifier = config["MODEL_IDENTIFIER"]

    # --- set colors for plotting ---
    if model_identifier == "rainbow":
        mv_average_color =  SAGE_GREEN
    elif model_identifier == "ddpg":
        mv_average_color = STORMY_TEAL
    elif model_identifier == "tdmpc2":
        mv_average_color == RUBY_RED
    else:
        raise NotImplementedError
    
    rewards_color = PALE_SKY

    episode_rewards_path = os.path.join(raw_data_path, f"ep_rew_data-{model_identifier}.npy")
    mv_avg_rewards_path = os.path.join(raw_data_path, f"mv_avg_rew_data-{model_identifier}.npy")

    episode_rewards = np.load(episode_rewards_path)
    mv_avg_rewards = np.load(mv_avg_rewards_path)
    mv_avg_window_size = len(mv_avg_rewards) / 100

     # ------ create reward plot -------

    # Create an x-axis that starts at self.mavg_window_size
    x_axis = np.arange(mv_avg_window_size, mv_avg_window_size + len(mv_avg_rewards))
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(episode_rewards, label="Episode Rewards", color=rewards_color, linewidth=1.5)
    plt.plot(x_axis, mv_avg_rewards, label="Moving Average Episode Rewards", color=mv_average_color, alpha=0.7)
    
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
        raw_data_path="/home/nils-klute/Documents/machine_learning/Reinforcement Learning/hockey007/experiments_rainbow/2026-02-23_15-05-34_rainbow plotting",
        final_avg_winrate=0.82)
    
