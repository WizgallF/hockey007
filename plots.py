import numpy as np
import seaborn as sns
import os
import yaml
import matplotlib.pyplot as plt
#from tueplots import bundles

SAGE_GREEN = "#5a8840"
STORMY_TEAL = "#246A73"
PASTEL_PINK = "#F8C0C8"
PALE_SKY = "#C0D6DF"
RUBY_RED = "#A31621"


def plot_rewards(
        raw_data_path,
        env_name = "Hockey-One-v0",
        opponent_name = "weak opponent"):
    
    config_path = os.path.join(raw_data_path, "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}


    sns.set_theme(style="ticks", font_scale=1.7)  
    sns.set_context("paper", font_scale=1.7)
    model_identifier = config["MODEL_IDENTIFIER"]

    # --- set colors for plotting ---
    if model_identifier == "rainbow":
        mv_average_color =  SAGE_GREEN
    elif model_identifier == "ddpg":
        mv_average_color = STORMY_TEAL
    elif model_identifier == "tdmpc2":
        mv_average_color = RUBY_RED
    else:
        raise NotImplementedError
    
    rewards_color = PALE_SKY

    episode_rewards_path = os.path.join(raw_data_path, f"ep_rew_data-{model_identifier}.npy")
    mv_avg_rewards_path = os.path.join(raw_data_path, f"mv_avg_rew_data-{model_identifier}.npy")

    episode_rewards = np.load(episode_rewards_path)
    mv_avg_rewards = np.load(mv_avg_rewards_path)
    mv_avg_window_size = len(mv_avg_rewards) / 100

    mv_avg_rewards_max = np.max(mv_avg_rewards)
    episodes_n = len(mv_avg_rewards)

    
    plt.rcParams.update({
    "text.usetex": True,  
    "font.family": "sans-serif",  
    "mathtext.fontset": "dejavusans"
    })

    # --- create figure
    fig, ax = plt.subplots(figsize=(8, 5))


    # --- plot data ---
    x_axis = np.arange(mv_avg_window_size, mv_avg_window_size + len(mv_avg_rewards))
    ax.plot(episode_rewards, label="Episode Rewards", color=rewards_color, linewidth=1.5, zorder=1)
    ax.plot(x_axis, mv_avg_rewards, label="Moving Average Episode Rewards", color=mv_average_color, linewidth=3, alpha=0.7)

    # --- plot max reward line ---
    ax.axhline(y=mv_avg_rewards_max, color=mv_average_color, linestyle='--', linewidth=2, alpha=0.6, zorder=2)

    # --- set title --
    title_fontsize = 24
    if env_name == "Hockey-One-v0":
        ax.set_title(f"Training against the {opponent_name}", fontsize=title_fontsize)
    elif env_name == "Pendulum-v1":
        ax.set_title(f"Solving the pendulum upswing problem", fontsize=title_fontsize)
    
    # --- customize x-axis ---
    ax.set_xlabel("Episode Number")
    ax.set_xlim(left=0, right=episodes_n)
    if env_name == "Hockey-One-v0":
        ax.set_xticks(ticks=np.arange(0, episodes_n, 5000))
        ax.set_xticks(ticks=np.arange(0, episodes_n, 1000), minor=True)
    else:
        ax.set_xticks(ticks=np.arange(0, episodes_n, 100))
        ax.set_xticks(ticks=np.arange(0, episodes_n, 20), minor=True)
    
    # --- customize y-axis ---
    ax.set_ylabel("Rewards per Episode")
    current_ticks = list(ax.get_yticks())
    
    current_ticks = [int(t) for t in current_ticks if abs(t - mv_avg_rewards_max) > (mv_avg_rewards_max * 0.05)]
    combined_ticks = sorted(current_ticks + [mv_avg_rewards_max])
    combined_ticks = combined_ticks[:-1]
    
    ax.set_yticks(combined_ticks)
    
    ytick_labels = []

    combined_ticks_max_reward_index = combined_ticks.index(mv_avg_rewards_max)
    
    plt.setp(ax.get_yticklabels()[combined_ticks_max_reward_index], color=mv_average_color)

    for val in combined_ticks:
        if np.isclose(val, mv_avg_rewards_max):
            ytick_labels.append(rf"$\mathbf{{{val:.2f}}}$") 
        else:
            ytick_labels.append(f"{val:.0f}")
            
    ax.set_yticklabels(ytick_labels)

    # --- customize plot design ---
    sns.despine()
    plt.subplots_adjust(left=0.15, right=0.97, top=0.9, bottom=0.13)
    fig.savefig(os.path.join(raw_data_path, f"episode_rewards_nice_plot-{model_identifier}.svg"), dpi=300)
    plt.close()


if __name__ == "__main__":

    plot_rewards(
        raw_data_path="/Users/georgtirpitz/Library/CloudStorage/OneDrive-Persönlich/Uni/Master/ReinfrocementLearning/hockey007/experiments/tdmpc2_adam_weak",
        env_name = "Hockey-One-v0" 
        )
    
