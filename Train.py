import numpy as np
import torch
import time
from agents.AgentBaseclass import Agent
import os
from datetime import datetime
import torch.optim as optim
import matplotlib.pyplot as plt
from itertools import count


class Training():
    def __init__(
        self,
        agent = None,
        env = None,
        base_dir = "experiments",
        save_intermediate_agents: bool = False,
        verbose=False,
        mavg_window_size = 15
        ):
        
        self.statistics = {
            "ep_rew": [0.0],
            "mv_avg_rew": [],
            "tr_loss": [],
            "min_q": [],
            "mean_q": [],
            "max_q": []
        }

        self.agent = agent
        self.env = env
        self.base_dir = base_dir
        self.save_intermediate_agents = save_intermediate_agents
        self.verbose = verbose
        self.mavg_window_size = mavg_window_size
        self.steps_done = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    def train(self):
        
        torch.set_default_dtype(torch.float32)

        # print hyperparameter settings to console 
        if self.verbose:
            self.agent.print_config()

        # Create experiment folder and save config
        self.experiment_path = self.agent.save_experiment_config(self.base_dir)


        start = time.time()
        best_mv_avg_reward = float('-inf')

        for i_episode in range(self.agent.NUM_EPISODES):
            self.agent.cur_episode = i_episode

            # --------- init environment -----------
            state, info = self.env.reset()
            
            for t in count():
                self.steps_done +=1

                # ------ act ------
                action = self.agent.act(self.env, state, self.steps_done, self.statistics)
                next_state, reward, terminated, truncated, _ = self.env.step(action)


                # ------ observe ------
                self.agent.observe(state, action, reward, next_state, terminated)
                self.statistics["ep_rew"][-1] += float(reward) 
                

                # ------ move to next state ------
                state = next_state


                # ------ update ------
                if i_episode >= self.agent.START_TRAINING:
                    self.agent.update(self.statistics)

                # ------ terminate episode ------
                done = terminated or truncated
                if done:
                    self.statistics["ep_rew"].append(0)
                    break
            


            # ------ save best performing agent ------
            n = len(self.statistics["ep_rew"]) 
            if n > self.mavg_window_size + 1:
                mv_avg_reward = np.mean(self.statistics["ep_rew"][-self.mavg_window_size:-1])
                self.statistics["mv_avg_rew"].append(mv_avg_reward)
                if mv_avg_reward > best_mv_avg_reward:
                    self.agent.save_dict(self.experiment_path)
                    best_mv_avg_reward = mv_avg_reward
                    if self.verbose:
                        print(f"Saved model with moving average reward: {mv_avg_reward}")


            # ------ print to console -----
            if self.verbose and i_episode % 5 == 0:
                end = time.time()
                print(f"\n** after {i_episode} th episode - {end - start:.5f} sec passed**\n")
            
        
        self.save_data()
    
    def save_data(self, mean_eval_score = 0):

        # ------ create reward plot -------
        mavg_data = np.array(self.statistics["mv_avg_rew"])
        # Create an x-axis that starts at self.mavg_window_size
        x_axis = np.arange(self.mavg_window_size, self.mavg_window_size + len(mavg_data))
        plt.figure(figsize=(8, 6), dpi=300)
        plt.plot(np.array(self.statistics["ep_rew"]))
        plt.plot(np.array(self.statistics["ep_rew"]), label="Episode Rewards", color="blue", linewidth=1.5)
        plt.plot(x_axis, np.array(self.statistics["mv_avg_rew"]), label="Moving Average Episode Rewards", color="red", linestyle="--", alpha=0.7)
        
        plt.xlabel("Episode Number")
        plt.ylabel("Rewards per Episode")
        plt.title(f"Rewards per Episode over Time. Mean Evaluation Score: {mean_eval_score:.3f}")
        plt.savefig(os.path.join(self.experiment_path, f"episode_rewards-{self.agent.MODEL_IDENTIFIER}.png"), dpi=300)
        plt.close()

        # ------ create loss plot ------
        plt.figure(figsize=(8, 6), dpi=300)
        plt.plot(np.array(self.statistics["tr_loss"]))
        plt.xlabel("Time Step")
        plt.ylabel("Loss")
        plt.title("Training Loss per Time Step")
        plt.savefig(os.path.join(self.experiment_path, f"training_losses-{self.agent.MODEL_IDENTIFIER}.png"), dpi=300)
        plt.close()

        # ------ create q-value plot ------
        plt.figure(figsize=(8, 6), dpi=300)
        plt.plot(np.array(self.statistics["mean_q"]), label="Mean Q", color="blue", linewidth=1.5)
        plt.plot(np.array(self.statistics["max_q"]), label="Max Q", color="green", linestyle="--", alpha=0.7)
        plt.plot(np.array(self.statistics["min_q"]), label="Min Q", color="red", linestyle="--", alpha=0.7)
        plt.xlabel("Time Step")
        plt.ylabel("Q-values")
        plt.title("Training Loss per Time Step")
        plt.savefig(os.path.join(self.experiment_path, f"q_values-{self.agent.MODEL_IDENTIFIER}.png"), dpi=300)
        plt.close()

    
    def train_self_play(self, *args, **kwargs):
        pass

    
    def save(self, *args, **kwargs):
        pass
