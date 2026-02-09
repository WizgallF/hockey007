import numpy as np
import torch
import time
from agents.AgentBaseclass import Agent 
from agents.RainbowAgent import RainbowAgent
import os
from datetime import datetime
import torch.optim as optim
import matplotlib.pyplot as plt
from itertools import count
from Wrapper import Envwrapper
import hockey.hockey_env as h_env
from copy import deepcopy


class Training():
    def __init__(
        self,
        agent = None,
        env = None,
        base_dir = "experiments",
        save_intermediate_agents: bool = False,
        verbose=False
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
        self.mavg_window_size = int(self.agent.NUM_EPISODES / 100)

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

        is_vector_env = hasattr(self.env, "num_envs")
        if not is_vector_env:
            for i_episode in range(self.agent.NUM_EPISODES):
                self.agent.cur_episode = i_episode

                # --------- init environment -----------
                state, info = self.env.reset()
                
                for t in count():

                    # ------ act ------
                    action = self.agent.act(self.env, state, i_episode, self.statistics)
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
        else:
            num_envs = int(self.env.num_envs)
            assert type(self.agent).__name__ == "DDPGAgent", (
                "Parallel training is only supported for DDPGAgent."
            )
            ep_rew_per_env = np.zeros(num_envs, dtype=np.float32)
            state, info = self.env.reset()
            episodes_finished = 0

            while episodes_finished < self.agent.NUM_EPISODES:
                self.agent.cur_episode = episodes_finished

                # ------ act ------
                action = self.agent.act(self.env, state, episodes_finished, self.statistics)

                next_state, reward, terminated, truncated, _ = self.env.step(action)

                # ------ observe ------
                self.agent.observe(state, action, reward, next_state, terminated)
                ep_rew_per_env += reward.astype(np.float32)

                # ------ move to next state ------
                state = next_state

                # ------ update ------
                if episodes_finished >= self.agent.START_TRAINING:
                    self.agent.update(self.statistics)

                # ------ terminate episodes ------
                done = np.logical_or(terminated, truncated)
                if np.any(done):
                    done_indices = np.where(done)[0]
                    for env_idx in done_indices:
                        self.statistics["ep_rew"].append(float(ep_rew_per_env[env_idx]))
                        ep_rew_per_env[env_idx] = 0.0

                    # Reset only completed environments if supported
                    if hasattr(self.env, "reset_done"):
                        reset_obs, reset_info = self.env.reset_done()
                        if isinstance(reset_obs, np.ndarray) and reset_obs.shape == state.shape:
                            state = np.where(done[:, None], reset_obs, state)
                        else:
                            for idx, env_idx in enumerate(done_indices):
                                state[env_idx] = reset_obs[idx]
                    else:
                        state, info = self.env.reset()

                # ------ save best performing agent ------
                n = len(self.statistics["ep_rew"]) 
                episodes_finished = n - 1
                if n > self.mavg_window_size + 1:
                    mv_avg_reward = np.mean(self.statistics["ep_rew"][-self.mavg_window_size:-1])
                    self.statistics["mv_avg_rew"].append(mv_avg_reward)
                    if mv_avg_reward > best_mv_avg_reward:
                        self.agent.save_dict(self.experiment_path)
                        best_mv_avg_reward = mv_avg_reward
                        if self.verbose:
                            print(f"Saved model with moving average reward: {mv_avg_reward}")

                # ------ print to console -----
                if self.verbose and episodes_finished % 5 == 0:
                    end = time.time()
                    print(
                        f"\n** after {episodes_finished} th episode - {end - start:.5f} sec passed**\n"
                    )
            

        self.save_data()

        if type(self.agent) == RainbowAgent:
            self.save_q_values()
    
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

    def save_q_values(self):
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

    
    def train_self_play(
            self, 
            opponent,
            discrete_actions = False):

        torch.set_default_dtype(torch.float32)

        self.original_env = self.env
        self.opponent = opponent
        self.best_opponent = deepcopy(opponent)
        self.population_size = 0

        # print hyperparameter settings to console 
        if self.verbose:
            self.agent.print_config()

        # Create experiment folder and save config
        self.experiment_path = self.agent.save_experiment_config(self.base_dir)



        # Add random agent to population for self play
        population_path = os.path.join(self.experiment_path, "agent_population")
        self.save_to_population(population_path, i_training_round=0)
        

        start = time.time()
        best_mv_avg_reward = float('-inf')
        self.agent_against_basic_opp = []


        for i_training_round in range(self.agent.TRAINING_ROUNDS):

            # Wrap environment with Player 2
            self.select_from_population(population_path)
            self.env = Envwrapper(self.original_env, self.opponent, discrete_actions)

            for i_episode in range(self.agent.NUM_EPISODES):
                self.agent.cur_episode = i_episode

                # --------- init environment -----------
                state, info = self.env.reset()
                
                for t in count():

                    # ------ act ------
                    action = self.agent.act(self.env, state, i_episode, self.statistics)
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
                


                # TODO: how to determine which agent to save? just save them regularly?
                # ------ save best performing agent ------
                n = len(self.statistics["ep_rew"]) 
                if n > self.mavg_window_size + 1:
                    mv_avg_reward = np.mean(self.statistics["ep_rew"][-self.mavg_window_size:-1])
                    self.statistics["mv_avg_rew"].append(mv_avg_reward)

                    """
                    if mv_avg_reward > best_mv_avg_reward:
                        self.agent.save_dict(self.experiment_path)
                        best_mv_avg_reward = mv_avg_reward
                        if self.verbose:
                            print(f"Saved model with moving average reward: {mv_avg_reward}")"""


                # ------ print to console -----
                if self.verbose and i_episode % 5 == 0:
                    end = time.time()
                    print(f"\n** after {self.agent.NUM_EPISODES* i_training_round + i_episode} th episode in {i_training_round} th training round - {end - start:.5f} sec passed**\n")

                if i_episode % 1000 == 0:
                    agent_basic_eval = self.agent_against_basicopp_eval(self.agent)
                    self.agent_against_basic_opp.append(agent_basic_eval)

            if self.agent_against_agent_eval(self.agent, self.best_opponent) > 0.4:
                self.best_opponent = deepcopy(self.agent)
                self.save_to_population(population_path, i_training_round)
            
            

        self.save_data()
        self.save_performance_against_basic_opp()

        if type(self.agent) == RainbowAgent:
            self.save_q_values()

        
        eval_results = self.evaluate_agents(population_path)
        

    def evaluate_agents(
            self,
            population_path):
        
        N = len(os.listdir(population_path))

        results = []
        start = time.time()
        for i in range(N):
            end = time.time()
            print(f"eval agents against opponent {i}, {end-start} seconds passed")
            agent_against_i_opponent = []

            for j in range(i+1, N):
                agent_load_path = os.path.join(population_path, self.opponent.MODEL_IDENTIFIER + f"_{j}") + ".pth"
                opponent_load_path = os.path.join(population_path, self.opponent.MODEL_IDENTIFIER + f"_{i}") + ".pth"

                self.agent.load_dict(agent_load_path)
                self.opponent.load_dict(opponent_load_path)

                agent_won_proportion = self.agent_against_agent_eval(self.agent, self.opponent)
                
                agent_against_i_opponent.append(agent_won_proportion)

            results.append(agent_against_i_opponent)

        print(results)
        return results


        # TODO: make population path self argument

    def agent_against_agent_eval(
            self, 
            player1, 
            player2= "basicopp",
            environment = 'Hockey-One-v0',
            num_episodes = 50):
        
        if environment == 'Hockey-One-v0':
            env = h_env.HockeyEnv()
            obs, info = env.reset()
            obs_agent2 = env.obs_agent_two()
            score = {"player1": 0, "player2": 0}

            for _ in range(num_episodes):
                d = False
                obs, info = env.reset()
                while not d:
                    
                    if type(player1) == RainbowAgent:
                        discrete_action = player1.act(env=env, state=obs, greedy=True) 
                        a1 = self._discrete_to_continuous(discrete_action)
                    else:
                        a1 = player1.act(obs) 

                    if type(player2) == RainbowAgent:
                        discrete_action = player2.act(env=env, state=obs_agent2, greedy=True) 
                        a2 = self._discrete_to_continuous(discrete_action)
                    else:
                        a2 = player2.act(obs_agent2)

                    obs, r, d, _, info = env.step(np.hstack([a1,a2]))   
                    obs_agent2 = env.obs_agent_two()
                if info["winner"] == 1:
                    score["player1"] += 1
                else:
                    score["player2"] += 1

        env.close()

        return score["player1"]/num_episodes
    
    def agent_against_basicopp_eval(
            self, 
            player1,
            environment = 'Hockey-One-v0',
            num_episodes = 50):
        
        if environment == 'Hockey-One-v0':
            env = h_env.HockeyEnv()
            obs, info = env.reset()
            obs_agent2 = env.obs_agent_two()
            score = {"player1": 0, "player2": 0}

            
            player2 = h_env.BasicOpponent()

            for _ in range(num_episodes):
                d = False
                obs, info = env.reset()
                while not d:
                    
                    if type(player1) == RainbowAgent:
                        discrete_action = player1.act(env=env, state=obs, greedy=True) 
                        a1 = self._discrete_to_continuous(discrete_action)
                    else:
                        a1 = player1.act(obs) 

                    a2 = player2.act(obs_agent2)

                    obs, r, d, _, info = env.step(np.hstack([a1,a2]))   
                    obs_agent2 = env.obs_agent_two()
                if info["winner"] == 1:
                    score["player1"] += 1
                else:
                    score["player2"] += 1

        env.close()

        return score["player1"]/num_episodes
    
    def save_performance_against_basic_opp(self):
        # ------ create performance_against_basic_opp plot ------
        plt.figure(figsize=(8, 6), dpi=300)
        plt.plot(np.array(self.agent_against_basic_opp ), label="Mean Q", color="blue", linewidth=1.5)
        plt.xlabel("Episodes (1k intervall)")
        plt.ylabel("Proportion of games won")
        plt.title("Performance against basic opponent")
        plt.savefig(os.path.join(self.experiment_path, f"basic_opponent-{self.agent.MODEL_IDENTIFIER}.png"), dpi=300)
        plt.close()
    
    def _discrete_to_continuous(self, discrete_action):
        """ discrete actions from 0 to 7
        Args:
            action: The discrete action
        Returns:
            continuous action
        """
        action_cont = [(discrete_action == 1) * -1.0 + (discrete_action == 2) * 1.0,  # player x
                   (discrete_action == 3) * -1.0 + (discrete_action == 4) * 1.0,  # player y
                   (discrete_action == 5) * -1.0 + (discrete_action == 6) * 1.0]  # player angle
        
        action_cont.append((discrete_action == 7) * 1.0)

        return action_cont

    def save_to_population(
            self,
            population_path,
            i_training_round):

        os.makedirs(population_path, exist_ok=True)
        self.agent.save_dict(population_path, identifier_extension=f"_{self.population_size}")
        self.population_size += 1
    
    def select_from_population(
            self,
            population_path):
        
        agent_index = np.random.randint(0, self.population_size)
        load_path = os.path.join(population_path, self.opponent.MODEL_IDENTIFIER + f"_{agent_index}") + ".pth"
        self.opponent.load_dict(load_path)




    def save(self, *args, **kwargs):
        pass
