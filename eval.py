import numpy as np
import time
from agents.AgentBaseclass import Agent 
from agents.RainbowAgent import RainbowAgent
from agents.DDPGAgent import DDPGAgent
from agents.TDMPC2Agent import TDMPC2Agent
import os
import hockey.hockey_env as h_env
import gymnasium as gym
from Wrapper import DiscreteActionWrapperHockey
import matplotlib.pyplot as plt
import yaml
from collections import defaultdict
import pandas as pd



def evaluate_agents(
            population_path,
            num_episodes = 50):
        
        env = gym.envs.make("Hockey-One-v0", mode=0, weak_opponent=True)
        base_obs_space = env.observation_space
        base_act_space = env.action_space

        env = DiscreteActionWrapperHockey(env)
        state, info = env.reset()

        n_observations = len(state)
        n_actions = env.action_space.n

        
        agents = [f for f in os.listdir(population_path) if f.split(".")[1] == "pth"]
        
        N = len(agents)

        results = np.zeros(shape=(N,N))
        result_dict = defauldict(list)
        print(results)
        start = time.time()
        for i in range(N):
            end = time.time()
            print(f"eval agents for agent " + agents[i] + f" {end-start} seconds passed")

            for j in range(i+1, N):

                agent_config_file = agents[i].split(".")[0] + ".yaml"
                opponent_config_file = agents[j].split(".")[0] + ".yaml"

                agent_load_path = os.path.join(population_path, agents[i])
                opponent_load_path = os.path.join(population_path, agents[j])

                agent_config_path = os.path.join(population_path, agent_config_file)
                opponent_config_path = os.path.join(population_path, opponent_config_file)

                with open(agent_config_path, "r") as f:
                        agent_config = yaml.safe_load(f) or {}

                with open(opponent_config_path, "r") as f:
                        opponent_config = yaml.safe_load(f) or {}

                agent_architecture = str(agent_config.get("MODEL_IDENTIFIER", "")).strip().lower()
                opponent_architecture = str(opponent_config.get("MODEL_IDENTIFIER", "")).strip().lower()

                # ----- load agent -----
                if agent_architecture == "rainbow":
                    agent = RainbowAgent(
                         n_observations=n_observations,
                         n_actions=n_actions,
                         config_path=agent_config_path
                    )
                    agent.load_dict(agent_load_path)
                elif agent_architecture == "ddpg":
                    agent = DDPGAgent(
                        observation_space=base_obs_space,
                        action_space=base_act_space,
                        config_path=agent_config_path
                    )
                    agent.load_dict(agent_load_path)
                elif agent_architecture == "tdmpc2":
                    agent =  TDMPC2Agent(
                        action_space=base_act_space,
                        observation_space=base_obs_space,
                        config_path=agent_config_path
                    )
                    agent.load_dict(agent_load_path)
                else:
                    raise NotImplementedError
                
                # ----- load opponent -----
                if opponent_architecture == "rainbow":
                    opponent = RainbowAgent(
                         n_observations=n_observations,
                         n_actions=n_actions,
                         config_path=opponent_config_path
                    )
                    opponent.load_dict(opponent_load_path)
                elif opponent_architecture == "ddpg":
                    opponent = DDPGAgent(
                        observation_space=base_obs_space,
                        action_space=base_act_space,
                        config_path=opponent_config_path
                    )
                    opponent.load_dict(opponent_load_path)
                elif opponent_architecture == "tdmpc2":
                    opponent =  TDMPC2Agent(
                        action_space=base_act_space,
                        observation_space=base_obs_space,
                        config_path=opponent_config_path
                    )
                    opponent.load_dict(opponent_load_path)
                else:
                    raise NotImplementedError


                

                player1_wins = agent_against_agent_eval(agent, opponent, num_episodes = num_episodes)
                
                result_dict[agents[i]].extend(list(player1_wins))
                result_dict[agents[j]].extend(list(1 - player1_wins))
                

        
        

        winrate_stats = dict()

        for i in range(N):
            i_mean_winrate = np.mean(result_dict(agents[i]))
            i_std_winrate = np.std(result_dict(agents[i]))
            winrate_stats[agents[i]] = (i_mean_winrate, i_std_winrate)

        """plt.barh(list(average_winrate.keys()), list(average_winrate.values()))
        plt.xlabel("Average Winrate")
        plt.ylabel("Agents")
        plt.title("Evaluation of Agent pool")
        plt.tight_layout()
        plt.savefig(os.path.join(population_path, "evaluation_against_agent_pool.svg"), dpi=300)
        plt.close()"""



        aw_dataframe =pd.DataFrame(winrate_stats)
        aw_dataframe.to_csv(os.path.join(population_path, "evaluation_against_agent_pool.csv"))


def evaluate_agents_against_basic(
            population_path,
            weak=True,
            num_episodes = 50):
        
        env = gym.envs.make("Hockey-One-v0", mode=0, weak_opponent=True)
        base_obs_space = env.observation_space
        base_act_space = env.action_space

        env = DiscreteActionWrapperHockey(env)
        state, info = env.reset()

        n_observations = len(state)
        n_actions = env.action_space.n
        if weak:
             opponent = "weakopp"
        else:
             opponent = "strongopp"

        
        agents = [f for f in os.listdir(population_path) if f.split(".")[1] == "pth"]
        
        N = len(agents)

        average_winrate = dict()
        
        start = time.time()
        for i in range(N):
            end = time.time()
            print(f"eval agents for agent " + agents[i] + f" {end-start} seconds passed")


            agent_config_file = agents[i].split(".")[0] + ".yaml"
            

            agent_load_path = os.path.join(population_path, agents[i])
            

            agent_config_path = os.path.join(population_path, agent_config_file)
            

            with open(agent_config_path, "r") as f:
                    agent_config = yaml.safe_load(f) or {}

            

            agent_architecture = str(agent_config.get("MODEL_IDENTIFIER", "")).strip().lower()
            

            # ----- load agent -----
            if agent_architecture == "rainbow":
                agent = RainbowAgent(
                        n_observations=n_observations,
                        n_actions=n_actions,
                        config_path=agent_config_path
                )
                agent.load_dict(agent_load_path)
            elif agent_architecture == "ddpg":
                agent = DDPGAgent(
                    observation_space=base_obs_space,
                    action_space=base_act_space,
                    config_path=agent_config_path
                )
                agent.load_dict(agent_load_path)
            elif agent_architecture == "tdmpc2":
                agent =  TDMPC2Agent(
                    action_space=base_act_space,
                    observation_space=base_obs_space,
                    config_path=agent_config_path
                )
                agent.load_dict(agent_load_path)
            else:
                raise NotImplementedError
            
        
            

            agent_won_proportion = agent_against_agent_eval(agent, opponent, num_episodes = num_episodes)
            average_winrate[agents[i]] = agent_won_proportion

        
        aw_dataframe =pd.DataFrame(average_winrate)
        aw_dataframe.to_csv(os.path.join(population_path, "evaluation_against_{opponent}.csv"))

        plt.barh(list(average_winrate.keys()), list(average_winrate.values()))
        plt.xlabel("Average Winrate")
        plt.ylabel("Agents")
        plt.title(f"Pool eval against {opponent}")
        plt.tight_layout()
        plt.savefig(os.path.join(population_path, f"evaluation_against_{opponent}.svg"), dpi=300)
        plt.close()


def agent_against_agent_eval(
        player1, 
        player2= "basicopp",
        environment = 'Hockey-One-v0',
        num_episodes = 50):
    
    if environment == 'Hockey-One-v0':
        env = h_env.HockeyEnv()
        obs, info = env.reset()
        obs_agent2 = env.obs_agent_two()
        score = {"player1": 0, "player2": 0}

        if isinstance(player2, str):
            key = player2.lower()
            if key in {"basicopp", "basicopponent"}:
                player2 =  h_env.BasicOpponent()
            if key in {"strongopp", "strongopponent"}:
                player2 = h_env.BasicOpponent(weak=False)
            if key in {"weakopp", "weakopponent"}:
                player2 = h_env.BasicOpponent(weak=True)

        for _ in range(num_episodes):
            d = False
            obs, info = env.reset()
            while not d:
                a1 = resolve_eval_action(player1, env, obs)
                a2 = resolve_eval_action(player2, env, obs_agent2)

                obs, r, d, _, info = env.step(np.hstack([a1,a2]))   
                obs_agent2 = env.obs_agent_two()
            if info["winner"] == 1:
                score["player1"] += 1
            else:
                score["player2"] += 1

    env.close()

    player1_wins = np.ones(shape=(score["player1"],)).extend(np.zeros(shape=(score["player2"],)))


    return player1_wins


def resolve_eval_action(player, env, obs):
        if isinstance(player, RainbowAgent):
            discrete_action = player.act(env=env, state=obs, greedy=True)
            return discrete_to_continuous(discrete_action)
        if isinstance(player, Agent):
            return player.act(env=env, state=obs, greedy=True)
        return player.act(obs)


def discrete_to_continuous(discrete_action):
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


if __name__ == "__main__":
    num_episodes = 1
    evaluate_agents(
          population_path="/home/stud308/hockey007/self_play_opponent_pool",
          num_episodes=num_episodes
    )
    
    """
    evaluate_agents_against_basic(
          population_path="/home/stud308/hockey007/self_play_opponent_pool",
          weak=False,
          num_episodes=num_episodes
    )
    evaluate_agents_against_basic(
          population_path="/home/stud308/hockey007/self_play_opponent_pool",
          weak=True,
          num_episodes=num_episodes
    )"""