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
        print(results)
        start = time.time()
        for i in range(N):
            end = time.time()
            print(f"eval agents for agent " + agents[i] + f" {end-start} seconds passed")

            for j in range(i+1, N):
                agent_architecture = agents[i].split("_")[0]
                opponent_architecture = agents[j].split("_")[0]

                agent_config_file = agents[i].split(".")[0] + ".yaml"
                opponent_config_file = agents[j].split(".")[0] + ".yaml"

                agent_load_path = os.path.join(population_path, agents[i])
                opponent_load_path = os.path.join(population_path, agents[j])

                agent_config_path = os.path.join(population_path, agent_config_file)
                opponent_config_path = os.path.join(population_path, opponent_config_file)

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


                

                agent_won_proportion = agent_against_agent_eval(agent, opponent, num_episodes = num_episodes)
                
                results[i][j] = agent_won_proportion
                results[j][i] =  1 - agent_won_proportion

        
        

        average_winrate = dict()

        for i in range(N):
            i_average_winrate = np.sum(results[i]) / (N-1)
            average_winrate[agents[i]] = float(i_average_winrate)

        plt.barh(list(average_winrate.keys()), list(average_winrate.values()))
        plt.xlabel("Average Winrate")
        plt.ylabel("Agents")
        plt.title("Evaluation of Agent pool")
        plt.tight_layout()
        plt.savefig(os.path.join(population_path, "evaluation.png"), dpi=300)
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

    return round(score["player1"]/num_episodes, ndigits=3)


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
    evaluate_agents(
          population_path="/home/stud217/hockey007/evaluation/eval_agent_pool"
    )