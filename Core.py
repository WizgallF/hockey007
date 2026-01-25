import numpy as np
import gymnasium as gym
import hockey.hockey_env as h_env
import time 
from Train import Training
from agents.AgentBaseclass import Agent
from agents.RainbowAgent import RainbowAgent
from Wrapper import Envwrapper, DiscreteActionWrapperPendulum, DiscreteActionWrapperHockey



class Core:
    def load_agent(self):
        pass

    
    def train_agent(
            self, 
            agent_name: str = None, 
            env_name: str = None, 
            base_dir = "experiments",
            save_intermediate_agents: bool = False,
            verbose=False,
            bins = 5):
        
        print(agent_name)
        
        if env_name == "Pendulum-v1":
            env = gym.make(env_name)
            env = DiscreteActionWrapperPendulum(env, bins)
            n_actions = env.action_space.n
            state, info = env.reset()
            n_observations = len(state)
            
        elif env_name == "Hockey-One-v0":
            env = gym.envs.make("Hockey-One-v0", mode=0, weak_opponent=True)
            env = DiscreteActionWrapperHockey(env)
            n_actions = env.action_space.n
            state, info = env.reset()
            n_observations = len(state)

        

        if agent_name == "rainbow": 
            agent = RainbowAgent(
                n_observations,
                n_actions,
                verbose)
        else:
            raise NotImplementedError
        

        
        train_class = Training(agent, env, base_dir, save_intermediate_agents, verbose)
        train_class.train()

    def evaluate(self):
        pass

    def train_self_play(self):
        pass

    def agent_against_human(self):
        pass

    def agent_against_agent_eval(self, agent1, agent2):
        pass

    def play(
        self,
        environment: str = 'hockey',
        player1: str | Agent = 'BasicOpp',
        player2: str | Agent = 'BasicOpp'):
        if environment == 'hockey':
            env = h_env.HockeyEnv()
            player1 = self._resolve_player(player1, env, player_id=1)
            player2 = self._resolve_player(player2, env, player_id=2)
            obs, info = env.reset()
            env.render()
            time.sleep(1)
            obs_agent2 = env.obs_agent_two()
            for _ in range(10000):
                time.sleep(0.1)
                env.render()
                a1 = player1.act(obs) 
                a2 = player2.act(obs_agent2)
                obs, r, d, _, info = env.step(np.hstack([a1,a2]))   
                obs_agent2 = env.obs_agent_two()
                if d: break
        env.close()

    def _resolve_player(self, player, env, player_id: int):
        if isinstance(player, str):
            key = player.lower()
            if key in {"human", "humanopponent"}:
                return h_env.HumanOpponent(env=env, player=player_id)
            if key in {"basicopp", "basicopponent"}:
                return h_env.BasicOpponent()
            if key in {"strongopp", "strongopponent"}:
                return h_env.BasicOpponent(weak=False)
            if key in {"weakopp", "weakopponent"}:
                return h_env.BasicOpponent(weak=True)
        return player
    


