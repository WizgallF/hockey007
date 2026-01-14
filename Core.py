import numpy as np
import hockey.hockey_env as h_env
import gymnasium as gym
from Train import Training
from agents.Agents import Agent
import time 


class Core:
    def load_agent(self):
        pass

    def train_agent(
            self, 
            agent: Agent = None, 
            env: gym.envs = None, 
            save_intermediate_agents: bool = False):
        train_class = Training(env, agent)
        train_class.train()
        pass

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
