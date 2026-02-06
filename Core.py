import numpy as np
import gymnasium as gym
import hockey.hockey_env as h_env
import time 
from Train import Training
from agents.AgentBaseclass import Agent
from agents.RainbowAgent import RainbowAgent
from agents.DDPGAgent import DDPGAgent
from Wrapper import Envwrapper, DiscreteActionWrapperPendulum, DiscreteActionWrapperHockey
from agents.TDMPC2Agent import TDMPC2Agent
from gymnasium import spaces


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
            bins = 5,
            num_parallel_envs = 1):
        
        print(agent_name)
        
        if env_name == "Pendulum-v1":
            def make_env():
                return gym.make(env_name)
        elif env_name == "Hockey-One-v0":
            def make_env():
                return gym.envs.make("Hockey-One-v0", mode=0, weak_opponent=True)
        else:
            raise NotImplementedError

        if num_parallel_envs > 1:
            if agent_name == "rainbow":
                if env_name == "Pendulum-v1":
                    env = gym.vector.SyncVectorEnv(
                        [lambda: DiscreteActionWrapperPendulum(make_env(), bins) for _ in range(num_parallel_envs)]
                    )
                else:
                    env = gym.vector.SyncVectorEnv(
                        [lambda: DiscreteActionWrapperHockey(make_env()) for _ in range(num_parallel_envs)]
                    )
            else:
                env = gym.vector.SyncVectorEnv([make_env for _ in range(num_parallel_envs)])
            base_obs_space = env.single_observation_space
            base_act_space = env.single_action_space
        else:
            env = make_env()
            if agent_name == "rainbow":
                env = DiscreteActionWrapperPendulum(env, bins) if env_name == "Pendulum-v1" else DiscreteActionWrapperHockey(env)
            base_obs_space = env.observation_space
            base_act_space = env.action_space

        state, info = env.reset()
        if num_parallel_envs > 1:
            n_observations = len(state[0])
        else:
            n_observations = len(state)

        if agent_name == "rainbow":
            n_actions = base_act_space.n

    
        if agent_name == "rainbow": 
            agent = RainbowAgent(
                n_observations,
                n_actions,
                verbose)
        elif agent_name == "ddpg":
            agent = DDPGAgent(
                base_obs_space,
                base_act_space,
                verbose=verbose)
        elif agent_name == "tdmpc2":
            agent =  TDMPC2Agent(
                base_act_space,
                base_obs_space
            )
        else:
            raise NotImplementedError
        

        
        train_class = Training(agent, env, base_dir, save_intermediate_agents, verbose)
        train_class.train()

    

    def evaluate(self):
        pass

    def train_agent_self_play(
            self, 
            agent_name: str = None, 
            env_name: str = None, 
            base_dir = "experiments",
            save_intermediate_agents: bool = False,
            verbose=False,
            bins = 5):
        
        print(agent_name)
            
        if env_name == "Hockey-One-v0":
            env = h_env.HockeyEnv()
            if agent_name == "rainbow":
                proxy_env = DiscreteActionWrapperHockey(env)
                n_actions = proxy_env.action_space.n
            state, info = proxy_env.reset()
            n_observations = len(state)

    
        if agent_name == "rainbow": 
            agent = RainbowAgent(
                n_observations,
                n_actions,
                verbose)
            opponent = RainbowAgent(
                n_observations,
                n_actions,
                verbose)
        else:
            raise NotImplementedError
        

        
        train_class = Training(agent, env, base_dir, save_intermediate_agents, verbose)
        discrete_actions = agent_name == "rainbow"
        train_class.train_self_play(opponent, discrete_actions)

    def agent_against_human(self):
        pass

    def agent_against_agent_eval(self, agent1, agent2):
        pass

    def play(
            self,
            environment: str = 'Hockey-One-v0',
            player1: str | Agent = 'BasicOpp',
            player2: str | Agent = 'BasicOpp',
            num_episodes: int = 50,
            agent1_type: str = 'rainbow',
            agent2_type: str = 'rainbow',
            agent1_path: str = None,
            agent2_path: str = None,
            agent1_config: str = None,
            agent2_config: str = None
            ):
        
        if environment == 'Hockey-One-v0':
            env = h_env.HockeyEnv()
            player1 = self._resolve_player(player1, env, player_id=1, agent_path=agent1_path, config_path=agent1_config)
            player2 = self._resolve_player(player2, env, player_id=2, agent_path=agent2_path, config_path=agent2_config)
            obs, info = env.reset()
            env.render()
            time.sleep(1)
            obs_agent2 = env.obs_agent_two()
            score = {"player1": 0, "player2": 0}
            for episode in range(num_episodes):
                d = False
                obs, info = env.reset()
                while not d:
                    time.sleep(0.05)
                    env.render()
                    
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
                print(
                    f"After game {episode + 1}: [{score['player1']} - {score['player2']}] "
                    f"Player {2 - info['winner']} WON!"
                )

        env.close()

    def _resolve_player(
            self, 
            player, 
            env, 
            player_id: int, 
            agent_path: str = None, 
            config_path: str = None):
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
            if key in {"rainbow"}:
                env = DiscreteActionWrapperHockey(env)
                n_actions = env.action_space.n
                state, info = env.reset()
                n_observations = len(state)

                agent = RainbowAgent(n_observations, n_actions, config_path=config_path)
                agent.load_dict(load_path=agent_path)
                return agent
            if key in {"ddpg"}:
                single_player_action_space = spaces.Box(
                    low=env.action_space.low[:4],
                    high=env.action_space.high[:4],
                    dtype=env.action_space.dtype,
                )
                agent = DDPGAgent(
                    env.observation_space,
                    single_player_action_space,
                    config_path=config_path
                )
                agent.load_dict(load_path=agent_path)
                return agent
        return player
    
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
    
