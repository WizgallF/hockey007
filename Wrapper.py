import numpy as np
import gymnasium as gym
from gymnasium import spaces
import hockey.hockey_env as h_env
from agents.RainbowAgent import RainbowAgent


class Envwrapper():
    def __init__(
            self,
            env,
            player2,
            discrete_actions = False):
        self.env = env
        self.player2 = self._resolve_player(player2)
        self.discrete_actions = discrete_actions

    def reset(self):
        return self.env.reset()

    def step(self, action_p1):
        obs_agent2 = self.env.obs_agent_two()

        if isinstance(self.player2, RainbowAgent):
            action_p2 = self.player2.act(env=self.env, state=obs_agent2, greedy=True)
        else:
            action_p2 = self.player2.act(obs_agent2)

        if self.discrete_actions:
            return self.env.step(np.hstack([self._convert_action(action_p1), self._convert_action(action_p2)]))
        else:
            return self.env.step(np.hstack([action_p1, action_p2]))


    def _resolve_player(self, player):
        if isinstance(player, str):
            key = player.lower()
            if key in {"basicopp", "basicopponent"}:
                return h_env.BasicOpponent()
            if key in {"strongopp", "strongopponent"}:
                return h_env.BasicOpponent(weak=False)
            if key in {"weakopp", "weakopponent"}:
                return h_env.BasicOpponent(weak=True)
        if isinstance(player, RainbowAgent):
            return player
        
    def _convert_action(self, discrete_action):
        action_cont = [(discrete_action == 1) * -1.0 + (discrete_action == 2) * 1.0,  # player x
                   (discrete_action == 3) * -1.0 + (discrete_action == 4) * 1.0,  # player y
                   (discrete_action == 5) * -1.0 + (discrete_action == 6) * 1.0]  # player angle
        
        action_cont.append((discrete_action == 7) * 1.0)

        return action_cont

    

class DiscreteActionWrapperPendulum(gym.ActionWrapper):
    def __init__(self, env: gym.Env, bins = 5):
        """A wrapper for converting a 1D continuous actions into discrete ones.
        Args:
            env: The environment to apply the wrapper
            bins: number of discrete actions
        """
        assert isinstance(env.action_space, spaces.Box)
        super().__init__(env)
        self.bins = bins
        self.orig_action_space = env.action_space
        self.action_space = spaces.Discrete(self.bins)

    def action(self, action):
        """ discrete actions from low to high in 'bins'
        Args:
            action: The discrete action
        Returns:
            continuous action
        """
        return self.orig_action_space.low + action/(self.bins-1.0)*(self.orig_action_space.high-self.orig_action_space.low)
    

class DiscreteActionWrapperHockey(gym.ActionWrapper):
    def __init__(self, env: gym.Env):
        """A wrapper for converting a 4D [-1; 1] intervall into 8 discrete ones.
        Args:
            env: The environment to apply the wrapper
        """
        assert isinstance(env.action_space, spaces.Box)
        super().__init__(env)
        self.orig_action_space = env.action_space
        self.action_space = spaces.Discrete(8)

    def action(self, discrete_action):
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



class BatchsizeObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

    def observation(self, obs):
        return np.expand_dims(obs, axis=0)
