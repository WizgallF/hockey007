import hockey.hockey_env as h_env
import numpy as np


class Envwrapper():
    def __init__(
            self,
            env,
            name,
            player2):
        self.env = env
        self.name = name
        self.player2 = self._resolve_player(player2)

    def reset(self):
        return self.env.reset()

    def step(self, action_p1):
        obs_agent2 = self.env.obs_agent_two()
        action_p2 = self.player2.act(obs_agent2)
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
        return player
        