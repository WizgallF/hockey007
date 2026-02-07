import os
import sys
import tempfile
import unittest

import numpy as np
import torch
from gymnasium import spaces


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from agents.DDPGAgent import DDPGAgent


CONFIG_PATH = os.path.join(ROOT, "configs", "ddpg_config.yaml")


def _make_agent() -> DDPGAgent:
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    return DDPGAgent(obs_space, act_space, config_path=CONFIG_PATH)


class TestDDPGTargets(unittest.TestCase):
    def test_copy_updates_q2_target(self) -> None:
        agent = _make_agent()
        if not agent._config["TWIN_DELAYED"]:
            self.skipTest("TWIN_DELAYED disabled in config.")

        for p in agent.Q2.parameters():
            p.data.add_(1.0)
        agent._copy_nets()

        q2_state = agent.Q2.state_dict()
        q2_t_state = agent.Q2_target.state_dict()
        for key, value in q2_state.items():
            self.assertTrue(torch.allclose(value, q2_t_state[key]))

    def test_soft_update_updates_q2_target(self) -> None:
        agent = _make_agent()
        if not agent._config["TWIN_DELAYED"]:
            self.skipTest("TWIN_DELAYED disabled in config.")

        before = {k: v.clone() for k, v in agent.Q2_target.state_dict().items()}
        for p in agent.Q2.parameters():
            p.data.add_(1.0)

        agent._soft_update(0.5)
        after = agent.Q2_target.state_dict()

        changed = any(not torch.allclose(before[k], after[k]) for k in before)
        self.assertTrue(changed, "Expected Q2_target parameters to change after soft update.")

    def test_save_load_includes_q2(self) -> None:
        agent = _make_agent()
        if not agent._config["TWIN_DELAYED"]:
            self.skipTest("TWIN_DELAYED disabled in config.")

        for p in agent.Q2.parameters():
            p.data.add_(1.0)
        for p in agent.Q2_target.parameters():
            p.data.add_(2.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            agent.save_dict(tmpdir)
            ckpt_path = os.path.join(tmpdir, f"{agent.MODEL_IDENTIFIER}.pth")

            agent2 = _make_agent()
            agent2.load_dict(ckpt_path)

            q2_state = agent.Q2.state_dict()
            q2_t_state = agent.Q2_target.state_dict()

            for key, value in q2_state.items():
                self.assertTrue(torch.allclose(value, agent2.Q2.state_dict()[key]))
            for key, value in q2_t_state.items():
                self.assertTrue(torch.allclose(value, agent2.Q2_target.state_dict()[key]))


if __name__ == "__main__":
    unittest.main()
