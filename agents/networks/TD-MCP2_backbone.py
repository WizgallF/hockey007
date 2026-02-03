import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class LatentState:
    z: torch.Tensor  

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=256, depth=2):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Encoder(nn.Module):
    """s -> z (latent)"""
    def __init__(self, obs_dim, z_dim, hidden=256):
        super().__init__()
        self.net = MLP(obs_dim, z_dim, hidden=hidden, depth=2)

    def forward(self, obs):
        z = self.net(obs)
        z = F.layer_norm(z, (z.shape[-1],))
        return z


class DynamicsModel(nn.Module):
    """(z, a) -> z_next"""
    def __init__(self, z_dim, act_dim, hidden=256):
        super().__init__()
        self.net = MLP(z_dim + act_dim, z_dim, hidden=hidden, depth=2)

    def forward(self, z, a):
        z_next = self.net(torch.cat([z, a], dim=-1))
        z_next = F.layer_norm(z_next, (z_next.shape[-1],))
        return z_next


class RewardModel(nn.Module):
    """(z, a) -> r"""
    def __init__(self, z_dim, act_dim, hidden=256):
        super().__init__()
        self.net = MLP(z_dim + act_dim, 1, hidden=hidden, depth=2)

    def forward(self, z, a):
        return self.net(torch.cat([z, a], dim=-1)).squeeze(-1)


class QModel(nn.Module):
    """(z, a) -> (q1, q2) double Q for TD-MPC2-style training."""
    def __init__(self, z_dim, act_dim, hidden=256):
        super().__init__()
        self.q1 = MLP(z_dim + act_dim, 1, hidden=hidden, depth=2)
        self.q2 = MLP(z_dim + act_dim, 1, hidden=hidden, depth=2)

    def forward(self, z, a):
        x = torch.cat([z, a], dim=-1)
        q1 = self.q1(x).squeeze(-1)
        q2 = self.q2(x).squeeze(-1)
        return q1, q2

    def min(self, z, a):
        q1, q2 = self.forward(z, a)
        return torch.min(q1, q2)


class PolicyPrior(nn.Module):
    """z -> a (proposal / warm-start for planning)"""
    def __init__(self, z_dim, act_dim, hidden=256):
        super().__init__()
        self.net = MLP(z_dim, act_dim, hidden=hidden, depth=2)

    def forward(self, z):
        return torch.tanh(self.net(z))  # assumes actions are scaled to [-1, 1]


class TDMPC2Model(nn.Module):
    """
    TD-MPC2-style model:
      z      = enc(obs)
      z_next = dyn(z, a)
      r_hat  = reward(z, a)
      q_hat  = q(z, a)
      a0     = pi(z)   (policy prior for MPC)
    """
    def __init__(self, obs_dim: int, act_dim: int, z_dim: int = 256, hidden: int = 512):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.z_dim = z_dim

        self.encoder = Encoder(obs_dim, z_dim, hidden=hidden)
        self.dynamics = DynamicsModel(z_dim, act_dim, hidden=hidden)
        self.reward = RewardModel(z_dim, act_dim, hidden=hidden)
        self.q = QModel(z_dim, act_dim, hidden=hidden)
        self.pi = PolicyPrior(z_dim, act_dim, hidden=hidden)

    def encode(self, obs) -> LatentState:
        return LatentState(z=self.encoder(obs))

    def step(self, state: LatentState, a: torch.Tensor):
        """
        One-step latent rollout used by MPC.
        state.z: [B, z_dim]
        a:       [B, act_dim]
        returns: next_state, r_hat
        """
        r_hat = self.reward(state.z, a)
        z_next = self.dynamics(state.z, a)
        return LatentState(z=z_next), r_hat

    @torch.no_grad()
    def rollout(self, z0: torch.Tensor, action_seq: torch.Tensor):
        """
        Rollout for MPC scoring.
        z0: [B, z_dim]
        action_seq: [B, H, act_dim]
        Returns:
          zs: [B, H+1, z_dim]
          rs: [B, H]
        """
        B, H, _ = action_seq.shape
        z = z0
        zs = [z]
        rs = []
        for t in range(H):
            a = action_seq[:, t]
            r = self.reward(z, a)
            z = self.dynamics(z, a)
            rs.append(r)
            zs.append(z)
        return torch.stack(zs, dim=1), torch.stack(rs, dim=1)
