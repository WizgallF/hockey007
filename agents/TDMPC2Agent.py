from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import time
import yaml
import os
import json
import random
import torch.optim as optim
from datetime import datetime
from agents.AgentBaseclass import Agent
from agents.utils.TDMPC2Utils import TDMPC2ReplayBuffer
from agents.networks.TD_MPC2_backbone import (
    TDMPC2,
    Encoder, 
    DynamicsModel, 
    RewardModel, 
    QModel, 
    LatentState
)


class TDMPC2Agent(Agent):
    def __init__(
            self,
            action_space,
            observation_space,
            verbose = False
            ):

        # ------ load configs from "tdmpc_config.yaml" ------
        with open("configs/tdmpc_config.yaml", 'r') as f:
            config = yaml.safe_load(f) or {}
        self.configs = config
        self.__dict__.update(config)

        # ------ device ------
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Determine flat dimensions for Box/Discrete spaces
        if hasattr(action_space, "shape") and action_space.shape is not None:
            self.act_dim = int(np.prod(action_space.shape))
        elif hasattr(action_space, "n"):
            self.act_dim = int(action_space.n)
        else:
            self.act_dim = len(action_space)

        if hasattr(observation_space, "shape") and observation_space.shape is not None:
            self.obs_dim = int(np.prod(observation_space.shape))
        elif hasattr(observation_space, "n"):
            self.obs_dim = int(observation_space.n)
        else:
            self.obs_dim = len(observation_space)
        self.verbose = verbose
        self.horizon = getattr(self, "HORIZON", getattr(self, "TRAIN_HORIZON", 12))
        self.t = 0

        # ------ training schedule ------
        self.NUM_EPISODES = getattr(self, "NUM_EPISODES", 1000)
        self.START_TRAINING = getattr(self, "START_TRAINING", 0)

        # ------ model sizes ------
        self.Z_DIM = getattr(self, "Z_DIM", 256)
        self.HIDDEN_DIM = getattr(self, "HIDDEN_DIM", 512)

        # ------ optimizer params ------
        self.LR = getattr(self, "LR", 3e-4)
        self.ADAM_BETA_1 = getattr(self, "ADAM_BETA_1", 0.9)
        self.ADAM_BETA_2 = getattr(self, "ADAM_BETA_2", 0.999)
        self.ADAM_EPS = getattr(self, "ADAM_EPS", 1e-8)

        # ------ replay buffer params ------
        self.CAPACITY = getattr(self, "CAPACITY", int(1e6))
        self.PRIORITIZED = getattr(self, "PRIORITIZED", False)
        self.ALPHA = getattr(self, "ALPHA", 0.6)
        self.BETA = getattr(self, "BETA", 0.4)
        self.EPSILON = getattr(self, "EPSILON", 1e-6)
        self.SEED = getattr(self, "SEED", None)

        # ---- MPC defaults (override via YAML if you have them) ----
        self.MPC_SAMPLES = getattr(self, "MPC_SAMPLES", 512)      # number of sampled sequences
        self.MPC_ITERS   = getattr(self, "MPC_ITERS", 3)          # MPPI iterations
        self.MPC_SIGMA   = getattr(self, "MPC_SIGMA", 0.4)        # std of action noise
        self.MPC_TEMP    = getattr(self, "MPC_TEMP", 1.0)         # temperature for softmax weights
        self.GAMMA       = getattr(self, "GAMMA", 0.99)           # discount

        # Optional exploration noise on executed action
        self.EXPL_NOISE  = getattr(self, "EXPL_NOISE", 0.1)

        # Warm-start mean action sequence (in [-1,1])
        self._a_mean = torch.zeros(self.horizon, self.act_dim, device=self.device)

        self.MODEL_IDENTIFIER = getattr(self, "MODEL_IDENTIFIER", "TD-MPC2-Agent")
        self.model = TDMPC2(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            z_dim=self.Z_DIM,
            hidden=self.HIDDEN_DIM,
        )
        self.model.to(self.device)

        # Target nets for TD targets (only need Q + pi, but easiest is copy whole model and use only parts)
        self.target_model = copy.deepcopy(self.model).to(self.device)
        for p in self.target_model.parameters():
            p.requires_grad_(False)

        # Soft update rate for target
        self.TAU = getattr(self, "TAU", 0.01)

        # Training params
        self.BATCH_SIZE = getattr(self, "BATCH_SIZE", 256)
        self.TRAIN_HORIZON = getattr(self, "TRAIN_HORIZON", self.horizon)
        self.LAMBDA_W = getattr(self, "LAMBDA_W", 0.9)      # time weighting across horizon
        self.GRAD_CLIP = getattr(self, "GRAD_CLIP", 10.0)

        # Optional: separate optimizer for policy prior (cleaner)
        self.PI_LR = getattr(self, "PI_LR", self.LR)
        self.pi_optimizer = optim.Adam(self.model.pi.parameters(), lr=self.PI_LR)
        self.replay_buffer = TDMPC2ReplayBuffer(
            capacity_steps=self.CAPACITY,
            prioritized=self.PRIORITIZED,
            alpha=self.ALPHA,
            beta=self.BETA,
            eps=self.EPSILON,
            seed=self.SEED
        )

        # ------ optimizer ------
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.LR, betas=(self.ADAM_BETA_1, self.ADAM_BETA_2), eps=self.ADAM_EPS)

    def _soft_update(self, tau: float) -> None:
        with torch.no_grad():
            for p_targ, p in zip(self.target_model.parameters(), self.model.parameters()):
                p_targ.data.mul_(1.0 - tau).add_(tau * p.data)

    
    @torch.no_grad()
    def act(self, env, state, episode_i, statistics) -> np.ndarray:
        # --- 0) prep tensors ---
        self.model.eval()
        obs = torch.as_tensor(state, dtype=torch.float32, device=self.device).view(1, -1)

        # --- 1) encode ---
        z0 = self.model.encoder(obs)  # [1, z_dim]

        H = self.horizon
        N = self.MPC_SAMPLES
        iters = self.MPC_ITERS
        act_dim = self.act_dim

        # Determine env action bounds if they exist
        low, high = None, None

        # Helper: map [-1,1] -> env bounds (if present)
        def unnormalize(a_tanh):
            if low is None or high is None:
                return a_tanh
            return low + (a_tanh + 1.0) * 0.5 * (high - low)

        # Helper: discount vector [H]
        discounts = (self.GAMMA ** torch.arange(H, device=self.device, dtype=torch.float32)).view(1, H)

        # --- 2) MPPI: iterative refinement of mean action sequence ---
        mean = self._a_mean.clone()  # [H, act_dim]
        sigma = self.MPC_SIGMA

        for _ in range(iters):
            # sample around mean: [N, H, act_dim]
            noise = torch.randn(N, H, act_dim, device=self.device) * sigma
            a_seq = mean.unsqueeze(0) + noise
            a_seq = torch.clamp(a_seq, -1.0, 1.0)

            # optional: inject some policy-prior sequences to help early training
            # e.g., replace first K with pi(z) repeated (simple warm-start)
            K = min(32, N)
            if K > 0:
                a0 = self.model.pi(z0).repeat(K, 1)  # [K, act_dim], tanh already
                a_seq[:K, 0, :] = a0

            # rollout: requires z0 [B,z_dim] and action_seq [B,H,act_dim]
            z0_rep = z0.repeat(N, 1)
            zs, rs = self.model.rollout(z0_rep, a_seq)  # zs: [N,H+1,z_dim], rs: [N,H]

            # terminal value: V ≈ min(Q(z_H, pi(z_H)))
            zH = zs[:, -1, :]                    # [N,z_dim]
            aH = self.model.pi(zH)               # [N,act_dim]
            vH = self.model.q.min(zH, aH)        # [N]

            # total score: discounted sum of rewards + discounted terminal value
            # rs: [N,H]
            returns = (rs * discounts).sum(dim=1) + (self.GAMMA ** H) * vH  # [N]

            # MPPI weights (softmax over returns)
            # stabilize by subtracting max
            scaled = (returns - returns.max()) / max(self.MPC_TEMP, 1e-6)
            w = torch.softmax(scaled, dim=0)  # [N]

            # update mean: weighted average of sampled sequences
            mean = (w.view(N, 1, 1) * a_seq).sum(dim=0)  # [H,act_dim]

        # --- 3) pick action: first action of planned mean ---
        a = mean[0]  # [-1,1] range

        # --- 4) exploration noise on executed action (optional) ---
        if self.EXPL_NOISE > 0 and self.model.training is False:
            # use a simple schedule if you want: decay with episode_i or self.t
            a = torch.clamp(a + self.EXPL_NOISE * torch.randn_like(a), -1.0, 1.0)

        # --- 5) warm-start for next call: shift mean ---
        mean_shifted = torch.roll(mean, shifts=-1, dims=0)
        mean_shifted[-1].zero_()  # or keep last action, or sample from pi(z) later
        self._a_mean = mean_shifted.detach()

        # --- 6) convert to env action scale and return ---
        a_env = unnormalize(a).cpu().numpy()
        return a_env

        

    def observe(
            self,
            state: np.ndarray, 
            action: int, 
            reward: int, 
            next_state: np.ndarray,
            terminated: bool
            ):
        """
        Saves observed transition in the replay buffer.
        ----------
        Parameters:
            state: Current state
            action: The action selected in current state
            reward: The reward given in current state for the selected action
            next_state: The next state of the environment
            terminated: Whether the episode has terminated after current transition

        """
        self.replay_buffer.add(state, action, reward, next_state, terminated)
        

    
    def update(
            self,
            statistics: dict[str, list] | None = None,
            ):
        """
        One training iteration of Ranbow Q-Learning with BATCH_SIZE samples drawn from the replay buffer.
        ----------
        Parameters:
            statistics: The training statistics
        """
        if len(self.replay_buffer) < max(self.BATCH_SIZE, self.TRAIN_HORIZON):
            return

        H = self.TRAIN_HORIZON
        B = self.BATCH_SIZE

        # Sample contiguous segments
        obs, act, rew, done, isw, idxes = self.replay_buffer.sample(
            batch_size=B,
            horizon=H,
        )

        # ---- to torch ----
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)          # [B,H+1,obs_dim]
        act = torch.as_tensor(act, dtype=torch.float32, device=self.device)          # [B,H,act_dim]
        rew = torch.as_tensor(rew, dtype=torch.float32, device=self.device)          # [B,H]
        done = torch.as_tensor(done, dtype=torch.float32, device=self.device)        # [B,H] (0/1)
        isw = torch.as_tensor(isw, dtype=torch.float32, device=self.device)          # [B]

        not_done = 1.0 - done

        # Time weights (TD-MPC2 uses a decay across horizon; call it lambda_w here)
        t_w = (self.LAMBDA_W ** torch.arange(H, device=self.device, dtype=torch.float32))  # [H]
        t_w = t_w.view(1, H)  # broadcast to [B,H]

        # ---- encode all observations ----
        # z_seq: [B,H+1,z_dim]
        z_seq = self.model.encoder(obs.view(B*(H+1), -1)).view(B, H+1, -1)
        z = z_seq[:, :-1, :]      # [B,H,z]
        z_next_targ = z_seq[:, 1:, :].detach()  # stop-grad target [B,H,z]

        # ---- predict next z via dynamics ----
        z_pred = self.model.dynamics(
            z.reshape(B*H, -1),
            act.reshape(B*H, -1)
        ).view(B, H, -1)

        # ---- representation / dynamics consistency loss ----
        rep_loss_t = F.mse_loss(z_pred, z_next_targ, reduction="none").mean(dim=-1)  # [B,H]

        # ---- reward prediction loss ----
        r_hat = self.model.reward(z.reshape(B*H, -1), act.reshape(B*H, -1)).view(B, H)  # [B,H]
        rew_loss_t = F.mse_loss(r_hat, rew, reduction="none")  # [B,H]

        # ---- Q prediction ----
        q1, q2 = self.model.q(z.reshape(B*H, -1), act.reshape(B*H, -1))
        q1 = q1.view(B, H)
        q2 = q2.view(B, H)

        # ---- TD target using target networks ----
        with torch.no_grad():
            # next action from target policy prior
            a_next = self.target_model.pi(z_next_targ.reshape(B*H, -1)).view(B, H, -1)
            # target double Q
            tq1, tq2 = self.target_model.q(
                z_next_targ.reshape(B*H, -1),
                a_next.reshape(B*H, -1)
            )
            tq = torch.min(tq1, tq2).view(B, H)
            y = rew + (self.GAMMA * not_done) * tq  # [B,H]

        q_loss_t = (F.mse_loss(q1, y, reduction="none") + F.mse_loss(q2, y, reduction="none"))  # [B,H]

        # ---- aggregate per-sample loss (PER weights apply per sample) ----
        per_sample = (t_w * (rep_loss_t + rew_loss_t + q_loss_t)).mean(dim=1)  # [B]
        loss_model = (isw * per_sample).mean()

        # ---- optimize world model + Q (and optionally pi too, but we separate below) ----
        self.optimizer.zero_grad(set_to_none=True)
        loss_model.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.GRAD_CLIP)
        self.optimizer.step()

        # ---- policy prior update (maximize Q at encoded states) ----
        # Keep it simple and stable: maximize Q(z, pi(z)) on sampled latents.
        z_detached = z.detach()  # don't push gradients into encoder from policy loss
        a_pi = self.model.pi(z_detached.reshape(B*H, -1)).view(B, H, -1)
        q_pi = self.model.q.min(z_detached.reshape(B*H, -1), a_pi.reshape(B*H, -1)).view(B, H)
        # maximize weighted q => minimize -q
        pi_loss = - (t_w * q_pi).mean()

        self.pi_optimizer.zero_grad(set_to_none=True)
        pi_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.pi.parameters(), self.GRAD_CLIP)
        self.pi_optimizer.step()

        # ---- update target nets ----
        self._soft_update(self.TAU)

        # ---- PER priority update: use mean abs TD error over horizon (or first step) ----
        if getattr(self, "PRIORITIZED", False):
            with torch.no_grad():
                td_err = (torch.min(q1, q2) - y).abs()            # [B,H]
                pr = (t_w * td_err).mean(dim=1).detach().cpu().numpy()  # [B]
            self.replay_buffer.update_priorities(idxes, pr)

    # ---- optional logging ----
    # if statistics is not None:
    #     statistics["loss_model"].append(float(loss_model.item()))
    #     statistics["loss_pi"].append(float(pi_loss.item()))

        

    def save_dict(
            self,
            save_path: str = ""
            ):
        """
        Save the models state dict to specified path.
        ----------
        Parameter:
            save_path: The path where the model's state dictionary will be saved
        """
        saving_dir = os.path.join(save_path, self.MODEL_IDENTIFIER + ".pth")
        torch.save(self.model.state_dict(), saving_dir)
    
    
    def load_dict(
            self,
            load_path: str = ""
            ):
        """
        Load the models state dict from specified path.
        ----------
        Parameter:
            load_path: The path from which the model's state dictionary will be loaded
        """
        checkpoint = torch.load(load_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)


    def save_experiment_config(
            self, 
            base_dir: str = ""
            ):
        """
        Creates a unique folder name and saves experiment configs to it.
        ----------
        Parameter:
            base_dir: The base directory for the experiment folder
        """
        # Create a unique folder name using timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder_name = f"{timestamp}_{self.MODEL_IDENTIFIER}"
        experiment_path = os.path.join(base_dir, folder_name)
        
        os.makedirs(experiment_path, exist_ok=True)
        
        # Save configs to experiment folder
        with open(f"{experiment_path}/config.yaml", 'w') as f:
            yaml.dump(self.configs, f)
            
        return experiment_path
    

    def print_config(self):
        """
        Prints the config to the terminal.
        """
        # indent=4 makes it look like a structured config file
        pretty_conf = json.dumps(self.configs, indent=4)
        print(f"Loading Agent: {self.MODEL_IDENTIFIER}")
        print(pretty_conf)
        
    
