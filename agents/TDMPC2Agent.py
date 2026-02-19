from dataclasses import dataclass, field
import importlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import time
import yaml
from pathlib import Path
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
            verbose = False,
            config_overrides: dict | None = None
            ):

        # ------ load configs from "tdmpc_config.yaml" ------
        repo_root = Path(__file__).resolve().parent.parent
        config_path = repo_root / "configs" / "tdmpc_config.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}
        if config_overrides:
            config.update(config_overrides)
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

        # Action scaling helpers (assume continuous Box actions when available)
        self._action_shape = getattr(action_space, "shape", None)
        self._action_low = None
        self._action_high = None
        self._action_low_t = None
        self._action_high_t = None
        if hasattr(action_space, "low") and hasattr(action_space, "high"):
            self._action_low = np.asarray(action_space.low, dtype=np.float32).reshape(-1)
            self._action_high = np.asarray(action_space.high, dtype=np.float32).reshape(-1)
            if self._action_low.size == self.act_dim and self._action_high.size == self.act_dim:
                self._action_low_t = torch.as_tensor(self._action_low, device=self.device)
                self._action_high_t = torch.as_tensor(self._action_high, device=self.device)
            else:
                self._action_low = None
                self._action_high = None

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
        self.TRAINING_ROUNDS = getattr(self, "TRAINING_ROUNDS", 1)

        # ------ model sizes ------
        self.Z_DIM = getattr(self, "Z_DIM", 256)
        self.HIDDEN_DIM = getattr(self, "HIDDEN_DIM", 512)
        self.Q_ENSEMBLE_SIZE = max(1, int(getattr(self, "Q_ENSEMBLE_SIZE", 8)))

        # ------ optimizer params ------
        self.LR = getattr(self, "LR", 3e-4)
        self.ADAM_BETA_1 = getattr(self, "ADAM_BETA_1", 0.9)
        self.ADAM_BETA_2 = getattr(self, "ADAM_BETA_2", 0.999)
        self.ADAM_EPS = getattr(self, "ADAM_EPS", 1e-8)
        self.WEIGHT_DECAY = float(getattr(self, "WEIGHT_DECAY", 0.0))
        self.OPTIMIZER = str(getattr(self, "OPTIMIZER", "adam")).lower()

        # Optional separate policy optimizer params
        self.PI_LR = float(getattr(self, "PI_LR", self.LR))
        self.PI_WEIGHT_DECAY = float(getattr(self, "PI_WEIGHT_DECAY", self.WEIGHT_DECAY))

        # Muon (KellerJordan) hyperparameters
        self.MUON_LR = float(getattr(self, "MUON_LR", self.LR))
        self.MUON_PI_LR = float(getattr(self, "MUON_PI_LR", self.PI_LR))
        self.MUON_MOMENTUM = float(getattr(self, "MUON_MOMENTUM", 0.95))
        self.MUON_WEIGHT_DECAY = float(getattr(self, "MUON_WEIGHT_DECAY", 0.0))
        self.MUON_PI_WEIGHT_DECAY = float(
            getattr(self, "MUON_PI_WEIGHT_DECAY", self.MUON_WEIGHT_DECAY)
        )

        # Auxiliary Adam hyperparameters for non-matrix params in Muon optimizer
        self.MUON_ADAM_LR = float(getattr(self, "MUON_ADAM_LR", self.LR))
        self.MUON_ADAM_BETA_1 = float(getattr(self, "MUON_ADAM_BETA_1", self.ADAM_BETA_1))
        self.MUON_ADAM_BETA_2 = float(getattr(self, "MUON_ADAM_BETA_2", self.ADAM_BETA_2))
        self.MUON_ADAM_EPS = float(getattr(self, "MUON_ADAM_EPS", self.ADAM_EPS))
        self.MUON_ADAM_WEIGHT_DECAY = float(
            getattr(self, "MUON_ADAM_WEIGHT_DECAY", self.WEIGHT_DECAY)
        )
        self.MUON_PI_ADAM_LR = float(getattr(self, "MUON_PI_ADAM_LR", self.PI_LR))
        self.MUON_PI_ADAM_WEIGHT_DECAY = float(
            getattr(self, "MUON_PI_ADAM_WEIGHT_DECAY", self.PI_WEIGHT_DECAY)
        )

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

        # Optional epistemic bonus for MPC trajectory scoring
        self.USE_EPISTEMIC_EXPLORATION = bool(
            getattr(self, "USE_EPISTEMIC_EXPLORATION", False)
        )
        self.EPISTEMIC_BONUS_BETA = float(getattr(self, "EPISTEMIC_BONUS_BETA", 0.0))
        self.EPISTEMIC_BONUS_BETA_END = float(
            getattr(self, "EPISTEMIC_BONUS_BETA_END", self.EPISTEMIC_BONUS_BETA)
        )
        self.EPISTEMIC_BONUS_DECAY_STEPS = max(
            1, int(getattr(self, "EPISTEMIC_BONUS_DECAY_STEPS", 1))
        )
        self.EPISTEMIC_METHOD = str(getattr(self, "EPISTEMIC_METHOD", "ensemble")).lower()
        self.EPISTEMIC_NORMALIZE = bool(getattr(self, "EPISTEMIC_NORMALIZE", True))

        # Warm-start mean action sequence (in [-1,1])
        self._a_mean = torch.zeros(self.horizon, self.act_dim, device=self.device)

        self.MODEL_IDENTIFIER = getattr(self, "MODEL_IDENTIFIER", "TD-MPC2-Agent")
        self.model = TDMPC2(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            z_dim=self.Z_DIM,
            hidden=self.HIDDEN_DIM,
            q_ensemble_size=self.Q_ENSEMBLE_SIZE,
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

        self.replay_buffer = TDMPC2ReplayBuffer(
            capacity_steps=self.CAPACITY,
            prioritized=self.PRIORITIZED,
            alpha=self.ALPHA,
            beta=self.BETA,
            eps=self.EPSILON,
            seed=self.SEED
        )

        # ------ optimizers ------
        self._reset_optimizers()

    def _soft_update(self, tau: float) -> None:
        with torch.no_grad():
            for p_targ, p in zip(self.target_model.parameters(), self.model.parameters()):
                p_targ.data.mul_(1.0 - tau).add_(tau * p.data)

    def _unnormalize_action(self, a_tanh: torch.Tensor) -> torch.Tensor:
        if self._action_low_t is None or self._action_high_t is None:
            return a_tanh
        return self._action_low_t + (a_tanh + 1.0) * 0.5 * (self._action_high_t - self._action_low_t)

    def _normalize_action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if self._action_low is None or self._action_high is None:
            return action
        norm = 2.0 * (action - self._action_low) / (self._action_high - self._action_low + 1e-8) - 1.0
        return np.clip(norm, -1.0, 1.0)

    def _resolve_muon_optimizer_cls(self):
        candidates = (
            ("muon", "SingleDeviceMuonWithAuxAdam"),
            ("muon_optimizer", "SingleDeviceMuonWithAuxAdam"),
            ("muon_optimizer.muon", "SingleDeviceMuonWithAuxAdam"),
        )
        for module_name, class_name in candidates:
            try:
                module = importlib.import_module(module_name)
            except Exception:
                continue
            cls = getattr(module, class_name, None)
            if cls is not None:
                return cls
        return None

    def _build_adam_optimizer(self, params, lr: float, weight_decay: float):
        return optim.Adam(
            params,
            lr=float(lr),
            betas=(self.ADAM_BETA_1, self.ADAM_BETA_2),
            eps=self.ADAM_EPS,
            weight_decay=float(weight_decay),
        )

    def _build_muon_optimizer(
        self,
        params,
        muon_lr: float,
        muon_weight_decay: float,
        adam_lr: float,
        adam_weight_decay: float,
    ):
        muon_cls = self._resolve_muon_optimizer_cls()
        if muon_cls is None:
            raise RuntimeError(
                "OPTIMIZER='muon' requested, but Muon is not installed. "
                "Install KellerJordan Muon, e.g. "
                "`pip install git+https://github.com/KellerJordan/Muon`."
            )

        params = [p for p in params if p.requires_grad]
        matrix_params = [p for p in params if p.ndim >= 2]
        aux_params = [p for p in params if p.ndim < 2]

        # Muon requires at least one matrix-shaped parameter group.
        if len(matrix_params) == 0:
            return self._build_adam_optimizer(params, lr=adam_lr, weight_decay=adam_weight_decay)

        param_groups = [
            dict(
                params=matrix_params,
                lr=float(muon_lr),
                momentum=float(self.MUON_MOMENTUM),
                weight_decay=float(muon_weight_decay),
                use_muon=True,
            )
        ]
        if len(aux_params) > 0:
            param_groups.append(
                dict(
                    params=aux_params,
                    lr=float(adam_lr),
                    betas=(self.MUON_ADAM_BETA_1, self.MUON_ADAM_BETA_2),
                    eps=float(self.MUON_ADAM_EPS),
                    weight_decay=float(adam_weight_decay),
                    use_muon=False,
                )
            )

        return muon_cls(param_groups)

    def _reset_optimizers(self) -> None:
        opt = self.OPTIMIZER.lower()
        if opt == "adam":
            self.optimizer = self._build_adam_optimizer(
                self.model.parameters(),
                lr=self.LR,
                weight_decay=self.WEIGHT_DECAY,
            )
            self.pi_optimizer = self._build_adam_optimizer(
                self.model.pi.parameters(),
                lr=self.PI_LR,
                weight_decay=self.PI_WEIGHT_DECAY,
            )
            return

        if opt == "muon":
            self.optimizer = self._build_muon_optimizer(
                self.model.parameters(),
                muon_lr=self.MUON_LR,
                muon_weight_decay=self.MUON_WEIGHT_DECAY,
                adam_lr=self.MUON_ADAM_LR,
                adam_weight_decay=self.MUON_ADAM_WEIGHT_DECAY,
            )
            self.pi_optimizer = self._build_muon_optimizer(
                self.model.pi.parameters(),
                muon_lr=self.MUON_PI_LR,
                muon_weight_decay=self.MUON_PI_WEIGHT_DECAY,
                adam_lr=self.MUON_PI_ADAM_LR,
                adam_weight_decay=self.MUON_PI_ADAM_WEIGHT_DECAY,
            )
            return

        raise ValueError(
            f"Unsupported OPTIMIZER='{self.OPTIMIZER}'. Expected one of: 'adam', 'muon'."
        )

    def _current_epistemic_beta(self) -> float:
        if not self.USE_EPISTEMIC_EXPLORATION:
            return 0.0
        frac = min(1.0, float(self.t) / float(self.EPISTEMIC_BONUS_DECAY_STEPS))
        return self.EPISTEMIC_BONUS_BETA + frac * (
            self.EPISTEMIC_BONUS_BETA_END - self.EPISTEMIC_BONUS_BETA
        )

    def _standardize(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean()
        std = x.std(unbiased=False)
        return (x - mean) / (std + 1e-6)

    def _epistemic_uncertainty(
        self,
        obs: torch.Tensor,
        zs: torch.Tensor,
        rs: torch.Tensor,
        a_seq: torch.Tensor,
        discounts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimates trajectory-level epistemic uncertainty for MPC scoring.
        """
        N, H, _ = a_seq.shape
        method = self.EPISTEMIC_METHOD
        use_q = method in {"q", "q_disagreement", "combined", "both"}
        use_ens = method in {"ensemble"}
        use_model = method in {"model", "model_disagreement", "combined", "both"}
        if not use_q and not use_ens and not use_model:
            use_q = True

        terms = []

        if use_q:
            q1, q2 = self.model.q(
                zs[:, :-1, :].reshape(N * H, -1),
                a_seq.reshape(N * H, -1),
            )
            q_disc = (q1 - q2).abs().view(N, H)
            q_unc = (q_disc * discounts).sum(dim=1)
            terms.append(q_unc)

        if use_ens:
            # Ensemble epistemic as variance of model-wise trajectory means:
            # 1) per-head Q predictions along the rollout
            # 2) discounted mean across horizon for each head
            # 3) variance across heads -> one scalar uncertainty per trajectory
            q_all = self.model.q.all(
                zs[:, :-1, :].reshape(N * H, -1),
                a_seq.reshape(N * H, -1),
            ).view(int(self.model.q.ensemble_size), N, H)  # [E,N,H]
            disc = discounts.view(1, 1, H)
            disc_w = disc / (disc.sum(dim=-1, keepdim=True) + 1e-8)
            q_head_mean = (q_all * disc_w).sum(dim=-1)  # [E,N]
            ens_unc = q_head_mean.var(dim=0, unbiased=False)  # [N]
            terms.append(ens_unc)

        if use_model:
            z0_targ = self.target_model.encoder(obs).repeat(N, 1)
            zs_targ, rs_targ = self.target_model.rollout(z0_targ, a_seq)

            z_disc = (zs[:, 1:, :] - zs_targ[:, 1:, :]).pow(2).mean(dim=-1)
            r_disc = (rs - rs_targ).pow(2)

            z_unc = (z_disc * discounts).sum(dim=1)
            r_unc = (r_disc * discounts).sum(dim=1)
            model_unc = torch.sqrt(z_unc + r_unc + 1e-8)
            terms.append(model_unc)

        if len(terms) == 0:
            return torch.zeros(N, device=self.device)

        if self.EPISTEMIC_NORMALIZE:
            terms = [self._standardize(t) for t in terms]

        return torch.stack(terms, dim=0).mean(dim=0)

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

        # Helper: map [-1,1] -> env bounds (if present)
        def unnormalize(a_tanh):
            return self._unnormalize_action(a_tanh)

        # Helper: discount vector [H]
        discounts = (self.GAMMA ** torch.arange(H, device=self.device, dtype=torch.float32)).view(1, H)

        # --- 2) MPPI: iterative refinement of mean action sequence ---
        mean = self._a_mean.clone()  # [H, act_dim]
        sigma = self.MPC_SIGMA
        beta = self._current_epistemic_beta()
        last_uncertainty = None

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
            planning_score = returns

            if beta > 0.0:
                uncertainty = self._epistemic_uncertainty(
                    obs=obs,
                    zs=zs,
                    rs=rs,
                    a_seq=a_seq,
                    discounts=discounts,
                )
                planning_score = planning_score + beta * uncertainty
                last_uncertainty = uncertainty

            # MPPI weights (softmax over returns)
            # stabilize by subtracting max
            scaled = (planning_score - planning_score.max()) / max(self.MPC_TEMP, 1e-6)
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
        if self._action_shape is not None and self._action_low is not None:
            a_env = a_env.reshape(self._action_shape)
        self.t += 1

        if statistics is not None:
            if "epistemic_beta" in statistics:
                statistics["epistemic_beta"].append(float(beta))
            if last_uncertainty is not None and "epistemic_unc" in statistics:
                statistics["epistemic_unc"].append(float(last_uncertainty.mean().item()))
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
        norm_action = self._normalize_action(action)
        self.replay_buffer.add(state, norm_action, reward, next_state, terminated)
        

    
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
        obs, act, rew, done, isw, idxes = self.replay_buffer.sample_sequences(
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

        # ---- Q prediction (ensemble) ----
        n_q = int(self.model.q.ensemble_size)
        q_all = self.model.q.all(
            z.reshape(B * H, -1),
            act.reshape(B * H, -1),
        ).view(n_q, B, H)

        # ---- TD target using target networks ----
        with torch.no_grad():
            # next action from target policy prior
            a_next = self.target_model.pi(z_next_targ.reshape(B*H, -1)).view(B, H, -1)
            # target min over ensemble Q
            tq = self.target_model.q.min(
                z_next_targ.reshape(B*H, -1),
                a_next.reshape(B*H, -1)
            ).view(B, H)
            y = rew + (self.GAMMA * not_done) * tq  # [B,H]

        q_loss_t = F.mse_loss(
            q_all,
            y.unsqueeze(0).expand_as(q_all),
            reduction="none",
        ).mean(dim=0)  # [B,H]

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
                q_min = q_all.min(dim=0).values
                td_err = (q_min - y).abs()            # [B,H]
                pr = (t_w * td_err).mean(dim=1).detach().cpu().numpy()  # [B]
            self.replay_buffer.update_priorities(idxes, pr)

        # ---- logging ----
        if statistics is not None and "tr_loss" in statistics:
            statistics["tr_loss"].append(float(loss_model.detach().cpu().item()))

        

    def save_dict(
            self,
            save_path: str = "",
            identifier_extension: str = "",
            ):
        """
        Save the models state dict to specified path.
        ----------
        Parameter:
            save_path: The path where the model's state dictionary will be saved
        """
        saving_dir = os.path.join(save_path, self.MODEL_IDENTIFIER + identifier_extension + ".pth")
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
        
    
