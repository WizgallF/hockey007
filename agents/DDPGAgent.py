from __future__ import annotations

from datetime import datetime
import os
from typing import Any

import numpy as np
import torch
from gymnasium import spaces
import yaml

from agents.AgentBaseclass import Agent
from agents.networks.DDPGNetwork import DDPGNetwork
from agents.utils.DDPGUtils import Memory


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(1)


class UnsupportedSpace(Exception):
    """Exception raised when the Sensor or Action space are not compatible."""

    def __init__(self, message: str = "Unsupported Space"):
        super().__init__(message)


class QFunction(DDPGNetwork):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_sizes: list[int] | tuple[int, ...] = (256, 256),
        learning_rate: float = 0.0002,
        activation_fun: torch.nn.Module | None = None,
    ):
        if activation_fun is None:
            activation_fun = torch.nn.ReLU()
        super().__init__(
            input_size=observation_dim + action_dim,
            hidden_sizes=list(hidden_sizes),
            output_size=1,
            activation_fun=activation_fun,
            output_activation=None,
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, eps=1e-6)
        self.loss = torch.nn.SmoothL1Loss()

    def fit(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:
        self.train()
        self.optimizer.zero_grad()
        pred = self.Q_value(observations, actions)
        loss = self.loss(pred, targets)
        loss.backward()
        self.optimizer.step()
        return float(loss.detach().cpu().item())

    def Q_value(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.forward(torch.hstack([observations, actions]))


class OUNoise:
    def __init__(self, shape: int | tuple[int, ...], theta: float = 0.15, dt: float = 1e-2):
        self._shape = shape
        self._theta = theta
        self._dt = dt
        self.noise_prev = np.zeros(self._shape, dtype=np.float32)
        self.reset()

    def __call__(self) -> np.ndarray:
        noise = (
            self.noise_prev
            + self._theta * (-self.noise_prev) * self._dt
            + np.sqrt(self._dt) * np.random.normal(size=self._shape)
        )
        self.noise_prev = noise
        return noise

    def reset(self) -> None:
        self.noise_prev = np.zeros(self._shape, dtype=np.float32)


class DDPGAgent(Agent):
    """
    DDPG agent compatible with the local Agent wrapper API.
    """

    def __init__(self, 
                 observation_space: spaces.Box, 
                 action_space: spaces.Box, 
                 config_path: str = "configs/ddpg_config.yaml",  
                 **userconfig: Any):
        
        if not isinstance(observation_space, spaces.Box):
            raise UnsupportedSpace(
                f"Observation space {observation_space} incompatible with {self}. (Require: Box)"
            )
        if not isinstance(action_space, spaces.Box):
            raise UnsupportedSpace(
                f"Action space {action_space} incompatible with {self}. (Require: Box)"
            )

        self._observation_space = observation_space
        self._action_space = action_space
        self._obs_dim = int(self._observation_space.shape[0])
        self._action_n = int(self._action_space.shape[0])
        self.device = device

        if userconfig:
            extra_keys = {key for key in userconfig.keys() if key != "verbose"}
            if extra_keys:
                raise ValueError(
                    "DDPGAgent only uses configs/ddpg_config.yaml; remove overrides "
                    f"(unsupported keys: {', '.join(sorted(extra_keys))})."
                )
        self.verbose = bool(userconfig.get("verbose", False)) if userconfig else False

        self._config = self._load_config(config_path)

        self.MODEL_IDENTIFIER = self._config["MODEL_IDENTIFIER"]
        self.NUM_EPISODES = self._config["NUM_EPISODES"]
        self.START_TRAINING = self._config["START_TRAINING"]
        self._eps = self._config["EPS"]

        self._action_low = np.asarray(self._action_space.low, dtype=np.float32)
        self._action_high = np.asarray(self._action_space.high, dtype=np.float32)
        self._action_low_t = torch.as_tensor(self._action_low, device=self.device)
        self._action_high_t = torch.as_tensor(self._action_high, device=self.device)

        self.action_noise = OUNoise(
            (self._action_n,),
            theta=self._config["ACTION_NOISE_THETA"],
            dt=self._config["ACTION_NOISE_DT"],
        )
        self.buffer = Memory(max_size=self._config["BUFFER_SIZE"])

        self.Q = QFunction(
            observation_dim=self._obs_dim,
            action_dim=self._action_n,
            hidden_sizes=self._config["HIDDEN_SIZES_CRITIC"],
            learning_rate=self._config["LEARNING_RATE_CRITIC"],
        ).to(self.device)
        self.Q_target = QFunction(
            observation_dim=self._obs_dim,
            action_dim=self._action_n,
            hidden_sizes=self._config["HIDDEN_SIZES_CRITIC"],
            learning_rate=0.0,
        ).to(self.device)
        if self._config["TWIN_DELAYED"]:
            self.Q2 = QFunction(
                observation_dim=self._obs_dim,
                action_dim=self._action_n,
                hidden_sizes=self._config["HIDDEN_SIZES_CRITIC"],
                learning_rate=self._config["LEARNING_RATE_CRITIC"],
            ).to(self.device)
            self.Q2_target = QFunction(
                observation_dim=self._obs_dim,
                action_dim=self._action_n,
                hidden_sizes=self._config["HIDDEN_SIZES_CRITIC"],
                learning_rate=0.0,
            ).to(self.device)

        self.policy = DDPGNetwork(
            input_size=self._obs_dim,
            hidden_sizes=self._config["HIDDEN_SIZES_ACTOR"],
            output_size=self._action_n,
            activation_fun=torch.nn.ReLU(),
            output_activation=torch.nn.Tanh(),
        ).to(self.device)
        self.policy_target = DDPGNetwork(
            input_size=self._obs_dim,
            hidden_sizes=self._config["HIDDEN_SIZES_ACTOR"],
            output_size=self._action_n,
            activation_fun=torch.nn.ReLU(),
            output_activation=torch.nn.Tanh(),
        ).to(self.device)

        self._copy_nets()

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self._config["LEARNING_RATE_ACTOR"],
            eps=1e-6,
        )
        self.train_iter = 0

    def _copy_nets(self) -> None:
        self.Q_target.load_state_dict(self.Q.state_dict())
        self.policy_target.load_state_dict(self.policy.state_dict())

    def _soft_update(self, tau: float) -> None:
        with torch.no_grad():
            for target_param, param in zip(self.Q_target.parameters(), self.Q.parameters()):
                target_param.data.mul_(1 - tau).add_(tau * param.data)
            for target_param, param in zip(self.policy_target.parameters(), self.policy.parameters()):
                target_param.data.mul_(1 - tau).add_(tau * param.data)

    def _scale_action_np(self, action: np.ndarray) -> np.ndarray:
        return self._action_low + (action + 1.0) * 0.5 * (self._action_high - self._action_low)

    def _scale_action_torch(self, action: torch.Tensor) -> torch.Tensor:
        return self._action_low_t + (action + 1.0) * 0.5 * (self._action_high_t - self._action_low_t)

    def reset(self) -> None:
        self.action_noise.reset()

    def act(
        self,
        env,
        state: np.ndarray | None = None,
        i_episode: int | None = None,
        statistics: dict[str, list] | None = None,
        greedy: bool = False,
        eps: float | None = None,
    ) -> np.ndarray:
        """
        Selects a continuous action. Accepts both (state) and (env, state, ...) call styles.
        """
        if state is None:
            state = np.asarray(env, dtype=np.float32)
        else:
            state = np.asarray(state, dtype=np.float32)

        if eps is None:
            eps = self._eps

        if state.ndim == 1:
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                raw_action = self.policy(state_t).squeeze(0).cpu().numpy()

            if not greedy:
                raw_action = raw_action + eps * self.action_noise()
            raw_action = np.clip(raw_action, -1.0, 1.0)
            return self._scale_action_np(raw_action)

        # Vectorized path for parallel environments: state shape (N, obs_dim)
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            raw_action = self.policy(state_t).cpu().numpy()

        if not greedy:
            noise = np.stack([self.action_noise() for _ in range(raw_action.shape[0])], axis=0)
            raw_action = raw_action + eps * noise
        raw_action = np.clip(raw_action, -1.0, 1.0)
        return self._scale_action_np(raw_action)

    def observe(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
    ) -> None:
        state = np.asarray(state, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        next_state = np.asarray(next_state, dtype=np.float32)

        if state.ndim == 1:
            transition = (
                state,
                action,
                float(reward),
                next_state,
                float(terminated),
            )
            self.buffer.add_transition(transition)
            return

        # Vectorized path for parallel environments
        for idx in range(state.shape[0]):
            transition = (
                state[idx],
                action[idx],
                float(reward[idx]),
                next_state[idx],
                float(terminated[idx]),
            )
            self.buffer.add_transition(transition)

    def update(self, statistics: dict[str, list] | None = None) -> list[tuple[float, float]] | None:
        last_critic_loss = None
        last_q_values = None
        if self.buffer.size < self._config["BATCH_SIZE"]:
            return None

        self.train_iter += 1
        if self._config["USE_TARGET_NET"] and self._config["TAU"] is None:
            if self.train_iter % self._config["UPDATE_TARGET_EVERY"] == 0:
                self._copy_nets()

        losses: list[tuple[float, float]] = []
        iter_fit = int(self._config["TRAIN_ITERATIONS"])
        self.grad_step = getattr(self, "grad_step", 0)

        for _ in range(iter_fit):
            self.grad_step += 1
            data = self.buffer.sample(batch=self._config["BATCH_SIZE"])

            s = torch.from_numpy(np.stack(data[:, 0])).float().to(self.device)
            a = torch.from_numpy(np.stack(data[:, 1])).float().to(self.device)
            rew = torch.from_numpy(np.stack(data[:, 2])[:, None]).float().to(self.device)
            s_prime = torch.from_numpy(np.stack(data[:, 3])).float().to(self.device)
            done = torch.from_numpy(np.stack(data[:, 4])[:, None]).float().to(self.device)

            with torch.no_grad():
                assert self._config["USE_TARGET_NET"], "TD3 expects target nets; re-add else if needed"
                if self._config["USE_TARGET_NET"]:
                    next_actions = self._scale_action_torch(self.policy_target(s_prime))
                    if self._config["POLICY_NOISE"] > 0:
                        noise = (
                            torch.randn_like(next_actions) * self._config["POLICY_NOISE"]
                        ).clamp(-self._config["NOISE_CLIP"], self._config["NOISE_CLIP"])
                        next_actions = (next_actions + noise)
                    next_actions = torch.clamp(next_actions, self._action_low_t, self._action_high_t)
                    if self._config["TWIN_DELAYED"]:
                        q1_prime = self.Q_target.Q_value(s_prime, next_actions)
                        q2_prime = self.Q2_target.Q_value(s_prime, next_actions)
                        q_prime = torch.min(q1_prime, q2_prime)
                    else:
                        q_prime = self.Q_target.Q_value(s_prime, next_actions)
                #else:
                #    if self._config['TWIN_DELAYED']:
                #        q1_prime = self.Q.Q_value(s_prime, next_actions)
                #        q2_prime = self.Q2.Q_value(s_prime, next_actions)
                #        q_prime = torch.min(q1_prime, q2_prime)
                #    else:
                #        next_actions = self._scale_action_torch(self.policy(s_prime))
                #        q_prime = self.Q.Q_value(s_prime, next_actions)

                gamma = float(self._config["DISCOUNT"])
                td_target = rew + gamma * (1.0 - done) * q_prime

            critic_loss = self.Q.fit(s, a, td_target)
            last_critic_loss = critic_loss

            if self._config["TWIN_DELAYED"]:
                critic2_loss = self.Q2.fit(s, a, td_target)

            policy_delay = int(self._config.get("POLICY_DELAY", 2))
            if self.grad_step % policy_delay == 0:
                self.optimizer.zero_grad()
                current_actions = self._scale_action_torch(self.policy(s))
                q_values = self.Q.Q_value(s, current_actions)
                last_q_values = q_values.detach()
                actor_loss = -torch.mean(q_values)
                actor_loss.backward()
                self.optimizer.step()

                losses.append((critic_loss, float(actor_loss.detach().cpu().item())))

                if self._config["USE_TARGET_NET"] and self._config["TAU"] is not None:
                    self._soft_update(float(self._config["TAU"]))

        if statistics is not None:
            if last_critic_loss is not None and "tr_loss" in statistics:
                statistics["tr_loss"].append(last_critic_loss)

            if last_q_values is not None:
                if "mean_q" in statistics:
                    statistics["mean_q"].append(float(last_q_values.mean().cpu().item()))
                if "min_q" in statistics:
                    statistics["min_q"].append(float(last_q_values.min().cpu().item()))
                if "max_q" in statistics:
                    statistics["max_q"].append(float(last_q_values.max().cpu().item()))

        return losses

    def save_dict(self, save_path: str = "") -> None:
        saving_dir = os.path.join(save_path, f"{self.MODEL_IDENTIFIER}.pth")
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "critic": self.Q.state_dict(),
                "policy_target": self.policy_target.state_dict(),
                "critic_target": self.Q_target.state_dict(),
                "config": self._config,
            },
            saving_dir,
        )

    def load_dict(self, load_path: str = "") -> None:
        checkpoint = torch.load(load_path, map_location=self.device)
        if isinstance(checkpoint, dict) and "policy" in checkpoint:
            self.policy.load_state_dict(checkpoint["policy"])
            self.Q.load_state_dict(checkpoint["critic"])
            if "policy_target" in checkpoint and "critic_target" in checkpoint:
                self.policy_target.load_state_dict(checkpoint["policy_target"])
                self.Q_target.load_state_dict(checkpoint["critic_target"])
            else:
                self._copy_nets()
            return

        self.policy.load_state_dict(checkpoint)
        self._copy_nets()

    def save_experiment_config(self, base_dir: str = "") -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder_name = f"{timestamp}_{self.MODEL_IDENTIFIER}"
        experiment_path = os.path.join(base_dir, folder_name)

        os.makedirs(experiment_path, exist_ok=True)
        with open(os.path.join(experiment_path, "config.yaml"), "w") as f:
            yaml.dump(self._config, f)
        return experiment_path

    def print_config(self) -> None:
        print(f"Loading Agent: {self.MODEL_IDENTIFIER}")
        print(yaml.dump(self._config, sort_keys=False))

    def _load_config(self, config_path: str) -> dict[str, Any]:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}

        required = [
            "MODEL_IDENTIFIER",
            "NUM_EPISODES",
            "START_TRAINING",
            "EPS",
            "DISCOUNT",
            "BUFFER_SIZE",
            "BATCH_SIZE",
            "TRAIN_ITERATIONS",
            "LEARNING_RATE_ACTOR",
            "LEARNING_RATE_CRITIC",
            "HIDDEN_SIZES_ACTOR",
            "HIDDEN_SIZES_CRITIC",
            "UPDATE_TARGET_EVERY",
            "USE_TARGET_NET",
            "TAU",
            "ACTION_NOISE_THETA",
            "ACTION_NOISE_DT",
            "TWIN_DELAYED",
            "POLICY_NOISE",
            "NOISE_CLIP",
            "POLICY_DELAY",
        ]
        missing = [key for key in required if key not in config]
        if missing:
            raise KeyError(f"Missing DDPG config keys: {', '.join(missing)}")
        return config
