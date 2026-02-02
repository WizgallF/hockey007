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

    def __init__(self, observation_space: spaces.Box, action_space: spaces.Box, **userconfig: Any):
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

        self._config = {
            "model_identifier": "DDPG",
            "num_episodes": 1000,
            "start_training": 0,
            "eps": 0.1,
            "discount": 0.95,
            "buffer_size": int(1e6),
            "batch_size": 128,
            "train_iterations": 1,
            "learning_rate_actor": 0.00001,
            "learning_rate_critic": 0.0001,
            "hidden_sizes_actor": [128, 128],
            "hidden_sizes_critic": [128, 128, 64],
            "update_target_every": 100,
            "use_target_net": True,
            "tau": None,
            "action_noise_theta": 0.15,
            "action_noise_dt": 1e-2,
        }
        self._apply_yaml_config("configs/ddpg_config.yaml")
        self._apply_user_config(userconfig)

        self.MODEL_IDENTIFIER = self._config["model_identifier"]
        self.NUM_EPISODES = self._config["num_episodes"]
        self.START_TRAINING = self._config["start_training"]
        self._eps = self._config["eps"]

        self._action_low = np.asarray(self._action_space.low, dtype=np.float32)
        self._action_high = np.asarray(self._action_space.high, dtype=np.float32)
        self._action_low_t = torch.as_tensor(self._action_low, device=self.device)
        self._action_high_t = torch.as_tensor(self._action_high, device=self.device)

        self.action_noise = OUNoise(
            (self._action_n,),
            theta=self._config["action_noise_theta"],
            dt=self._config["action_noise_dt"],
        )
        self.buffer = Memory(max_size=self._config["buffer_size"])

        self.Q = QFunction(
            observation_dim=self._obs_dim,
            action_dim=self._action_n,
            hidden_sizes=self._config["hidden_sizes_critic"],
            learning_rate=self._config["learning_rate_critic"],
        ).to(self.device)
        self.Q_target = QFunction(
            observation_dim=self._obs_dim,
            action_dim=self._action_n,
            hidden_sizes=self._config["hidden_sizes_critic"],
            learning_rate=0.0,
        ).to(self.device)

        self.policy = DDPGNetwork(
            input_size=self._obs_dim,
            hidden_sizes=self._config["hidden_sizes_actor"],
            output_size=self._action_n,
            activation_fun=torch.nn.ReLU(),
            output_activation=torch.nn.Tanh(),
        ).to(self.device)
        self.policy_target = DDPGNetwork(
            input_size=self._obs_dim,
            hidden_sizes=self._config["hidden_sizes_actor"],
            output_size=self._action_n,
            activation_fun=torch.nn.ReLU(),
            output_activation=torch.nn.Tanh(),
        ).to(self.device)

        self._copy_nets()

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self._config["learning_rate_actor"],
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

        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            raw_action = self.policy(state_t).squeeze(0).cpu().numpy()

        if not greedy:
            raw_action = raw_action + eps * self.action_noise()
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
        transition = (
            np.asarray(state, dtype=np.float32),
            np.asarray(action, dtype=np.float32),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            float(terminated),
        )
        self.buffer.add_transition(transition)

    def update(self, statistics: dict[str, list] | None = None) -> list[tuple[float, float]] | None:
        if self.buffer.size < self._config["batch_size"]:
            return None

        self.train_iter += 1
        if self._config["use_target_net"] and self._config["tau"] is None:
            if self.train_iter % self._config["update_target_every"] == 0:
                self._copy_nets()

        losses: list[tuple[float, float]] = []
        iter_fit = int(self._config["train_iterations"])
        for _ in range(iter_fit):
            data = self.buffer.sample(batch=self._config["batch_size"])

            s = torch.from_numpy(np.stack(data[:, 0])).float().to(self.device)
            a = torch.from_numpy(np.stack(data[:, 1])).float().to(self.device)
            rew = torch.from_numpy(np.stack(data[:, 2])[:, None]).float().to(self.device)
            s_prime = torch.from_numpy(np.stack(data[:, 3])).float().to(self.device)
            done = torch.from_numpy(np.stack(data[:, 4])[:, None]).float().to(self.device)

            with torch.no_grad():
                if self._config["use_target_net"]:
                    next_actions = self._scale_action_torch(self.policy_target(s_prime))
                    q_prime = self.Q_target.Q_value(s_prime, next_actions)
                else:
                    next_actions = self._scale_action_torch(self.policy(s_prime))
                    q_prime = self.Q.Q_value(s_prime, next_actions)

                gamma = float(self._config["discount"])
                td_target = rew + gamma * (1.0 - done) * q_prime

            critic_loss = self.Q.fit(s, a, td_target)

            self.optimizer.zero_grad()
            current_actions = self._scale_action_torch(self.policy(s))
            q_values = self.Q.Q_value(s, current_actions)
            actor_loss = -torch.mean(q_values)
            actor_loss.backward()
            self.optimizer.step()

            losses.append((critic_loss, float(actor_loss.detach().cpu().item())))

            if self._config["use_target_net"] and self._config["tau"] is not None:
                self._soft_update(float(self._config["tau"]))

        if statistics is not None and losses:
            if "tr_loss" in statistics:
                statistics["tr_loss"].append(losses[-1][0])
            if "mean_q" in statistics:
                statistics["mean_q"].append(float(q_values.mean().detach().cpu().item()))
            if "min_q" in statistics:
                statistics["min_q"].append(float(q_values.min().detach().cpu().item()))
            if "max_q" in statistics:
                statistics["max_q"].append(float(q_values.max().detach().cpu().item()))

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

    def _apply_yaml_config(self, config_path: str) -> None:
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f) or {}
        except FileNotFoundError:
            return

        key_map = {
            "MODEL_IDENTIFIER": "model_identifier",
            "NUM_EPISODES": "num_episodes",
            "START_TRAINING": "start_training",
            "EPS": "eps",
            "DISCOUNT": "discount",
            "BUFFER_SIZE": "buffer_size",
            "BATCH_SIZE": "batch_size",
            "TRAIN_ITERATIONS": "train_iterations",
            "LEARNING_RATE_ACTOR": "learning_rate_actor",
            "LEARNING_RATE_CRITIC": "learning_rate_critic",
            "HIDDEN_SIZES_ACTOR": "hidden_sizes_actor",
            "HIDDEN_SIZES_CRITIC": "hidden_sizes_critic",
            "UPDATE_TARGET_EVERY": "update_target_every",
            "USE_TARGET_NET": "use_target_net",
            "TAU": "tau",
            "ACTION_NOISE_THETA": "action_noise_theta",
            "ACTION_NOISE_DT": "action_noise_dt",
        }

        for key, value in config.items():
            if key in key_map:
                self._config[key_map[key]] = value
            elif key in self._config:
                self._config[key] = value

    def _apply_user_config(self, userconfig: dict[str, Any]) -> None:
        if not userconfig:
            return

        key_map = {
            "MODEL_IDENTIFIER": "model_identifier",
            "NUM_EPISODES": "num_episodes",
            "START_TRAINING": "start_training",
            "EPS": "eps",
            "DISCOUNT": "discount",
            "BUFFER_SIZE": "buffer_size",
            "BATCH_SIZE": "batch_size",
            "TRAIN_ITERATIONS": "train_iterations",
            "LEARNING_RATE_ACTOR": "learning_rate_actor",
            "LEARNING_RATE_CRITIC": "learning_rate_critic",
            "HIDDEN_SIZES_ACTOR": "hidden_sizes_actor",
            "HIDDEN_SIZES_CRITIC": "hidden_sizes_critic",
            "UPDATE_TARGET_EVERY": "update_target_every",
            "USE_TARGET_NET": "use_target_net",
            "TAU": "tau",
            "ACTION_NOISE_THETA": "action_noise_theta",
            "ACTION_NOISE_DT": "action_noise_dt",
        }

        for key, value in userconfig.items():
            if key in key_map:
                self._config[key_map[key]] = value
            elif key in self._config:
                self._config[key] = value
