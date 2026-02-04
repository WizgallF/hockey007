#!/usr/bin/env python3
"""
TDMPC2_tune.py

WandB-based grid tuning for TD-MPC2. You can specify a grid with min/max values
and the best configuration will be saved as a YAML in the experiments folder.
The YAML is updated every time a new best score is found, so you keep the latest
best parameters even if the tuning run is interrupted.
"""

from __future__ import annotations

import argparse
import inspect
import itertools
import json
import math
import os
import random
import sys
from typing import Any, Dict, List

import numpy as np
import torch
import torch.optim as optim

try:
    import wandb
except Exception:  # pragma: no cover - handled at runtime
    wandb = None

try:
    import yaml
except Exception:
    yaml = None

import gymnasium as gym
from gymnasium import spaces
import hockey.hockey_env as h_env

from Train import Training
from agents.TDMPC2Agent import TDMPC2Agent


INT_PARAMS = {
    "BATCH_SIZE",
    "TRAIN_HORIZON",
    "NUM_EPISODES",
    "MPC_SAMPLES",
    "MPC_ITERS",
    "Z_DIM",
    "HIDDEN_DIM",
}


DEFAULT_GRID = {
    "LR": {"min": 1e-4, "max": 3e-4, "num": 2, "scale": "log"},
    "BATCH_SIZE": {"min": 128, "max": 512, "num": 2, "scale": "linear"},
    "TRAIN_HORIZON": {"min": 8, "max": 16, "num": 2, "scale": "linear"},
    "MPC_SAMPLES": {"min": 256, "max": 512, "num": 2, "scale": "linear"},
    "MPC_SIGMA": {"min": 0.2, "max": 0.6, "num": 2, "scale": "linear"},
}


def _load_grid_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        if path.endswith((".yaml", ".yml")):
            if yaml is None:
                raise RuntimeError("PyYAML not available; use a JSON grid config instead.")
            return yaml.safe_load(f) or {}
        return json.load(f)


def _values_from_spec(spec: Any) -> List[Any]:
    if isinstance(spec, dict):
        if "values" in spec:
            return list(spec["values"])
        if "min" in spec and "max" in spec:
            vmin = float(spec["min"])
            vmax = float(spec["max"])
            num = int(spec.get("num", 5))
            if num <= 1:
                return [vmin]
            scale = str(spec.get("scale", "linear")).lower()
            if scale == "log":
                return np.logspace(math.log10(vmin), math.log10(vmax), num=num).tolist()
            return np.linspace(vmin, vmax, num=num).tolist()
    if isinstance(spec, (list, tuple)):
        return list(spec)
    return [spec]


def _cast_param(name: str, value: Any) -> Any:
    if name in INT_PARAMS:
        return int(round(float(value)))
    return value


def build_grid(grid_spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    keys = list(grid_spec.keys())
    values = [_values_from_spec(grid_spec[k]) for k in keys]
    grid = []
    for combo in itertools.product(*values):
        cfg = {k: _cast_param(k, v) for k, v in zip(keys, combo)}
        grid.append(cfg)
    return grid


class TDMPC2NormalizeActionWrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._enabled = isinstance(env.action_space, spaces.Box)
        if self._enabled:
            self._low = np.asarray(env.action_space.low, dtype=np.float32)
            self._high = np.asarray(env.action_space.high, dtype=np.float32)
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=self._low.shape,
                dtype=np.float32,
            )

    def action(self, action):
        if not self._enabled:
            return action
        a = np.asarray(action, dtype=np.float32)
        a = np.clip(a, -1.0, 1.0)
        return self._low + (a + 1.0) * 0.5 * (self._high - self._low)


class TDMPC2NormalizeObsWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._enabled = isinstance(env.observation_space, spaces.Box)
        self._can_scale = False
        if self._enabled:
            self._low = np.asarray(env.observation_space.low, dtype=np.float32)
            self._high = np.asarray(env.observation_space.high, dtype=np.float32)
            self._can_scale = (
                np.all(np.isfinite(self._low))
                and np.all(np.isfinite(self._high))
                and np.all(self._high > self._low)
            )
            if self._can_scale:
                self.observation_space = spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=self._low.shape,
                    dtype=np.float32,
                )
            else:
                self.observation_space = spaces.Box(
                    low=self._low,
                    high=self._high,
                    shape=self._low.shape,
                    dtype=np.float32,
                )

    def observation(self, obs):
        if not self._enabled:
            return obs
        x = np.asarray(obs, dtype=np.float32)
        x = np.clip(x, self._low, self._high)
        if self._can_scale:
            x = 2.0 * (x - self._low) / (self._high - self._low + 1e-8) - 1.0
            x = np.clip(x, -1.0, 1.0)
        return x


def make_env(env_name: str):
    if env_name == "Pendulum-v1":
        env = gym.make(env_name, disable_env_checker=True)
    elif env_name == "Hockey-One-v0":
        env = gym.make("Hockey-One-v0", mode=0, weak_opponent=True, disable_env_checker=True)
    else:
        env = gym.make(env_name, disable_env_checker=True)
    env = TDMPC2NormalizeObsWrapper(env)
    env = TDMPC2NormalizeActionWrapper(env)
    return env


def set_seeds(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def score_from_stats(stats: Dict[str, List[float]], mavg_window: int) -> float:
    mv = stats.get("mv_avg_rew", [])
    if mv:
        return float(np.max(np.asarray(mv)))
    ep = stats.get("ep_rew", [])
    if len(ep) > 0:
        ep = ep[:-1]
    if not ep:
        return float("-inf")
    window = max(1, min(len(ep), mavg_window))
    return float(np.mean(ep[-window:]))


def run_trial(
    trial_idx: int,
    config: Dict[str, Any],
    args: argparse.Namespace,
):
    if wandb is None:
        raise RuntimeError("wandb is not installed. Install it or remove wandb usage.")

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=config,
        name=f"tdmpc2_tune_{trial_idx}",
        mode=args.wandb_mode,
        reinit=True,
    )

    trial_seed = None
    if "SEED" in config:
        trial_seed = int(config["SEED"]) + trial_idx
    set_seeds(trial_seed)

    env = make_env(args.env)
    try:
        cfg_dict = dict(run.config)
        agent = _build_agent(env, cfg_dict)
        if args.num_episodes is not None:
            agent.NUM_EPISODES = int(args.num_episodes)

        trainer = Training(
            agent=agent,
            env=env,
            base_dir=args.base_dir,
            save_intermediate_agents=False,
            verbose=args.verbose,
        )
        trainer.train()

        score = score_from_stats(trainer.statistics, trainer.mavg_window_size)
        run.summary["best_mavg_rew"] = score
        wandb.log({"best_mavg_rew": score})
        return score, dict(run.config), trainer.experiment_path
    finally:
        env.close()
        run.finish()


def write_best_yaml(
    base_dir: str,
    best_score: float,
    best_config: Dict[str, Any],
    best_experiment_path: str,
):
    os.makedirs(base_dir, exist_ok=True)
    yaml_path = os.path.join(base_dir, "td_best_params.yaml")

    payload = {
        "best_mavg_rew": float(best_score),
        "experiment_path": str(best_experiment_path),
        "params": dict(best_config),
    }

    if yaml is None:
        lines = [
            f"best_mavg_rew: {payload['best_mavg_rew']}",
            f"experiment_path: {payload['experiment_path']}",
            "params:",
        ]
        for k, v in payload["params"].items():
            lines.append(f"  {k}: {v}")
        content = "\n".join(lines) + "\n"
        with open(yaml_path, "w") as f:
            f.write(content)
    else:
        with open(yaml_path, "w") as f:
            yaml.safe_dump(payload, f, sort_keys=False)

    return yaml_path


def _build_agent(env, cfg_dict: Dict[str, Any]):
    sig = inspect.signature(TDMPC2Agent.__init__)
    if "config_overrides" in sig.parameters:
        return TDMPC2Agent(
            env.action_space,
            env.observation_space,
            config_overrides=cfg_dict,
        )

    # Fallback for older TDMPC2Agent versions (no config_overrides)
    agent = TDMPC2Agent(env.action_space, env.observation_space)

    # Disallow model-shape changes without full re-init
    incompatible = {"Z_DIM", "HIDDEN_DIM"}
    if any(k in cfg_dict for k in incompatible):
        raise RuntimeError(
            "This TDMPC2Agent version does not support config_overrides. "
            "Grid params include model-size keys (Z_DIM/HIDDEN_DIM). "
            "Update TDMPC2Agent or remove those keys from the grid."
        )

    _apply_overrides_compat(agent, cfg_dict)
    return agent


def _apply_overrides_compat(agent: Any, cfg_dict: Dict[str, Any]) -> None:
    if hasattr(agent, "configs") and isinstance(agent.configs, dict):
        agent.configs.update(cfg_dict)
    agent.__dict__.update(cfg_dict)

    # Update horizon-dependent buffers if needed
    if "TRAIN_HORIZON" in cfg_dict or "HORIZON" in cfg_dict:
        agent.TRAIN_HORIZON = getattr(agent, "TRAIN_HORIZON", cfg_dict.get("TRAIN_HORIZON", 12))
        agent.horizon = cfg_dict.get("HORIZON", agent.TRAIN_HORIZON)
        agent._a_mean = torch.zeros(agent.horizon, agent.act_dim, device=agent.device)

    # Recreate optimizers if optimizer hyperparams were overridden
    if any(k in cfg_dict for k in ("LR", "ADAM_BETA_1", "ADAM_BETA_2", "ADAM_EPS")):
        agent.LR = cfg_dict.get("LR", agent.LR)
        agent.ADAM_BETA_1 = cfg_dict.get("ADAM_BETA_1", agent.ADAM_BETA_1)
        agent.ADAM_BETA_2 = cfg_dict.get("ADAM_BETA_2", agent.ADAM_BETA_2)
        agent.ADAM_EPS = cfg_dict.get("ADAM_EPS", agent.ADAM_EPS)
        agent.optimizer = optim.Adam(
            agent.model.parameters(),
            lr=agent.LR,
            betas=(agent.ADAM_BETA_1, agent.ADAM_BETA_2),
            eps=agent.ADAM_EPS,
        )

    if "PI_LR" in cfg_dict:
        agent.PI_LR = cfg_dict.get("PI_LR", agent.PI_LR)
        agent.pi_optimizer = optim.Adam(agent.model.pi.parameters(), lr=agent.PI_LR)

    # Recreate replay buffer if its construction params were overridden
    if any(k in cfg_dict for k in ("CAPACITY", "PRIORITIZED", "ALPHA", "BETA", "EPSILON", "SEED")):
        agent.CAPACITY = cfg_dict.get("CAPACITY", agent.CAPACITY)
        agent.PRIORITIZED = cfg_dict.get("PRIORITIZED", agent.PRIORITIZED)
        agent.ALPHA = cfg_dict.get("ALPHA", agent.ALPHA)
        agent.BETA = cfg_dict.get("BETA", agent.BETA)
        agent.EPSILON = cfg_dict.get("EPSILON", agent.EPSILON)
        agent.SEED = cfg_dict.get("SEED", agent.SEED)
        rb_cls = agent.replay_buffer.__class__
        agent.replay_buffer = rb_cls(
            capacity_steps=agent.CAPACITY,
            prioritized=agent.PRIORITIZED,
            alpha=agent.ALPHA,
            beta=agent.BETA,
            eps=agent.EPSILON,
            seed=agent.SEED,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TD-MPC2 WandB grid tuning")
    parser.add_argument("--env", type=str, default="Hockey-One-v0")
    parser.add_argument("--base_dir", type=str, default="experiments")
    parser.add_argument("--grid_config", type=str, default=None, help="Path to YAML/JSON grid config")
    parser.add_argument("--wandb_project", type=str, default="tdmpc2-tune")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default=None, help="online|offline|disabled")
    parser.add_argument("--num_episodes", type=int, default=None)
    parser.add_argument("--max_runs", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.wandb_mode is None:
        args.wandb_mode = os.environ.get("WANDB_MODE", None)

    grid_spec = DEFAULT_GRID if args.grid_config is None else _load_grid_config(args.grid_config)
    grid = build_grid(grid_spec)

    if args.max_runs is not None:
        grid = grid[: args.max_runs]

    if len(grid) == 0:
        print("Grid is empty. Provide a valid grid configuration.", file=sys.stderr)
        return 1

    best_score = float("-inf")
    best_config = None
    best_experiment_path = ""
    best_yaml_path = None

    for idx, cfg in enumerate(grid):
        score, run_cfg, exp_path = run_trial(idx, cfg, args)
        if score > best_score:
            best_score = score
            best_config = run_cfg
            best_experiment_path = exp_path
            best_yaml_path = write_best_yaml(args.base_dir, best_score, best_config, best_experiment_path)

    if best_config is None:
        print("No successful runs.", file=sys.stderr)
        return 1

    if best_yaml_path is None:
        best_yaml_path = write_best_yaml(args.base_dir, best_score, best_config, best_experiment_path)
    print(f"Best config saved to: {best_yaml_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
