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
from pathlib import Path
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
try:
    import fcntl
except Exception:
    fcntl = None

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
    "Q_ENSEMBLE_SIZE",
    "EPISTEMIC_BONUS_DECAY_STEPS",
}


DEFAULT_GRID = {
    "LR": {"min": 1e-4, "max": 3e-4, "num": 2, "scale": "log"},
    "BATCH_SIZE": {"min": 128, "max": 512, "num": 2, "scale": "linear"},
    "TRAIN_HORIZON": {"min": 8, "max": 16, "num": 2, "scale": "linear"},
    "MPC_SAMPLES": {"min": 256, "max": 512, "num": 2, "scale": "linear"},
    "MPC_SIGMA": {"min": 0.2, "max": 0.6, "num": 2, "scale": "linear"},
}


SWEEP_META_KEYS = {"num_parallel_envs", "num_episodes", "training_rounds"}


def _to_plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _resolved_params_for_yaml(
    agent: Any,
    run_cfg: Dict[str, Any],
    num_parallel_envs: int,
    num_episodes: int | None,
    training_rounds: int,
    env_name: str,
    self_play: bool,
) -> Dict[str, Any]:
    # Start from full agent config (config file defaults + sweep overrides),
    # then layer in any explicit runtime meta-params.
    params = dict(getattr(agent, "configs", {}) or {})
    params.update(run_cfg)
    params["num_parallel_envs"] = int(num_parallel_envs)
    params["num_episodes"] = int(num_episodes) if num_episodes is not None else int(agent.NUM_EPISODES)
    params["training_rounds"] = int(training_rounds)
    params["env"] = str(env_name)
    params["self_play"] = bool(self_play)
    return _to_plain(params)


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


def _resolve_num_parallel_envs(cfg: Dict[str, Any], args: argparse.Namespace) -> int:
    raw = cfg.get("num_parallel_envs", args.num_parallel_envs)
    return max(1, int(raw))


def _resolve_num_episodes(cfg: Dict[str, Any], args: argparse.Namespace) -> int | None:
    raw = cfg.get("num_episodes", args.num_episodes)
    if raw is None:
        return None
    return int(raw)


def _resolve_training_rounds(cfg: Dict[str, Any], args: argparse.Namespace) -> int:
    raw = cfg.get("training_rounds", cfg.get("TRAINING_ROUNDS", args.training_rounds))
    return max(1, int(raw))


def _metric_value_or_floor(score: float, floor: float = -1e12) -> float:
    val = float(score)
    if np.isfinite(val):
        return val
    return float(floor)


def _build_env_bundle(env_name: str, num_parallel_envs: int):
    if num_parallel_envs > 1:
        env = gym.vector.SyncVectorEnv([lambda: make_env(env_name) for _ in range(num_parallel_envs)])
        obs_space = env.single_observation_space
        act_space = env.single_action_space
    else:
        env = make_env(env_name)
        obs_space = env.observation_space
        act_space = env.action_space
    return env, obs_space, act_space


def _build_self_play_bundle():
    env = h_env.HockeyEnv()
    obs_space = env.observation_space
    if not isinstance(env.action_space, spaces.Box):
        raise RuntimeError("Self-play requires a continuous Box action space.")
    low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
    if low.shape[0] < 4 or high.shape[0] < 4:
        raise RuntimeError("Could not extract single-player action space (expected at least 4 action dims).")
    act_space = spaces.Box(
        low=low[:4],
        high=high[:4],
        shape=(4,),
        dtype=np.float32,
    )
    return env, obs_space, act_space


def _run_training_once(
    cfg_dict: Dict[str, Any],
    args: argparse.Namespace,
    allow_partial_on_interrupt: bool = False,
):
    run_cfg = dict(cfg_dict)
    num_episodes = _resolve_num_episodes(run_cfg, args)
    training_rounds = _resolve_training_rounds(run_cfg, args)
    num_parallel_envs = _resolve_num_parallel_envs(run_cfg, args)
    for key in SWEEP_META_KEYS:
        run_cfg.pop(key, None)

    self_play_population_path: str | None = None
    if args.self_play_population_path:
        p = Path(args.self_play_population_path)
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Self-play opponent pool not found: {p}")
        if not p.is_dir():
            raise NotADirectoryError(f"Self-play opponent pool is not a directory: {p}")
        self_play_population_path = str(p)
        run_cfg["self_play_population_path"] = self_play_population_path
        # If an external pool is provided, fixed-opponent sampling should be enabled.
        run_cfg["FIXED_OPPONENTS"] = True

    if args.self_play:
        if args.env != "Hockey-One-v0":
            raise RuntimeError("Self-play is currently supported only for Hockey-One-v0.")
        env, obs_space, act_space = _build_self_play_bundle()
        try:
            agent = _build_agent(act_space, obs_space, run_cfg)
            opponent = _build_agent(act_space, obs_space, run_cfg)
            if num_episodes is not None:
                agent.NUM_EPISODES = int(num_episodes)
                opponent.NUM_EPISODES = int(num_episodes)
            agent.TRAINING_ROUNDS = int(training_rounds)
            opponent.TRAINING_ROUNDS = int(training_rounds)

            trainer = Training(
                agent=agent,
                env=env,
                base_dir=args.base_dir,
                save_intermediate_agents=False,
                verbose=args.verbose,
            )
            interrupted = False
            try:
                trainer.train_self_play(
                    opponent,
                    discrete_actions=False,
                    population_path=self_play_population_path,
                    num_parallel_envs=num_parallel_envs,
                )
            except KeyboardInterrupt:
                if not allow_partial_on_interrupt:
                    raise
                interrupted = True

            score = score_from_stats(trainer.statistics, trainer.mavg_window_size)
            resolved_params = _resolved_params_for_yaml(
                agent=agent,
                run_cfg=run_cfg,
                num_parallel_envs=num_parallel_envs,
                num_episodes=num_episodes,
                training_rounds=training_rounds,
                env_name=args.env,
                self_play=bool(args.self_play),
            )
            return score, resolved_params, trainer.experiment_path, num_parallel_envs, interrupted
        finally:
            env.close()

    env, obs_space, act_space = _build_env_bundle(args.env, num_parallel_envs)
    try:
        agent = _build_agent(act_space, obs_space, run_cfg)
        if num_episodes is not None:
            agent.NUM_EPISODES = int(num_episodes)

        trainer = Training(
            agent=agent,
            env=env,
            base_dir=args.base_dir,
            save_intermediate_agents=False,
            verbose=args.verbose,
        )
        interrupted = False
        try:
            trainer.train()
        except KeyboardInterrupt:
            if not allow_partial_on_interrupt:
                raise
            interrupted = True

        score = score_from_stats(trainer.statistics, trainer.mavg_window_size)
        resolved_params = _resolved_params_for_yaml(
            agent=agent,
            run_cfg=run_cfg,
            num_parallel_envs=num_parallel_envs,
            num_episodes=num_episodes,
            training_rounds=training_rounds,
            env_name=args.env,
            self_play=bool(args.self_play),
        )
        return score, resolved_params, trainer.experiment_path, num_parallel_envs, interrupted
    finally:
        env.close()


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

    cfg_dict = dict(run.config)
    trial_seed = args.seed
    if "SEED" in cfg_dict:
        trial_seed = int(cfg_dict["SEED"]) + trial_idx
    set_seeds(trial_seed)

    try:
        score, resolved_params, exp_path, num_parallel_envs, _ = _run_training_once(cfg_dict, args)
        run.summary["best_mavg_rew"] = score
        run.summary["num_parallel_envs"] = int(num_parallel_envs)
        wandb.log({"best_mavg_rew": score})
        return score, resolved_params, exp_path
    finally:
        run.finish()


def write_best_yaml(
    base_dir: str,
    best_score: float,
    best_config: Dict[str, Any],
    best_experiment_path: str,
    best_yaml_name: str = "td_best_params.yaml",
):
    os.makedirs(base_dir, exist_ok=True)
    yaml_path = _resolve_best_yaml_path(base_dir, best_yaml_name)
    os.makedirs(os.path.dirname(yaml_path) or ".", exist_ok=True)

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
        tmp_path = f"{yaml_path}.tmp"
        with open(tmp_path, "w") as f:
            f.write(content)
    else:
        tmp_path = f"{yaml_path}.tmp"
        with open(tmp_path, "w") as f:
            yaml.safe_dump(payload, f, sort_keys=False)
    os.replace(tmp_path, yaml_path)

    return yaml_path


def _resolve_best_yaml_path(base_dir: str, best_yaml_name: str) -> str:
    if os.path.isabs(best_yaml_name):
        return best_yaml_name
    return os.path.join(base_dir, best_yaml_name)


def _read_best_score_from_path(yaml_path: str) -> float:
    if not os.path.exists(yaml_path):
        return float("-inf")
    try:
        with open(yaml_path, "r") as f:
            if yaml is None:
                for line in f:
                    if line.strip().startswith("best_mavg_rew:"):
                        _, value = line.split(":", 1)
                        return float(value.strip())
                return float("-inf")
            data = yaml.safe_load(f) or {}
            return float(data.get("best_mavg_rew", float("-inf")))
    except Exception:
        return float("-inf")


def _read_existing_best_score(
    base_dir: str,
    best_yaml_name: str = "td_best_params.yaml",
) -> float:
    yaml_path = _resolve_best_yaml_path(base_dir, best_yaml_name)
    return _read_best_score_from_path(yaml_path)


def maybe_update_best_yaml(
    base_dir: str,
    score: float,
    config: Dict[str, Any],
    experiment_path: str,
    best_yaml_name: str = "td_best_params.yaml",
) -> tuple[bool, str | None]:
    os.makedirs(base_dir, exist_ok=True)
    yaml_path = _resolve_best_yaml_path(base_dir, best_yaml_name)
    lock_path = f"{yaml_path}.lock"
    os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
    with open(lock_path, "w") as lock_file:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            prev_best = _read_existing_best_score(base_dir, best_yaml_name=best_yaml_name)
            if score <= prev_best:
                return False, None
            yaml_path = write_best_yaml(
                base_dir,
                score,
                config,
                experiment_path,
                best_yaml_name=best_yaml_name,
            )
            return True, yaml_path
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _build_agent(action_space, observation_space, cfg_dict: Dict[str, Any]):
    sig = inspect.signature(TDMPC2Agent.__init__)
    if "config_overrides" in sig.parameters:
        return TDMPC2Agent(
            action_space,
            observation_space,
            config_overrides=cfg_dict,
        )

    # Fallback for older TDMPC2Agent versions (no config_overrides)
    agent = TDMPC2Agent(action_space, observation_space)

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

    # Recreate optimizers if optimizer settings were overridden
    optimizer_keys = {
        "OPTIMIZER",
        "LR",
        "PI_LR",
        "WEIGHT_DECAY",
        "PI_WEIGHT_DECAY",
        "ADAM_BETA_1",
        "ADAM_BETA_2",
        "ADAM_EPS",
        "MUON_LR",
        "MUON_PI_LR",
        "MUON_MOMENTUM",
        "MUON_WEIGHT_DECAY",
        "MUON_PI_WEIGHT_DECAY",
        "MUON_ADAM_LR",
        "MUON_ADAM_BETA_1",
        "MUON_ADAM_BETA_2",
        "MUON_ADAM_EPS",
        "MUON_ADAM_WEIGHT_DECAY",
        "MUON_PI_ADAM_LR",
        "MUON_PI_ADAM_WEIGHT_DECAY",
    }
    if any(k in cfg_dict for k in optimizer_keys):
        for k in optimizer_keys:
            if k in cfg_dict:
                setattr(agent, k, cfg_dict[k])
        if hasattr(agent, "_reset_optimizers"):
            agent._reset_optimizers()
        else:
            # Backward-compatible fallback to Adam
            agent.optimizer = optim.Adam(
                agent.model.parameters(),
                lr=agent.LR,
                betas=(agent.ADAM_BETA_1, agent.ADAM_BETA_2),
                eps=agent.ADAM_EPS,
            )
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
    parser = argparse.ArgumentParser(description="TD-MPC2 WandB tuning")
    parser.add_argument("--env", type=str, default="Hockey-One-v0")
    parser.add_argument("--base_dir", type=str, default="experiments")
    parser.add_argument("--grid_config", type=str, default=None, help="Path to YAML/JSON grid config")
    parser.add_argument("--wandb_project", type=str, default="tdmpc2-tune")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default=None, help="online|offline|disabled")
    parser.add_argument("--num_parallel_envs", type=int, default=1)
    parser.add_argument("--num_episodes", type=int, default=None)
    parser.add_argument("--training_rounds", type=int, default=1)
    parser.add_argument(
        "--best_yaml_name",
        type=str,
        default="td_best_params.yaml",
        help="Filename (or absolute path) for best-params YAML output.",
    )
    parser.add_argument("--max_runs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--self_play",
        action="store_true",
        help="Use self-play training (Hockey-One-v0 only).",
    )
    parser.add_argument(
        "--self_play_population_path",
        type=str,
        default=None,
        help="Optional folder with opponent checkpoints (.pth) for fixed-opponent self-play.",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Single-run mode for W&B sweeps (uses wandb.config as hyperparameters).",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def run_grid_tuning(args: argparse.Namespace) -> int:
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
            best_yaml_path = write_best_yaml(
                args.base_dir,
                best_score,
                best_config,
                best_experiment_path,
                best_yaml_name=args.best_yaml_name,
            )

    if best_config is None:
        print("No successful runs.", file=sys.stderr)
        return 1

    if best_yaml_path is None:
        best_yaml_path = write_best_yaml(
            args.base_dir,
            best_score,
            best_config,
            best_experiment_path,
            best_yaml_name=args.best_yaml_name,
        )
    print(f"Best config saved to: {best_yaml_path}")
    return 0


def run_sweep_once(args: argparse.Namespace) -> int:
    if wandb is None:
        raise RuntimeError("wandb is not installed. Install it or remove wandb usage.")

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config={},
        mode=args.wandb_mode,
        reinit=True,
    )

    cfg_dict = dict(run.config)
    run_seed = args.seed
    if "SEED" in cfg_dict:
        run_seed = int(cfg_dict["SEED"])
    set_seeds(run_seed)

    score = float("-inf")
    resolved_params: Dict[str, Any] = dict(cfg_dict)
    exp_path = ""
    num_parallel_envs = _resolve_num_parallel_envs(cfg_dict, args)
    interrupted = False
    updated_best = False
    best_yaml_path = None

    try:
        score, resolved_params, exp_path, num_parallel_envs, interrupted = _run_training_once(
            cfg_dict, args, allow_partial_on_interrupt=True
        )
        updated_best, best_yaml_path = maybe_update_best_yaml(
            args.base_dir,
            score,
            resolved_params,
            exp_path,
            best_yaml_name=args.best_yaml_name,
        )
    except Exception as e:
        run.summary["run_error"] = str(e)
        run.summary["run_failed"] = True
        raise
    finally:
        score_to_log = _metric_value_or_floor(score)
        run.summary["best_mavg_rew"] = score_to_log
        run.summary["experiment_path"] = exp_path
        run.summary["num_parallel_envs"] = int(num_parallel_envs)
        run.summary["interrupted"] = bool(interrupted)
        run.summary["best_yaml_updated"] = bool(updated_best)
        if best_yaml_path is not None:
            run.summary["best_yaml_path"] = best_yaml_path
        wandb.log(
            {
                "best_mavg_rew": score_to_log,
                "interrupted": int(interrupted),
            }
        )
        run.finish()
    return 0


def main() -> int:
    args = parse_args()

    # Ensure relative paths (like configs/tdmpc_config.yaml) resolve from repo root
    repo_root = Path(__file__).resolve().parent
    os.chdir(repo_root)

    if args.wandb_mode is None:
        args.wandb_mode = os.environ.get("WANDB_MODE", None)

    if args.sweep:
        return run_sweep_once(args)
    return run_grid_tuning(args)


if __name__ == "__main__":
    raise SystemExit(main())
