#!/usr/bin/env python3
"""
DDPG_tune.py

WandB-based sweep runner for DDPG without modifying main.py or Train.py.
Builds a merged config per run, trains via Training, and logs a summary score.
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

try:
    import wandb
except Exception:  # pragma: no cover - handled at runtime
    wandb = None

try:
    import yaml
except Exception:
    yaml = None

import gymnasium as gym

from Train import Training
from agents.DDPGAgent import DDPGAgent


INT_KEYS = {
    "NUM_EPISODES",
    "START_TRAINING",
    "BUFFER_SIZE",
    "BATCH_SIZE",
    "TRAIN_ITERATIONS",
    "UPDATE_TARGET_EVERY",
    "POLICY_DELAY",
    "NUM_QUANTILES",
    "TQC_TOP_K",
}

BOOL_KEYS = {
    "USE_TARGET_NET",
    "TWIN_DELAYED",
    "USE_DISTRIBUTIONAL",
}

STR_KEYS = {
    "MODEL_IDENTIFIER",
}


PARAM_MAP = {
    "model_identifier": "MODEL_IDENTIFIER",
    "start_training": "START_TRAINING",
    "eps": "EPS",
    "learning_rate_actor": "LEARNING_RATE_ACTOR",
    "learning_rate_critic": "LEARNING_RATE_CRITIC",
    "batch_size": "BATCH_SIZE",
    "tau": "TAU",
    "update_target_every": "UPDATE_TARGET_EVERY",
    "twin_delayed": "TWIN_DELAYED",
    "policy_noise": "POLICY_NOISE",
    "discount": "DISCOUNT",
    "train_iterations": "TRAIN_ITERATIONS",
    "policy_delay": "POLICY_DELAY",
    "action_noise_theta": "ACTION_NOISE_THETA",
    "action_noise_dt": "ACTION_NOISE_DT",
    "noise_clip": "NOISE_CLIP",
    "use_distributional": "USE_DISTRIBUTIONAL",
    "num_quantiles": "NUM_QUANTILES",
    "tqc_top_k": "TQC_TOP_K",
}


def set_seeds(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env(env_name: str):
    if env_name == "Pendulum-v1":
        return gym.make(env_name, disable_env_checker=True)
    if env_name == "Hockey-One-v0":
        return gym.make("Hockey-One-v0", mode=0, weak_opponent=True, disable_env_checker=True)
    return gym.make(env_name, disable_env_checker=True)


def load_config(path: str) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML not available; install it to load DDPG config.")
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def apply_overrides(cfg: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(cfg)
    for key, value in overrides.items():
        cfg_key = PARAM_MAP.get(key, key)
        if cfg_key not in merged:
            continue
        if cfg_key in STR_KEYS:
            merged[cfg_key] = str(value)
        elif cfg_key in BOOL_KEYS:
            merged[cfg_key] = bool(value)
        elif cfg_key in INT_KEYS:
            merged[cfg_key] = int(round(float(value)))
        else:
            merged[cfg_key] = float(value)
    return merged


def write_config(base_dir: str, run_id: str, cfg: Dict[str, Any]) -> str:
    os.makedirs(base_dir, exist_ok=True)
    path = os.path.join(base_dir, f"ddpg_sweep_{run_id}.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DDPG WandB sweep runner")
    parser.add_argument("--env", type=str, default="Hockey-One-v0")
    parser.add_argument("--base_dir", type=str, default="experiments")
    parser.add_argument("--config", type=str, default="configs/ddpg_config.yaml")
    parser.add_argument("--num_parallel_envs", type=int, default=1)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default=None, help="online|offline|disabled")
    parser.add_argument("--num_episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    if wandb is None:
        raise RuntimeError("wandb is not installed. Install it or remove wandb usage.")

    args = parse_args()

    # Ensure relative paths resolve from repo root
    repo_root = Path(__file__).resolve().parent
    os.chdir(repo_root)

    if args.wandb_mode is None:
        args.wandb_mode = os.environ.get("WANDB_MODE", None)
    if args.wandb_project is None:
        args.wandb_project = os.environ.get("WANDB_PROJECT", "ddpg-tune")
    if args.wandb_entity is None:
        args.wandb_entity = os.environ.get("WANDB_ENTITY", None)

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config={},
        mode=args.wandb_mode,
        reinit=True,
    )

    set_seeds(args.seed)

    base_cfg = load_config(args.config)
    run_cfg = apply_overrides(base_cfg, dict(run.config))
    config_dir = os.path.join(args.base_dir, "ddpg_sweep_configs")
    cfg_path = write_config(config_dir, run.id, run_cfg)

    if args.num_parallel_envs > 1:
        env = gym.vector.SyncVectorEnv([lambda: make_env(args.env) for _ in range(args.num_parallel_envs)])
        base_obs_space = env.single_observation_space
        base_act_space = env.single_action_space
    else:
        env = make_env(args.env)
        base_obs_space = env.observation_space
        base_act_space = env.action_space

    try:
        agent = DDPGAgent(base_obs_space, base_act_space, config_path=cfg_path, verbose=args.verbose)
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
    finally:
        env.close()
        run.finish()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
