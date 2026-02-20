#!/usr/bin/env python3
"""
Retrain TD-MPC2 from a saved best-params YAML.

Expected YAML format:
  best_mavg_rew: ...
  experiment_path: ...
  params:
    LR: ...
    ...
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Train import Training
from TDMPC2_tune import (
    SWEEP_META_KEYS,
    _build_agent,
    _build_env_bundle,
    _build_self_play_bundle,
    score_from_stats,
    set_seeds,
)


def _load_best_params(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    if isinstance(data, dict) and isinstance(data.get("params"), dict):
        return dict(data["params"])
    if isinstance(data, dict):
        return dict(data)
    raise RuntimeError(f"Unsupported YAML structure in {path}")


def _resolve_existing_path(path_str: str, repo_root: Path, label: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrain TD-MPC2 from best-params YAML")
    parser.add_argument(
        "--best_yaml",
        type=str,
        default="experiments/td_best_params.yaml",
        help="Path to best-params YAML (e.g. experiments/td_best_params_muon.yaml).",
    )
    parser.add_argument("--env", type=str, default="Hockey-One-v0")
    parser.add_argument("--base_dir", type=str, default="experiments")
    parser.add_argument("--num_parallel_envs", type=int, default=None)
    parser.add_argument("--num_episodes", type=int, default=None)
    parser.add_argument("--training_rounds", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--agent_load_path",
        type=str,
        default=None,
        help="Optional checkpoint path to warm-start retraining from (.pth).",
    )
    parser.add_argument(
        "--self_play",
        action="store_true",
        help="Use self-play retraining (Hockey-One-v0 only).",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    os.chdir(repo_root)

    best_yaml_path = _resolve_existing_path(args.best_yaml, repo_root, "Best-params YAML")
    agent_load_path = None
    if args.agent_load_path is not None and str(args.agent_load_path).strip():
        agent_load_path = _resolve_existing_path(
            args.agent_load_path,
            repo_root,
            "Agent checkpoint",
        )

    cfg_dict = _load_best_params(str(best_yaml_path))

    if args.seed is not None:
        cfg_dict["SEED"] = int(args.seed)
    set_seeds(cfg_dict.get("SEED", None))

    if args.num_episodes is not None:
        cfg_dict["num_episodes"] = int(args.num_episodes)
    if args.training_rounds is not None:
        cfg_dict["training_rounds"] = int(args.training_rounds)
    if args.num_parallel_envs is not None:
        cfg_dict["num_parallel_envs"] = int(args.num_parallel_envs)

    run_cfg = dict(cfg_dict)
    num_episodes = run_cfg.get("num_episodes", None)
    training_rounds = int(run_cfg.get("training_rounds", run_cfg.get("TRAINING_ROUNDS", 1)))
    num_parallel_envs = int(run_cfg.get("num_parallel_envs", 1))
    for key in SWEEP_META_KEYS:
        run_cfg.pop(key, None)

    if args.self_play:
        if args.env != "Hockey-One-v0":
            raise RuntimeError("Self-play retraining is currently supported only for Hockey-One-v0.")
        env, obs_space, act_space = _build_self_play_bundle()
        try:
            agent = _build_agent(act_space, obs_space, run_cfg)
            opponent = _build_agent(act_space, obs_space, run_cfg)
            if agent_load_path is not None:
                agent.load_dict(str(agent_load_path))
                opponent.load_dict(str(agent_load_path))
                if args.verbose:
                    print(f"Warm-start checkpoint loaded: {agent_load_path}")
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
            trainer.train_self_play(opponent, discrete_actions=False)
            score = score_from_stats(trainer.statistics, trainer.mavg_window_size)
        finally:
            env.close()
    else:
        env, obs_space, act_space = _build_env_bundle(args.env, num_parallel_envs)
        try:
            agent = _build_agent(act_space, obs_space, run_cfg)
            if agent_load_path is not None:
                agent.load_dict(str(agent_load_path))
                if args.verbose:
                    print(f"Warm-start checkpoint loaded: {agent_load_path}")
            if num_episodes is not None:
                agent.NUM_EPISODES = int(num_episodes)

            trainer = Training(
                agent=agent,
                env=env,
                base_dir=args.base_dir,
                save_intermediate_agents=False,
                verbose=args.verbose,
            )
            trainer.train()
            score = score_from_stats(trainer.statistics, trainer.mavg_window_size)
        finally:
            env.close()

    # Always keep one explicit final checkpoint in addition to any "best" checkpoint logic.
    agent.save_dict(trainer.experiment_path, identifier_extension="_retrain_final")

    print(f"Retrain finished. score={score:.6f}")
    print(f"Artifacts written to: {trainer.experiment_path}")
    print(f"Final checkpoint: {trainer.experiment_path}/{agent.MODEL_IDENTIFIER}_retrain_final.pth")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
