import argparse
import os
from typing import Any, Dict

import yaml


def _log_uniform_around(value: float, factor: float = 10.0, min_floor: float = 1e-8) -> Dict[str, Any]:
    v = float(value)
    lo = max(v / factor, min_floor)
    hi = v * factor
    return {"distribution": "log_uniform_values", "min": lo, "max": hi}


def _uniform_around(value: float, factor: float = 2.0, min_floor: float = 1e-6, max_cap: float = 1.0) -> Dict[str, Any]:
    v = float(value)
    lo = max(v / factor, min_floor)
    hi = min(v * factor, max_cap)
    return {"distribution": "uniform", "min": lo, "max": hi}


def _batch_candidates(value: int) -> Dict[str, Any]:
    v = int(value)
    candidates = sorted({max(16, v // 2), v, v * 2})
    return {"values": candidates}


def build_sweep_config(
    ddpg_cfg: Dict[str, Any],
    env_name: str,
    base_dir: str,
    num_parallel_envs: int,
    metric_name: str,
    method: str,
) -> Dict[str, Any]:
    params = {}
    if "MODEL_IDENTIFIER" in ddpg_cfg:
        params["model_identifier"] = {"values": [str(ddpg_cfg["MODEL_IDENTIFIER"])]}
    if "START_TRAINING" in ddpg_cfg:
        params["start_training"] = _batch_candidates(ddpg_cfg["START_TRAINING"])
    if "EPS" in ddpg_cfg:
        params["eps"] = _uniform_around(ddpg_cfg["EPS"], factor=2.0, min_floor=0.01, max_cap=0.5)
    if "LEARNING_RATE_ACTOR" in ddpg_cfg:
        params["learning_rate_actor"] = _log_uniform_around(ddpg_cfg["LEARNING_RATE_ACTOR"])
    if "LEARNING_RATE_CRITIC" in ddpg_cfg:
        params["learning_rate_critic"] = _log_uniform_around(ddpg_cfg["LEARNING_RATE_CRITIC"])
    if "BATCH_SIZE" in ddpg_cfg:
        params["batch_size"] = _batch_candidates(ddpg_cfg["BATCH_SIZE"])
    if "TAU" in ddpg_cfg:
        params["tau"] = _uniform_around(ddpg_cfg["TAU"])
    if "UPDATE_TARGET_EVERY" in ddpg_cfg:
        params["update_target_every"] = _batch_candidates(ddpg_cfg["UPDATE_TARGET_EVERY"])
    if "TWIN_DELAYED" in ddpg_cfg:
        params["twin_delayed"] = {"values": [True, False]}
    if "POLICY_NOISE" in ddpg_cfg:
        params["policy_noise"] = _log_uniform_around(ddpg_cfg["POLICY_NOISE"], factor=5.0, min_floor=1e-4)
    if "DISCOUNT" in ddpg_cfg:
        params["discount"] = _uniform_around(ddpg_cfg["DISCOUNT"], factor=1.02, min_floor=0.90, max_cap=0.999)
    if "TRAIN_ITERATIONS" in ddpg_cfg:
        params["train_iterations"] = _batch_candidates(ddpg_cfg["TRAIN_ITERATIONS"])
    if "POLICY_DELAY" in ddpg_cfg:
        params["policy_delay"] = {"values": sorted({1, int(ddpg_cfg["POLICY_DELAY"]), 3})}
    if "ACTION_NOISE_THETA" in ddpg_cfg:
        params["action_noise_theta"] = _log_uniform_around(
            ddpg_cfg["ACTION_NOISE_THETA"], factor=5.0, min_floor=1e-4
        )
    if "ACTION_NOISE_DT" in ddpg_cfg:
        params["action_noise_dt"] = _log_uniform_around(
            ddpg_cfg["ACTION_NOISE_DT"], factor=5.0, min_floor=1e-4
        )
    if "NOISE_CLIP" in ddpg_cfg:
        params["noise_clip"] = _log_uniform_around(ddpg_cfg["NOISE_CLIP"], factor=5.0, min_floor=1e-3)
    if "USE_DISTRIBUTIONAL" in ddpg_cfg:
        params["use_distributional"] = {"values": [True, False]}
    if "NUM_QUANTILES" in ddpg_cfg:
        params["num_quantiles"] = {"values": [10, 25, 50, 100]}
    if "TQC_TOP_K" in ddpg_cfg:
        base_k = int(ddpg_cfg["TQC_TOP_K"])
        params["tqc_top_k"] = {"values": sorted({max(0, base_k - 5), base_k, base_k + 5})}

    sweep = {
        "program": "hockey007/DDPG_tune.py",
        "method": method,
        "metric": {"name": metric_name, "goal": "maximize"},
        "parameters": params,
        "command": [
            "${env}",
            "python",
            "${program}",
            "--env",
            env_name,
            "--base_dir",
            base_dir,
            "--num_parallel_envs",
            str(num_parallel_envs),
        ],
    }
    return sweep


def build_sbatch_script(num_jobs: int) -> str:
    array_spec = f"0-{num_jobs - 1}"
    return f"""#!/bin/bash

####
#a) Define slurm job parameters
####

#SBATCH --job-name=ddpg-sweep

#resources:

#SBATCH --cpus-per-task=4
# the job can use and see 4 CPUs (from max 24).

#SBATCH --partition=week
# the slurm partition the job is queued to.

#SBATCH --mem-per-cpu=3G
# the job will need 12GB of memory equally distributed on 4 cpus.

#SBATCH --gres=gpu:1
# the job can use and see 1 GPUs.

#SBATCH --time=3-00:00
# the maximum time the scripts needs to run

#SBATCH --error=./job_outputs/job.%J.err
# write the error output to ./job_outputs/job.jobID.err

#SBATCH --output=./job_outputs/job.%J.out
# write the standard output to ./job_outputs/job.jobID.out

#SBATCH --array={array_spec}

####
#c) Execute your file.
####

# Example: module load cuda
# source /path/to/venv/bin/activate

# Required env vars:
# export WANDB_API_KEY=your_key
# export WANDB_PROJECT=your_project
# export WANDB_ENTITY=your_entity

# Set the sweep id obtained from `wandb sweep`
SWEEP_ID=${{SWEEP_ID:-REPLACE_ME}}

wandb agent ${{WANDB_ENTITY}}/${{WANDB_PROJECT}}/${{SWEEP_ID}}

echo DONE!
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a W&B sweep config and sbatch launcher for DDPG.")
    parser.add_argument(
        "--config",
        default="hockey007/configs/ddpg_config.yaml",
        help="Path to DDPG config yaml.",
    )
    parser.add_argument("--env", default="Hockey-One-v0", help="Environment name.")
    parser.add_argument("--base_dir", default="experiments", help="Base directory for experiments.")
    parser.add_argument(
        "--num_parallel_envs",
        type=int,
        default=1,
        help="Number of parallel envs for training.",
    )
    parser.add_argument(
        "--metric_name",
        default="eval/mean_reward",
        help="W&B metric name to optimize.",
    )
    parser.add_argument(
        "--method",
        default="bayes",
        choices=["bayes", "random", "grid"],
        help="W&B sweep method.",
    )
    parser.add_argument(
        "--num_jobs",
        type=int,
        default=8,
        help="Number of parallel sbatch jobs.",
    )
    parser.add_argument(
        "--output_dir",
        default="hockey007/sweeps",
        help="Output directory for sweep yaml and sbatch script.",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        ddpg_cfg = yaml.safe_load(f) or {}

    sweep_cfg = build_sweep_config(
        ddpg_cfg=ddpg_cfg,
        env_name=args.env,
        base_dir=args.base_dir,
        num_parallel_envs=args.num_parallel_envs,
        metric_name=args.metric_name,
        method=args.method,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    sweep_path = os.path.join(args.output_dir, "ddpg_sweep.yaml")
    sbatch_path = os.path.join(args.output_dir, "run_ddpg_sweep.sbatch")

    with open(sweep_path, "w") as f:
        yaml.safe_dump(sweep_cfg, f, sort_keys=False)

    with open(sbatch_path, "w") as f:
        f.write(build_sbatch_script(args.num_jobs))

    print("Wrote sweep config:", sweep_path)
    print("Wrote sbatch script:", sbatch_path)


if __name__ == "__main__":
    main()
