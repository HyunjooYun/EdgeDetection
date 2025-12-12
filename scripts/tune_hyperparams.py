"""Run Ray Tune hyperparameter searches for HED post-processing agents."""

from __future__ import annotations

import argparse
import random
from dataclasses import replace
from pathlib import Path
from typing import Dict

import numpy as np
import ray
from ray import tune
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

from hed_rl import HEDPostProcessConfig, HEDPostProcessEnv, HedConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algo", choices=["dqn", "ppo"], default="ppo")
    parser.add_argument("--prototxt", type=Path, help="Path to HED deploy.prototxt")
    parser.add_argument("--caffemodel", type=Path, help="Path to HED caffemodel")
    parser.add_argument("--image-dir", type=Path, default=Path("imgs/test"))
    parser.add_argument("--ground-truth-dir", type=Path, help="Directory with ground-truth edge maps")
    parser.add_argument("--edge-dir", type=Path, help="Directory with precomputed base HED edge maps")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--max-episode-steps", type=int, default=30)
    parser.add_argument("--hed-width", type=int, default=0)
    parser.add_argument("--hed-height", type=int, default=0)
    parser.add_argument("--no-cache-edges", action="store_true")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of Ray Tune trials")
    parser.add_argument("--cpus-per-trial", type=float, default=1.0)
    parser.add_argument("--gpus-per-trial", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-dir", type=Path, default=Path("runs/tune"), help="Directory for Ray Tune results")
    parser.add_argument("--run-name", type=str, default="hed_rl_tune")
    parser.add_argument("--metric", type=str, default="mean_reward")
    parser.add_argument("--mode", type=str, default="max", choices=["min", "max"])
    return parser.parse_args()


def build_env_config(args: argparse.Namespace) -> HEDPostProcessConfig:
    input_size = None
    if args.hed_width > 0 and args.hed_height > 0:
        input_size = (args.hed_width, args.hed_height)

    hed_config = None
    if args.prototxt and args.caffemodel:
        hed_config = HedConfig(
            prototxt_path=args.prototxt,
            caffemodel_path=args.caffemodel,
            input_size=input_size,
        )
    elif not args.edge_dir:
        raise ValueError("Either provide HED prototxt/caffemodel or a precomputed edge directory")

    return HEDPostProcessConfig(
        image_dir=args.image_dir,
        hed_config=hed_config,
        ground_truth_dir=args.ground_truth_dir,
        precomputed_edge_dir=args.edge_dir,
        max_steps=args.max_episode_steps,
        cache_edges=not args.no_cache_edges,
    )


def dqn_search_space() -> Dict[str, object]:
    return {
        "learning_rate": tune.loguniform(1e-5, 5e-4),
        "gamma": tune.uniform(0.90, 0.999),
        "batch_size": tune.choice([32, 64, 128]),
        "buffer_size": tune.choice([50_000, 100_000, 200_000]),
        "train_freq": tune.choice([1, 4, 8]),
        "target_update_interval": tune.choice([500, 1000, 2000]),
        "gradient_steps": tune.choice([1, 2, 4]),
        "exploration_fraction": tune.uniform(0.1, 0.4),
        "exploration_final_eps": tune.uniform(0.01, 0.1),
        "seed": tune.randint(0, 1_000_000),
    }


def ppo_search_space(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "learning_rate": tune.loguniform(1e-5, 5e-4),
        "gamma": tune.uniform(0.90, 0.999),
        "n_steps": tune.choice([128, 256, 512, 1024]),
        "batch_size": tune.choice([64, 128, 256, 512]),
        "gae_lambda": tune.uniform(0.8, 0.98),
        "clip_range": tune.uniform(0.1, 0.3),
        "ent_coef": tune.loguniform(1e-5, 1e-2),
        "vf_coef": tune.uniform(0.3, 1.0),
        "seed": tune.randint(0, 1_000_000),
    }


def train_dqn_tune(
    config: Dict[str, float],
    base_env_config: HEDPostProcessConfig,
    args: argparse.Namespace,
) -> None:
    seed = int(config.get("seed", args.seed))
    random.seed(seed)
    np.random.seed(seed)

    env_config = replace(base_env_config, random_seed=seed)

    def make_env() -> HEDPostProcessEnv:
        return HEDPostProcessEnv(env_config)

    vec_env = make_vec_env(make_env, n_envs=args.n_envs, seed=seed)
    model = DQN(
        "MlpPolicy",
        vec_env,
        learning_rate=float(config["learning_rate"]),
        gamma=float(config["gamma"]),
        batch_size=int(config["batch_size"]),
        buffer_size=int(config["buffer_size"]),
        train_freq=int(config["train_freq"]),
        gradient_steps=int(config["gradient_steps"]),
        target_update_interval=int(config["target_update_interval"]),
        exploration_fraction=float(config["exploration_fraction"]),
        exploration_final_eps=float(config["exploration_final_eps"]),
        seed=seed,
        verbose=0,
    )

    model.learn(total_timesteps=args.timesteps)

    eval_env = make_vec_env(make_env, n_envs=1, seed=seed + 1)
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=args.eval_episodes, deterministic=True
    )

    vec_env.close()
    eval_env.close()
    tune.report({"mean_reward": float(mean_reward), "reward_std": float(std_reward)})


def train_ppo_tune(
    config: Dict[str, float],
    base_env_config: HEDPostProcessConfig,
    args: argparse.Namespace,
) -> None:
    seed = int(config.get("seed", args.seed))
    random.seed(seed)
    np.random.seed(seed)

    env_config = replace(base_env_config, random_seed=seed)

    def make_env() -> HEDPostProcessEnv:
        return HEDPostProcessEnv(env_config)

    total_rollout = int(config["n_steps"]) * args.n_envs
    batch_size = min(int(config["batch_size"]), total_rollout)

    vec_env = make_vec_env(make_env, n_envs=args.n_envs, seed=seed)
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=float(config["learning_rate"]),
        gamma=float(config["gamma"]),
        n_steps=int(config["n_steps"]),
        batch_size=batch_size,
        gae_lambda=float(config["gae_lambda"]),
        clip_range=float(config["clip_range"]),
        ent_coef=float(config["ent_coef"]),
        vf_coef=float(config["vf_coef"]),
        seed=seed,
        verbose=0,
    )

    model.learn(total_timesteps=args.timesteps)

    eval_env = make_vec_env(make_env, n_envs=1, seed=seed + 1)
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=args.eval_episodes, deterministic=True
    )

    vec_env.close()
    eval_env.close()
    tune.report({"mean_reward": float(mean_reward), "reward_std": float(std_reward)})


def main() -> None:
    args = parse_args()
    args.image_dir = args.image_dir.resolve()
    if args.ground_truth_dir is not None:
        args.ground_truth_dir = args.ground_truth_dir.resolve()
    if args.edge_dir is not None:
        args.edge_dir = args.edge_dir.resolve()
    if args.prototxt is not None:
        args.prototxt = args.prototxt.resolve()
    if args.caffemodel is not None:
        args.caffemodel = args.caffemodel.resolve()

    args.local_dir.mkdir(parents=True, exist_ok=True)
    storage_uri = Path(args.local_dir).resolve().as_uri()

    env_config = build_env_config(args)

    ray.init(ignore_reinit_error=True)

    if args.algo == "dqn":
        trainable = tune.with_parameters(train_dqn_tune, base_env_config=env_config, args=args)
        search_space = dqn_search_space()
    else:
        trainable = tune.with_parameters(train_ppo_tune, base_env_config=env_config, args=args)
        search_space = ppo_search_space(args)

    def short_trial_name(trial) -> str:
        """Return a compact trial identifier to stay under Windows path limits."""
        return f"{trial.trial_id}"

    analysis = tune.run(
        trainable,
        config=search_space,
        metric=args.metric,
        mode=args.mode,
        num_samples=args.num_samples,
        resources_per_trial={"cpu": args.cpus_per_trial, "gpu": args.gpus_per_trial},
        storage_path=storage_uri,
        name=args.run_name,
        trial_name_creator=short_trial_name,
        trial_dirname_creator=short_trial_name,
    )

    try:
        best_result = analysis.best_result
    except AttributeError:
        best_trial = analysis.get_best_trial(metric=args.metric, mode=args.mode)
        best_result = getattr(best_trial, "metrics", getattr(best_trial, "last_result", {}))
        best_config = best_trial.config
    else:
        best_config = analysis.get_best_config(metric=args.metric, mode=args.mode)

    print("Best trial reward:", best_result.get(args.metric))
    print("Best config:", best_config)

    ray.shutdown()


if __name__ == "__main__":
    main()
