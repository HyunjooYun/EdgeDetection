"""Train PPO agent for HED post-processing."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from hed_rl import HEDPostProcessConfig, HEDPostProcessEnv, HedConfig
from hed_rl.training.callbacks import RolloutImageCallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prototxt", type=Path, help="Path to HED deploy.prototxt")
    parser.add_argument("--caffemodel", type=Path, help="Path to HED caffemodel")
    parser.add_argument("--image-dir", type=Path, default=Path("imgs/test"), help="Directory with training images")
    parser.add_argument("--ground-truth-dir", type=Path, help="Directory with ground-truth edge maps")
    parser.add_argument("--edge-dir", type=Path, help="Directory with precomputed base HED edge maps")
    parser.add_argument("--timesteps", type=int, default=200_000, help="Total training timesteps")
    parser.add_argument(
        "--total-timesteps",
        type=int,
        help="Alias for --timesteps to keep backward compatibility",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n-envs", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("--max-episode-steps", type=int, default=30, help="Maximum steps per episode")
    parser.add_argument("--tensorboard-log", type=Path, default=Path("runs/ppo_train"), help="TensorBoard log directory")
    parser.add_argument("--output", type=Path, default=Path("artifacts/ppo_train"), help="Path prefix for the saved model")
    parser.add_argument("--hed-width", type=int, default=0, help="Optional input width override for HED")
    parser.add_argument("--hed-height", type=int, default=0, help="Optional input height override for HED")
    parser.add_argument("--no-cache-edges", action="store_true", help="Disable edge caching inside the environment")
    parser.add_argument("--image-log-frequency", type=int, default=50_000, help="Timesteps between logging rollout images")
    parser.add_argument("--image-log-count", type=int, default=3, help="Number of dataset images to log as rollouts")
    parser.add_argument("--learning-rate", type=float, help="PPO learning rate override")
    parser.add_argument("--gamma", type=float, help="PPO discount factor override")
    parser.add_argument("--n-steps", type=int, help="Number of steps to run for each environment per update")
    parser.add_argument("--batch-size", type=int, help="Minibatch size for each gradient update")
    parser.add_argument("--gae-lambda", type=float, help="GAE lambda override")
    parser.add_argument("--clip-range", type=float, help="PPO clip range override")
    parser.add_argument("--ent-coef", type=float, help="Entropy coefficient override")
    parser.add_argument("--vf-coef", type=float, help="Value function loss coefficient override")
    return parser.parse_args()


def build_env_factory(args: argparse.Namespace):
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

    env_config = HEDPostProcessConfig(
        image_dir=args.image_dir,
        hed_config=hed_config,
        ground_truth_dir=args.ground_truth_dir,
        precomputed_edge_dir=args.edge_dir,
        max_steps=args.max_episode_steps,
        cache_edges=not args.no_cache_edges,
    )

    def make_env() -> HEDPostProcessEnv:
        return HEDPostProcessEnv(env_config)

    return make_env


def collect_image_names(sample_env: HEDPostProcessEnv, count: int) -> List[str]:
    image_paths: Iterable[Path] = sample_env.image_paths
    return [path.name for path in image_paths[:count]]


def main() -> None:
    args = parse_args()

    if args.total_timesteps is not None:
        args.timesteps = args.total_timesteps

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    make_env = build_env_factory(args)
    sample_env = make_env()
    image_names = collect_image_names(sample_env, args.image_log_count)

    vec_env = make_vec_env(make_env, n_envs=args.n_envs, seed=args.seed)

    ppo_kwargs = {
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "gae_lambda": args.gae_lambda,
        "clip_range": args.clip_range,
        "ent_coef": args.ent_coef,
        "vf_coef": args.vf_coef,
    }

    if ppo_kwargs["batch_size"] is not None and ppo_kwargs["n_steps"] is not None:
        rollout = ppo_kwargs["n_steps"] * args.n_envs
        if ppo_kwargs["batch_size"] > rollout:
            raise ValueError("batch_size cannot exceed n_steps * n_envs for PPO")

    # Drop keys that were not overridden to preserve SB3 defaults.
    ppo_kwargs = {key: value for key, value in ppo_kwargs.items() if value is not None}

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        tensorboard_log=str(args.tensorboard_log),
        seed=args.seed,
        **ppo_kwargs,
    )

    callback = RolloutImageCallback(
        eval_env_fn=make_env,
        image_names=image_names,
        log_dir=args.tensorboard_log,
        log_frequency=args.image_log_frequency,
    )

    model.learn(total_timesteps=args.timesteps, callback=callback)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.output))

    vec_env.close()
    if hasattr(sample_env, "close"):
        sample_env.close()


if __name__ == "__main__":
    main()
