"""Train PPO agent for HED post-processing."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Callable, Iterable, List

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env

from hed_rl import HEDPostProcessConfig, HEDPostProcessEnv, HedConfig
from hed_rl.training.callbacks import RolloutImageCallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prototxt", type=Path, help="Path to HED deploy.prototxt")
    parser.add_argument("--caffemodel", type=Path, help="Path to HED caffemodel")
    parser.add_argument("--image-dir", type=Path, default=Path("imgs/test"), help="Directory with training images")
    parser.add_argument(
        "--eval-image-dir",
        type=Path,
        help="Directory with images used for evaluation rollouts (defaults to --image-dir)",
    )
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
    parser.add_argument(
        "--image-log-names",
        type=str,
        help="Comma-separated list of image filenames to use for rollout images (e.g. '12003.jpg,15004.jpg')",
    )
    parser.add_argument("--cycle-images", action="store_true", help="Iterate through dataset images without replacement")
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
        random_seed=args.seed,
        cache_edges=not args.no_cache_edges,
        cycle_images=args.cycle_images,
    )

    def make_env() -> HEDPostProcessEnv:
        return HEDPostProcessEnv(env_config)

    return make_env


def collect_image_names(sample_env: HEDPostProcessEnv, count: int) -> List[str]:
    image_paths: Iterable[Path] = sample_env.image_paths
    return [path.name for path in image_paths[:count]]


class EvalRewardCallback(BaseCallback):
    """Periodically evaluate the deterministic policy and log reward stats.

    Mirrors the DQN EvalRewardCallback so PPO also logs
    ``eval/reward_mean``, ``eval/reward_std``, ``eval/reward_max`` on the
    evaluation image set (e.g., val).
    """

    def __init__(
        self,
        eval_env_fn: Callable[[], HEDPostProcessEnv],
        image_names: Iterable[str],
        eval_frequency: int,
    ) -> None:
        super().__init__()
        self.eval_env = eval_env_fn()
        self.image_names: List[str] = list(image_names)
        self.eval_frequency = max(1, int(eval_frequency))
        self.last_eval_step = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_eval_step < self.eval_frequency:
            return True

        self.last_eval_step = self.num_timesteps
        if not self.image_names:
            return True

        rewards: List[float] = []
        for image_name in self.image_names:
            obs, _ = self.eval_env.reset(options={"image_name": image_name})
            terminated = False
            truncated = False
            last_reward = 0.0
            steps = 0
            while not (terminated or truncated):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _info = self.eval_env.step(action)
                last_reward = float(reward)
                steps += 1
                if steps >= self.eval_env.config.max_steps:
                    break
            rewards.append(last_reward)

        if rewards:
            rewards_array = np.asarray(rewards, dtype=np.float32)
            self.logger.record("eval/reward_mean", float(rewards_array.mean()))
            self.logger.record("eval/reward_std", float(rewards_array.std()))
            self.logger.record("eval/reward_max", float(rewards_array.max()))

        return True

    def _on_training_end(self) -> None:
        if hasattr(self.eval_env, "close"):
            self.eval_env.close()


def main() -> None:
    args = parse_args()

    if args.total_timesteps is not None:
        args.timesteps = args.total_timesteps

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Training environment factory (typically using train images)
    make_env = build_env_factory(args)
    sample_env = make_env()

    # Evaluation/rollout environment factory (can be a separate val set)
    eval_image_dir = args.eval_image_dir or args.image_dir
    if eval_image_dir == args.image_dir:
        make_eval_env = make_env
        sample_eval_env = sample_env
    else:
        eval_args = argparse.Namespace(**vars(args))
        eval_args.image_dir = eval_image_dir
        make_eval_env = build_env_factory(eval_args)
        sample_eval_env = make_eval_env()

    # Choose which images to use for rollout logging (from eval set).
    # If --image-log-names is provided, use that subset (if available);
    # otherwise fall back to the first image_log_count images.
    if args.image_log_names:
        requested = [name.strip() for name in str(args.image_log_names).split(",") if name.strip()]
        available = {path.name for path in sample_eval_env.image_paths}
        image_names = [name for name in requested if name in available]
        if not image_names:
            image_names = collect_image_names(sample_eval_env, args.image_log_count)
    else:
        image_names = collect_image_names(sample_eval_env, args.image_log_count)

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

    rollout_callback = RolloutImageCallback(
        eval_env_fn=make_eval_env,
        image_names=image_names,
        log_dir=args.tensorboard_log,
        log_frequency=args.image_log_frequency,
    )

    # Reuse DQN-style eval frequency default for PPO: 10k steps
    eval_callback = EvalRewardCallback(
        eval_env_fn=make_eval_env,
        image_names=image_names,
        eval_frequency=10_000,
    )

    callback = CallbackList([rollout_callback, eval_callback])

    model.learn(total_timesteps=args.timesteps, callback=callback)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.output))

    vec_env.close()
    if hasattr(sample_env, "close"):
        sample_env.close()
    if sample_eval_env is not sample_env and hasattr(sample_eval_env, "close"):
        sample_eval_env.close()


if __name__ == "__main__":
    main()
