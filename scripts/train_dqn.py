"""Train a DQN agent to optimize HED post-processing parameters."""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env

from hed_rl import HEDPostProcessConfig, HEDPostProcessEnv, HedConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DQN on the HED post-processing environment")
    parser.add_argument("--prototxt", type=Path, help="Path to HED deploy.prototxt")
    parser.add_argument("--caffemodel", type=Path, help="Path to HED caffemodel")
    parser.add_argument("--image-dir", type=Path, default=Path("imgs/test"), help="Directory with training images")
    parser.add_argument("--ground-truth-dir", type=Path, help="Directory with ground-truth edge maps")
    parser.add_argument("--edge-dir", type=Path, help="Directory with precomputed base HED edge maps")
    parser.add_argument("--timesteps", type=int, default=50000, help="Total training timesteps")
    parser.add_argument("--seed", type=int, default=13, help="Random seed")
    parser.add_argument("--n-envs", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("--max-episode-steps", type=int, default=30, help="Maximum steps per episode")
    parser.add_argument("--tensorboard-log", type=Path, default=Path("runs/dqn"), help="TensorBoard log directory")
    parser.add_argument("--output", type=Path, default=Path("artifacts/dqn_hed"), help="Path prefix for the saved model")
    parser.add_argument("--hed-width", type=int, default=0, help="Optional input width override for HED")
    parser.add_argument("--hed-height", type=int, default=0, help="Optional input height override for HED")
    parser.add_argument("--no-cache-edges", action="store_true", help="Disable edge caching inside the environment")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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

    def make_env() -> HEDPostProcessEnv:
        cfg = HEDPostProcessConfig(
            image_dir=args.image_dir,
            hed_config=hed_config,
            ground_truth_dir=args.ground_truth_dir,
            precomputed_edge_dir=args.edge_dir,
            max_steps=args.max_episode_steps,
            cache_edges=not args.no_cache_edges,
        )
        return HEDPostProcessEnv(cfg)

    vec_env = make_vec_env(make_env, n_envs=args.n_envs, seed=args.seed)

    model = DQN(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=str(args.tensorboard_log),
        learning_rate=1e-3,
        buffer_size=50000,
        batch_size=64,
        train_freq=4,
        target_update_interval=1000,
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
        seed=args.seed,
    )

    model.learn(total_timesteps=args.timesteps, log_interval=100)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.output))
    vec_env.close()


if __name__ == "__main__":
    main()
