"""Quick smoke-test for the HED post-processing RL environment."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from hed_rl import HEDPostProcessConfig, HEDPostProcessEnv, HedConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate random actions in the HED RL environment")
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("imgs/test"),
        help="Directory containing sample images",
    )
    parser.add_argument("--steps", type=int, default=10, help="Number of random steps to execute")
    parser.add_argument("--seed", type=int, default=7, help="Random seed to ensure reproducibility")
    parser.add_argument("--hed-prototxt", type=Path, help="Path to HED deploy.prototxt (optional)")
    parser.add_argument("--hed-caffemodel", type=Path, help="Path to HED caffemodel (optional)")
    parser.add_argument("--hed-width", type=int, default=0, help="Optional width for HED input size")
    parser.add_argument("--hed-height", type=int, default=0, help="Optional height for HED input size")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hed_config = None
    if args.hed_prototxt and args.hed_caffemodel:
        input_size = None
        if args.hed_width > 0 and args.hed_height > 0:
            input_size = (args.hed_width, args.hed_height)
        hed_config = HedConfig(
            prototxt_path=args.hed_prototxt,
            caffemodel_path=args.hed_caffemodel,
            input_size=input_size,
        )

    config = HEDPostProcessConfig(image_dir=args.image_dir, random_seed=args.seed, hed_config=hed_config)
    env = HEDPostProcessEnv(config)

    observation, info = env.reset()
    print(f"Initial image: {info['image_id']}")
    print(f"Initial observation: {observation}")

    total_reward = 0.0
    for step_idx in range(args.steps):
        action = env.sample_action()
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(
            f"step={step_idx:02d} action={action} reward={reward:.3f} terminated={terminated} truncated={truncated}"
        )
        if terminated or truncated:
            break

    print(f"Total reward: {total_reward:.3f}")


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    main()
