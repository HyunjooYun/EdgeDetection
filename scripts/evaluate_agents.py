"""Evaluate trained PPO and DQN agents on the HED post-processing environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from torch.utils.tensorboard import SummaryWriter

from hed_rl import HEDPostProcessConfig, HEDPostProcessEnv, HedConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PPO and DQN agents on the HED post-processing task")
    parser.add_argument("--dqn-model", type=Path, help="Path to the trained DQN model zip file")
    parser.add_argument("--ppo-model", type=Path, help="Path to the trained PPO model zip file")
    parser.add_argument("--prototxt", type=Path, help="Path to HED deploy.prototxt")
    parser.add_argument("--caffemodel", type=Path, help="Path to HED caffemodel")
    parser.add_argument("--image-dir", type=Path, default=Path("inputs/train"), help="Directory with evaluation images")
    parser.add_argument("--ground-truth-dir", type=Path, help="Directory with ground-truth edge maps")
    parser.add_argument("--edge-dir", type=Path, help="Directory with precomputed base HED edge maps")
    parser.add_argument("--episodes", type=int, default=20, help="Number of evaluation episodes per model")
    parser.add_argument("--seed", type=int, default=42, help="Evaluation seed for environment")
    parser.add_argument("--max-episode-steps", type=int, default=30, help="Maximum steps per evaluation episode")
    parser.add_argument("--hed-width", type=int, default=0, help="Optional HED input width override")
    parser.add_argument("--hed-height", type=int, default=0, help="Optional HED input height override")
    parser.add_argument("--no-cache-edges", action="store_true", help="Disable edge caching inside the environment")
    parser.add_argument("--tensorboard-log", type=Path, help="Directory for TensorBoard evaluation logs")
    parser.add_argument("--image-log-count", type=int, default=3, help="How many rollout images to log per model")
    parser.add_argument("--output-json", type=Path, help="Optional path to save evaluation metrics as JSON")
    return parser.parse_args()


def prepare_env_factory(args: argparse.Namespace):
    input_size: Optional[tuple[int, int]] = None
    if args.hed_width > 0 and args.hed_height > 0:
        input_size = (args.hed_width, args.hed_height)

    hed_config: Optional[HedConfig] = None
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
            random_seed=args.seed,
            cache_edges=not args.no_cache_edges,
        )
        return HEDPostProcessEnv(cfg)

    sample_env = make_env()
    image_names = [path.name for path in sample_env.image_paths[: max(0, args.image_log_count)]]
    if hasattr(sample_env, "close"):
        sample_env.close()
    return make_env, image_names


def evaluate_model(
    name: str,
    model_path: Path,
    model_factory,
    env_factory,
    n_episodes: int,
    writer: Optional[SummaryWriter],
    image_names: Iterable[str],
    log_step: int,
) -> Dict[str, object]:
    env = Monitor(env_factory())
    try:
        model = model_factory.load(str(model_path), env=env)
        episode_rewards, episode_lengths = evaluate_policy(
            model,
            env,
            n_eval_episodes=n_episodes,
            deterministic=True,
            return_episode_rewards=True,
        )
        rewards_array = np.array(episode_rewards, dtype=np.float32)
        lengths_array = np.array(episode_lengths, dtype=np.float32)
        summary = {
            "model": name,
            "model_path": str(model_path),
            "episodes": len(episode_rewards),
            "mean_reward": float(rewards_array.mean()),
            "std_reward": float(rewards_array.std(ddof=0)),
            "min_reward": float(rewards_array.min()),
            "max_reward": float(rewards_array.max()),
            "mean_episode_length": float(lengths_array.mean()),
            "std_episode_length": float(lengths_array.std(ddof=0)),
        }
        if writer is not None:
            writer.add_scalar(f"{name}/mean_reward", summary["mean_reward"], log_step)
            writer.add_scalar(f"{name}/mean_episode_length", summary["mean_episode_length"], log_step)
            writer.add_scalar(f"{name}/reward_std", summary["std_reward"], log_step)
            log_rollout_images(model, env_factory, image_names, writer, name, log_step)
            writer.flush()
        return summary
    finally:
        if hasattr(env, "close"):
            env.close()


def log_rollout_images(
    model,
    env_factory,
    image_names: Iterable[str],
    writer: SummaryWriter,
    tag_prefix: str,
    step: int,
) -> None:
    names: List[str] = list(image_names)
    if not names:
        return
    env = env_factory()
    try:
        for image_name in names:
            obs, _ = env.reset(options={"image_name": image_name})
            terminated = False
            truncated = False
            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = env.step(action)
            if env.current_image is None:
                continue
            base_edge = env._get_base_edge(env.current_image)
            env._ensure_ground_truth(env.current_image)
            gt_edge = env._ground_truth_edges[env.current_image.name]
            pred_edge = env._apply_postprocessing(base_edge, env.current_params)
            canvas = compose_canvas(base_edge, pred_edge, gt_edge)
            writer.add_image(f"{tag_prefix}/rollout/{image_name}", canvas, step, dataformats="HWC")
        writer.flush()
    finally:
        if hasattr(env, "close"):
            env.close()


def compose_canvas(base_edge: np.ndarray, pred_edge: np.ndarray, gt_edge: np.ndarray) -> np.ndarray:
    def to_rgb(image: np.ndarray) -> np.ndarray:
        clipped = np.clip(image, 0.0, 1.0)
        if clipped.ndim == 2:
            clipped = clipped[:, :, None]
        if clipped.shape[2] == 1:
            clipped = np.repeat(clipped, 3, axis=2)
        return (clipped * 255).astype(np.uint8)

    base_rgb = to_rgb(base_edge)
    pred_rgb = to_rgb(pred_edge)
    gt_rgb = to_rgb(gt_edge)
    return np.concatenate([base_rgb, pred_rgb, gt_rgb], axis=1)


def main() -> None:
    args = parse_args()
    env_factory, image_names = prepare_env_factory(args)

    writer: Optional[SummaryWriter] = None
    if args.tensorboard_log:
        args.tensorboard_log.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(str(args.tensorboard_log))

    results: List[Dict[str, object]] = []
    log_step = 0

    if args.dqn_model and args.dqn_model.exists():
        summary = evaluate_model(
            name="dqn",
            model_path=args.dqn_model,
            model_factory=DQN,
            env_factory=env_factory,
            n_episodes=args.episodes,
            writer=writer,
            image_names=image_names,
            log_step=log_step,
        )
        results.append(summary)
        log_step += 1
        print(
            f"DQN mean_reward={summary['mean_reward']:.3f} std={summary['std_reward']:.3f} "
            f"mean_len={summary['mean_episode_length']:.2f}"
        )

    if args.ppo_model and args.ppo_model.exists():
        summary = evaluate_model(
            name="ppo",
            model_path=args.ppo_model,
            model_factory=PPO,
            env_factory=env_factory,
            n_episodes=args.episodes,
            writer=writer,
            image_names=image_names,
            log_step=log_step,
        )
        results.append(summary)
        log_step += 1
        print(
            f"PPO mean_reward={summary['mean_reward']:.3f} std={summary['std_reward']:.3f} "
            f"mean_len={summary['mean_episode_length']:.2f}"
        )

    if writer is not None:
        writer.close()

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as stream:
            json.dump({"results": results}, stream, indent=2)

    if not results:
        print("No models were evaluated.")


if __name__ == "__main__":
    main()
