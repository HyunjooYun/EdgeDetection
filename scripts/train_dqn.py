"""Train a DQN agent to optimize HED post-processing parameters."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Iterable, List

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from torch.utils.tensorboard import SummaryWriter

from hed_rl import HEDPostProcessConfig, HEDPostProcessEnv, HedConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DQN on the HED post-processing environment")
    parser.add_argument("--prototxt", type=Path, help="Path to HED deploy.prototxt")
    parser.add_argument("--caffemodel", type=Path, help="Path to HED caffemodel")
    parser.add_argument("--image-dir", type=Path, default=Path("imgs/test"), help="Directory with training images")
    parser.add_argument("--ground-truth-dir", type=Path, help="Directory with ground-truth edge maps")
    parser.add_argument("--edge-dir", type=Path, help="Directory with precomputed base HED edge maps")
    parser.add_argument("--timesteps", type=int, default=50000, help="Total training timesteps")
    parser.add_argument(
        "--total-timesteps",
        type=int,
        help="Alias for --timesteps to keep backward compatibility",
    )
    parser.add_argument("--seed", type=int, default=13, help="Random seed")
    parser.add_argument("--n-envs", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("--max-episode-steps", type=int, default=30, help="Maximum steps per episode")
    parser.add_argument("--tensorboard-log", type=Path, default=Path("runs/dqn"), help="TensorBoard log directory")
    parser.add_argument("--output", type=Path, default=Path("artifacts/dqn_hed"), help="Path prefix for the saved model")
    parser.add_argument("--hed-width", type=int, default=0, help="Optional input width override for HED")
    parser.add_argument("--hed-height", type=int, default=0, help="Optional input height override for HED")
    parser.add_argument("--no-cache-edges", action="store_true", help="Disable edge caching inside the environment")
    parser.add_argument("--image-log-frequency", type=int, default=5000, help="Timesteps between logging rollout images")
    parser.add_argument("--image-log-count", type=int, default=3, help="Number of dataset images to log as rollouts")
    parser.add_argument("--learning-rate", type=float, help="DQN learning rate override")
    parser.add_argument("--gamma", type=float, help="Discount factor override")
    parser.add_argument("--batch-size", type=int, help="Batch size for gradient updates")
    parser.add_argument("--buffer-size", type=int, help="Replay buffer size override")
    parser.add_argument("--train-freq", type=int, help="How often to update the model (env steps)")
    parser.add_argument("--target-update-interval", type=int, help="Target network update interval")
    parser.add_argument("--gradient-steps", type=int, help="Number of gradient steps at each update")
    parser.add_argument("--exploration-fraction", type=float, help="Fraction of training spent exploring")
    parser.add_argument("--exploration-final-eps", type=float, help="Final epsilon for epsilon-greedy policy")
    return parser.parse_args()


class RolloutImageCallback(BaseCallback):
    """Log HED rollout comparisons (base, prediction, GT) to TensorBoard."""

    def __init__(
        self,
        eval_env_fn: Callable[[], HEDPostProcessEnv],
        image_names: Iterable[str],
        log_dir: Path,
        log_frequency: int,
    ):
        super().__init__()
        self.eval_env = eval_env_fn()
        self.image_names: List[str] = list(image_names)
        self.log_frequency = max(1, log_frequency)
        self.last_logged_step = 0
        log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(log_dir / "rollout_images"))

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_logged_step >= self.log_frequency:
            self._log_images()
            self.last_logged_step = self.num_timesteps
        return True

    def _on_training_end(self) -> None:
        # Always log a final snapshot for comparison
        self._log_images()
        if hasattr(self.eval_env, "close"):
            self.eval_env.close()
        self.writer.close()

    def _log_images(self) -> None:
        if not self.image_names:
            return
        for image_name in self.image_names:
            obs, _ = self.eval_env.reset(options={"image_name": image_name})
            terminated = truncated = False
            steps = 0
            while not (terminated or truncated):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                steps += 1
                if steps >= self.eval_env.config.max_steps:
                    break
            base_edge = self.eval_env._get_base_edge(self.eval_env.current_image)
            self.eval_env._ensure_ground_truth(self.eval_env.current_image)
            gt_edge = self.eval_env._ground_truth_edges[self.eval_env.current_image.name]
            pred_edge = self.eval_env._apply_postprocessing(base_edge, self.eval_env.current_params)
            canvas = self._compose_canvas(base_edge, pred_edge, gt_edge)
            self.writer.add_image(
                f"rollout/{image_name}", canvas, self.num_timesteps, dataformats="HWC"
            )
        self.writer.flush()

    @staticmethod
    def _compose_canvas(base_edge: np.ndarray, pred_edge: np.ndarray, gt_edge: np.ndarray) -> np.ndarray:
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
    if args.total_timesteps is not None:
        args.timesteps = args.total_timesteps
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

    sample_env = make_env()
    image_names = [path.name for path in sample_env.image_paths[: args.image_log_count]]

    vec_env = make_vec_env(make_env, n_envs=args.n_envs, seed=args.seed)

    dqn_kwargs = {
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "batch_size": args.batch_size,
        "buffer_size": args.buffer_size,
        "train_freq": args.train_freq,
        "target_update_interval": args.target_update_interval,
        "gradient_steps": args.gradient_steps,
        "exploration_fraction": args.exploration_fraction,
        "exploration_final_eps": args.exploration_final_eps,
    }

    default_kwargs = {
        "learning_rate": 1e-3,
        "buffer_size": 50_000,
        "batch_size": 64,
        "train_freq": 4,
        "target_update_interval": 1_000,
        "gradient_steps": 1,
        "exploration_fraction": 0.2,
        "exploration_final_eps": 0.05,
        "gamma": 0.99,
    }

    for key, default in default_kwargs.items():
        if dqn_kwargs[key] is None:
            dqn_kwargs[key] = default

    model = DQN(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=str(args.tensorboard_log),
        **dqn_kwargs,
        seed=args.seed,
    )

    callback = RolloutImageCallback(
        eval_env_fn=make_env,
        image_names=image_names,
        log_dir=args.tensorboard_log,
        log_frequency=args.image_log_frequency,
    )

    model.learn(total_timesteps=args.timesteps, log_interval=100, callback=callback)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.output))
    vec_env.close()
    if hasattr(sample_env, "close"):
        sample_env.close()


if __name__ == "__main__":
    main()
