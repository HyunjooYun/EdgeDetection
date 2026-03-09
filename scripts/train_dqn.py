"""Train a DQN agent to optimize HED post-processing parameters."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Iterable, List

import numpy as np
import torch as th
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from torch.utils.tensorboard import SummaryWriter

from hed_rl import HEDPostProcessConfig, HEDPostProcessEnv, HedConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DQN on the HED post-processing environment")
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
    parser.add_argument(
        "--image-log-names",
        type=str,
        help="Comma-separated list of image filenames to use for rollout images (e.g. '12003.jpg,15004.jpg')",
    )
    parser.add_argument("--cycle-images", action="store_true", help="Iterate through dataset images without replacement")
    parser.add_argument("--learning-rate", type=float, help="DQN learning rate override")
    parser.add_argument("--gamma", type=float, help="Discount factor override")
    parser.add_argument("--batch-size", type=int, help="Batch size for gradient updates")
    parser.add_argument("--buffer-size", type=int, help="Replay buffer size override")
    parser.add_argument("--train-freq", type=int, help="How often to update the model (env steps)")
    parser.add_argument("--target-update-interval", type=int, help="Target network update interval")
    parser.add_argument("--gradient-steps", type=int, help="Number of gradient steps at each update")
    parser.add_argument("--exploration-fraction", type=float, help="Fraction of training spent exploring")
    parser.add_argument("--exploration-final-eps", type=float, help="Final epsilon for epsilon-greedy policy")
    parser.add_argument(
        "--eval-frequency",
        type=int,
        default=10000,
        help="Timesteps between evaluation rollouts for scalar metrics",
    )
    parser.add_argument(
        "--diag-frequency",
        type=int,
        default=5000,
        help="Timesteps between Q-value/TD-error diagnostics from the replay buffer",
    )
    parser.add_argument(
        "--diag-samples",
        type=int,
        default=1024,
        help="Number of samples from the replay buffer for diagnostics",
    )
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
        self._writer_failed = False

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
        if self._writer_failed or not self.image_names:
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
            try:
                self.writer.add_image(
                    f"rollout/{image_name}", canvas, self.num_timesteps, dataformats="HWC"
                )
            except Exception:
                self._writer_failed = True
                return
        if not self._writer_failed:
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


class EvalRewardCallback(BaseCallback):
    """Periodically evaluate the deterministic policy and log reward stats.

    This runs rollouts on a fixed set of images and logs scalar metrics such as
    mean/std/max episode reward to TensorBoard (e.g. ``eval/reward_mean``).
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


class QDiagnosticsCallback(BaseCallback):
    """Log Q-value and TD error statistics from the replay buffer.

    Periodically samples a minibatch from the replay buffer and computes
    summary statistics of the predicted Q-values and 1-step TD errors.
    These are logged to TensorBoard under ``diagnostics/*``.
    """

    def __init__(self, sample_size: int, log_frequency: int) -> None:
        super().__init__()
        self.sample_size = max(1, int(sample_size))
        self.log_frequency = max(1, int(log_frequency))
        self.last_log_step = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_log_step < self.log_frequency:
            return True

        self.last_log_step = self.num_timesteps

        model = self.model
        replay_buffer = getattr(model, "replay_buffer", None)
        if replay_buffer is None:
            return True

        # Some stable-baselines3 versions do not implement __len__ for ReplayBuffer.
        # Prefer the "buffer_size" attribute when available, otherwise fall back to len().
        buffer_len = getattr(replay_buffer, "buffer_size", None)
        if buffer_len is None:
            try:
                buffer_len = len(replay_buffer)  # type: ignore[arg-type]
            except TypeError:
                return True

        if buffer_len == 0:
            return True

        batch_size = min(self.sample_size, int(buffer_len))

        # Sample a batch similarly to the DQN.train() method
        replay_data = replay_buffer.sample(batch_size, env=model._vec_normalize_env)

        with th.no_grad():
            # Q-value statistics for current observations
            all_q_values = model.q_net(replay_data.observations)
            q_np = all_q_values.detach().cpu().numpy()
            self.logger.record("diagnostics/q_mean", float(q_np.mean()))
            self.logger.record("diagnostics/q_std", float(q_np.std()))
            self.logger.record("diagnostics/q_max", float(q_np.max()))
            self.logger.record("diagnostics/q_min", float(q_np.min()))

            # Compute 1-step TD targets, mirroring DQN.train()
            next_q_values = model.q_net_target(replay_data.next_observations)
            next_q_values, _ = next_q_values.max(dim=1)
            next_q_values = next_q_values.reshape(-1, 1)

            discounts = (
                replay_data.discounts if replay_data.discounts is not None else model.gamma
            )
            target_q_values = (
                replay_data.rewards
                + (1 - replay_data.dones) * discounts * next_q_values
            )

            current_q_values = model.q_net(replay_data.observations)
            current_q_values = th.gather(
                current_q_values, dim=1, index=replay_data.actions.long()
            )

            td_errors = (target_q_values - current_q_values).detach().cpu().numpy()
            self.logger.record("diagnostics/td_error_mean", float(td_errors.mean()))
            self.logger.record("diagnostics/td_error_std", float(td_errors.std()))
            self.logger.record(
                "diagnostics/td_error_abs_mean", float(np.abs(td_errors).mean())
            )

        return True


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

    # Training environment (typically using train images)
    def make_env() -> HEDPostProcessEnv:
        cfg = HEDPostProcessConfig(
            image_dir=args.image_dir,
            hed_config=hed_config,
            ground_truth_dir=args.ground_truth_dir,
            precomputed_edge_dir=args.edge_dir,
            max_steps=args.max_episode_steps,
            random_seed=args.seed,
            cache_edges=not args.no_cache_edges,
            cycle_images=args.cycle_images,
        )
        return HEDPostProcessEnv(cfg)

    # Evaluation/rollout environment (can be a separate val set)
    eval_image_dir = args.eval_image_dir or args.image_dir

    def make_eval_env() -> HEDPostProcessEnv:
        cfg = HEDPostProcessConfig(
            image_dir=eval_image_dir,
            hed_config=hed_config,
            ground_truth_dir=args.ground_truth_dir,
            precomputed_edge_dir=args.edge_dir,
            max_steps=args.max_episode_steps,
            random_seed=args.seed,
            cache_edges=not args.no_cache_edges,
            cycle_images=args.cycle_images,
        )
        return HEDPostProcessEnv(cfg)

    sample_env = make_env()
    sample_eval_env = make_eval_env()

    # Choose which images to use for rollout logging (from eval set).
    # If --image-log-names is provided, use that subset (if available);
    # otherwise fall back to the first image_log_count images.
    if args.image_log_names:
        requested = [name.strip() for name in str(args.image_log_names).split(",") if name.strip()]
        available = {path.name for path in sample_eval_env.image_paths}
        image_names = [name for name in requested if name in available]
        if not image_names:
            image_names = [path.name for path in sample_eval_env.image_paths[: args.image_log_count]]
    else:
        image_names = [path.name for path in sample_eval_env.image_paths[: args.image_log_count]]

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

    rollout_callback = RolloutImageCallback(
        eval_env_fn=make_eval_env,
        image_names=image_names,
        log_dir=args.tensorboard_log,
        log_frequency=args.image_log_frequency,
    )

    eval_callback = EvalRewardCallback(
        eval_env_fn=make_eval_env,
        image_names=image_names,
        eval_frequency=args.eval_frequency,
    )

    q_diag_callback = QDiagnosticsCallback(
        sample_size=args.diag_samples,
        log_frequency=args.diag_frequency,
    )

    callback = CallbackList([rollout_callback, eval_callback, q_diag_callback])

    model.learn(total_timesteps=args.timesteps, log_interval=100, callback=callback)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.output))
    vec_env.close()
    if hasattr(sample_env, "close"):
        sample_env.close()
    if hasattr(sample_eval_env, "close"):
        sample_eval_env.close()


if __name__ == "__main__":
    main()
