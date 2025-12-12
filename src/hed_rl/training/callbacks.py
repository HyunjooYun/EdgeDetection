"""Shared training callbacks."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, List

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter

from ..envs.hed_postprocess_env import HEDPostProcessEnv


class RolloutImageCallback(BaseCallback):
    """Log base/prediction/GT edge maps to TensorBoard at a fixed interval."""

    def __init__(
        self,
        eval_env_fn: Callable[[], HEDPostProcessEnv],
        image_names: Iterable[str],
        log_dir: Path,
        log_frequency: int,
    ) -> None:
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
