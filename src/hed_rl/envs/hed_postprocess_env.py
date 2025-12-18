"""Custom reinforcement learning environment for HED post-processing parameter tuning."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - optional dependency
    gym = None  # type: ignore[assignment]
    spaces = None  # type: ignore[assignment]

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - required at runtime
    raise RuntimeError("numpy is required to use HEDPostProcessEnv") from exc

import cv2

from ..pipeline.hed_inference import HEDModel, HedConfig


@dataclass(frozen=True)
class ParameterSpec:
    """Configuration for a tunable post-processing parameter."""

    name: str
    minimum: float
    maximum: float
    step: float
    initial: float

    def clip(self, value: float) -> float:
        return min(self.maximum, max(self.minimum, value))

    def normalize(self, value: float) -> float:
        span = self.maximum - self.minimum
        if span == 0:  # pragma: no cover - defensive guard
            return 0.0
        return (value - self.minimum) / span


@dataclass
class HEDPostProcessConfig:
    """Runtime configuration for :class:`HEDPostProcessEnv`."""

    image_dir: Path
    hed_config: Optional[HedConfig] = None
    ground_truth_dir: Optional[Path] = None
    precomputed_edge_dir: Optional[Path] = None
    parameter_specs: Sequence[ParameterSpec] = field(
        default_factory=lambda: (
            ParameterSpec("threshold", 0.05, 1.0, 0.05, 0.45),
            ParameterSpec("blur_sigma", 0.2, 3.0, 0.2, 1.2),
            ParameterSpec("nms_strength", 0.2, 1.0, 0.1, 0.6),
            ParameterSpec("morphology_radius", 1.0, 7.0, 1.0, 3.0),
        )
    )
    max_steps: int = 30
    reward_tolerance: float = 0.05
    random_seed: Optional[int] = None
    target_parameter_map: Optional[Dict[str, Dict[str, float]]] = None
    cache_edges: bool = True
    cycle_images: bool = False

    def resolve_targets(self, image_ids: Iterable[str]) -> Dict[str, Dict[str, float]]:
        if self.target_parameter_map is not None:
            return self.target_parameter_map
        defaults: Dict[str, Dict[str, float]] = {}
        for image_id in image_ids:
            defaults[image_id] = {
                spec.name: random.uniform(spec.minimum, spec.maximum) for spec in self.parameter_specs
            }
        return defaults


class HEDPostProcessEnv(gym.Env if gym else object):  # type: ignore[misc]
    """Environment that simulates HED 후처리 파라메터 최적화."""

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, config: HEDPostProcessConfig):
        if not config.image_dir.exists():
            raise FileNotFoundError(f"image directory not found: {config.image_dir}")

        self.config = config
        self.random = random.Random(config.random_seed)
        self.parameter_specs: Tuple[ParameterSpec, ...] = tuple(config.parameter_specs)
        self.image_paths: List[Path] = self._collect_images(config.image_dir)
        if not self.image_paths:
            raise ValueError(f"no images found in {config.image_dir}")
        self._image_lookup = {path.name: path for path in self.image_paths}

        if config.ground_truth_dir is not None and not config.ground_truth_dir.exists():
            raise FileNotFoundError(f"ground truth directory not found: {config.ground_truth_dir}")
        if config.precomputed_edge_dir is not None and not config.precomputed_edge_dir.exists():
            raise FileNotFoundError(f"precomputed edge directory not found: {config.precomputed_edge_dir}")

        self.hed_model: Optional[HEDModel] = None
        if config.hed_config is not None:
            self.hed_model = HEDModel(config.hed_config)

        self.target_parameters = config.resolve_targets(path.name for path in self.image_paths)
        self._action_table: List[Tuple[int, float]] = self._build_action_table()
        self.current_image: Optional[Path] = None
        self.current_params: Dict[str, float] = {}
        self.base_stats: Dict[str, float] = {}
        self.steps_taken: int = 0
        self._base_edge_cache: Dict[str, np.ndarray] = {}
        self._ground_truth_edges: Dict[str, np.ndarray] = {}
        self.cycle_images = bool(config.cycle_images)
        self._cycle_indices: List[int] = list(range(len(self.image_paths)))
        self._cycle_position: int = 0
        if self.cycle_images:
            self.random.shuffle(self._cycle_indices)

        obs_dim = 4 + len(self.parameter_specs) * 2
        if spaces:
            low = np.zeros(obs_dim, dtype=np.float32)
            high = np.ones(obs_dim, dtype=np.float32)
            self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
            self.action_space = spaces.Discrete(len(self._action_table))
        else:  # pragma: no cover - gymnasium optional
            self.observation_space = None
            self.action_space = None

    @staticmethod
    def _collect_images(image_dir: Path) -> List[Path]:
        supported = {".png", ".jpg", ".jpeg", ".bmp"}
        return sorted(
            path
            for path in image_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in supported
        )

    def _build_action_table(self) -> List[Tuple[int, float]]:
        actions: List[Tuple[int, float]] = []
        for idx, spec in enumerate(self.parameter_specs):
            actions.append((idx, spec.step))
            actions.append((idx, -spec.step))
        return actions

    def _load_image_stats(self, image_path: Path) -> Dict[str, float]:
        """Generate lightweight image statistics for the observation vector."""
        file_size = image_path.stat().st_size
        normalized_size = min(1.0, math.log10(file_size + 1) / 7.0)
        checksum = sum(image_path.read_bytes()) % 10_000
        normalized_checksum = checksum / 10_000
        edge_density = 0.0
        if self.hed_model is not None:
            base_edges = self._get_base_edge(image_path)
            edge_density = float(np.mean(base_edges > 0.5))
        return {
            "size_hint": normalized_size,
            "texture_hint": normalized_checksum,
            "edge_density": edge_density,
        }

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, object]] = None):
        if seed is not None:
            self.random.seed(seed)

        self.steps_taken = 0
        forced_image = None
        if options:
            if "image_path" in options and options["image_path"] is not None:
                candidate = Path(str(options["image_path"]))
                if not candidate.exists():  # pragma: no cover - defensive guard
                    raise FileNotFoundError(f"forced image path not found: {candidate}")
                forced_image = candidate
            elif "image_name" in options and options["image_name"] is not None:
                lookup_key = Path(str(options["image_name"]))
                forced_image = self._image_lookup.get(lookup_key.name)
                if forced_image is None:  # pragma: no cover - defensive guard
                    raise KeyError(f"image name '{options['image_name']}' not found in dataset")
        if forced_image is not None:
            self.current_image = forced_image
        elif self.cycle_images:
            if self._cycle_position >= len(self._cycle_indices):
                self._cycle_indices = list(range(len(self.image_paths)))
                self.random.shuffle(self._cycle_indices)
                self._cycle_position = 0
            index = self._cycle_indices[self._cycle_position]
            self._cycle_position += 1
            self.current_image = self.image_paths[index]
        else:
            self.current_image = self.random.choice(self.image_paths)
        self.current_params = {spec.name: spec.initial for spec in self.parameter_specs}
        if self.hed_model is not None and self.current_image is not None:
            self._ensure_ground_truth(self.current_image)
        self.base_stats = self._load_image_stats(self.current_image)
        observation = self._encode_observation()
        info = {"image_id": self.current_image.name}
        return observation, info

    def step(self, action: int):
        if not (0 <= action < len(self._action_table)):
            raise ValueError(f"invalid action index: {action}")
        param_index, delta = self._action_table[action]
        spec = self.parameter_specs[param_index]
        new_value = spec.clip(self.current_params[spec.name] + delta)
        self.current_params[spec.name] = new_value
        self.steps_taken += 1

        observation = self._encode_observation()
        reward = self._calculate_reward()
        terminated = reward >= 1.0 - self.config.reward_tolerance
        truncated = self.steps_taken >= self.config.max_steps
        info = {
            "image_id": self.current_image.name if self.current_image else None,
            "params": dict(self.current_params),
            "target": self._target_params_for_current_image(),
        }
        if self.hed_model is not None and self.current_image is not None:
            info["reward_metric"] = "f1"
        return observation, reward, terminated, truncated, info

    def render(self):  # pragma: no cover - simple console render
        if self.current_image is None:
            print("Environment not reset.")
            return
        reward = self._calculate_reward()
        print(
            f"Image: {self.current_image.name}\n"
            f"Params: {self.current_params}\n"
            f"Reward: {reward:.3f}\n"
            f"Step: {self.steps_taken}/{self.config.max_steps}"
        )

    def sample_action(self) -> int:
        """Sample a valid random action regardless of gymnasium availability."""
        return self.random.randrange(len(self._action_table))

    def _target_params_for_current_image(self) -> Dict[str, float]:
        assert self.current_image is not None
        return self.target_parameters[self.current_image.name]

    def _calculate_reward(self) -> float:
        if self.hed_model is not None and self.current_image is not None:
            return self._calculate_reward_with_edges(self.current_image, self.current_params)
        target = self._target_params_for_current_image()
        distance = 0.0
        norm_factor = 0.0
        for spec in self.parameter_specs:
            ideal = target.get(spec.name, spec.initial)
            current = self.current_params.get(spec.name, spec.initial)
            span = spec.maximum - spec.minimum
            if span == 0:
                continue
            norm_factor += 1.0
            distance += abs((current - ideal) / span)
        if norm_factor == 0:  # pragma: no cover - defensive guard
            return 0.0
        normalized_distance = distance / norm_factor
        return max(0.0, 1.0 - normalized_distance)

    def _encode_observation(self) -> np.ndarray:
        params_normalized = [
            spec.normalize(self.current_params[spec.name]) for spec in self.parameter_specs
        ]
        targets = self._target_params_for_current_image()
        deltas = [
            abs(spec.normalize(self.current_params[spec.name]) - spec.normalize(targets.get(spec.name, spec.initial)))
            for spec in self.parameter_specs
        ]
        progress = self.steps_taken / max(1, self.config.max_steps)
        edge_density = self.base_stats.get("edge_density", 0.0)
        features = [self.base_stats["size_hint"], self.base_stats["texture_hint"], edge_density, progress]
        obs = np.array(features + params_normalized + deltas, dtype=np.float32)
        if spaces and self.observation_space is not None:
            obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        return obs

    def _get_base_edge(self, image_path: Path) -> np.ndarray:
        cache_key = image_path.name
        if cache_key in self._base_edge_cache:
            return self._base_edge_cache[cache_key]
        base_edge: Optional[np.ndarray] = None
        if self.config.precomputed_edge_dir is not None:
            base_edge = self._load_precomputed_edge(image_path)
        if base_edge is None:
            if self.hed_model is None:
                raise RuntimeError("HED model not configured; cannot compute edge map")
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"unable to load image at {image_path}")
            base_edge = self.hed_model.infer(image)
            if base_edge is None:
                raise ValueError(f"failed to compute edge map for {image_path}")
        if self.config.cache_edges:
            self._base_edge_cache[cache_key] = base_edge
        return base_edge

    def _ensure_ground_truth(self, image_path: Path) -> None:
        cache_key = image_path.name
        if cache_key in self._ground_truth_edges:
            return
        target_edge: Optional[np.ndarray] = None
        if self.config.ground_truth_dir is not None:
            target_edge = self._load_ground_truth_edge(image_path)
        if target_edge is None:
            target_params = self.target_parameters.get(cache_key)
            if target_params is None:
                target_params = {spec.name: spec.initial for spec in self.parameter_specs}
            base_edge = self._get_base_edge(image_path)
            target_edge = self._apply_postprocessing(base_edge, target_params)
        self._ground_truth_edges[cache_key] = target_edge

    def _calculate_reward_with_edges(self, image_path: Path, params: Dict[str, float]) -> float:
        self._ensure_ground_truth(image_path)
        gt = self._ground_truth_edges[image_path.name]
        base_edge = self._get_base_edge(image_path)
        prediction = self._apply_postprocessing(base_edge, params)
        return self._f1_score(prediction, gt)

    def _load_precomputed_edge(self, image_path: Path) -> Optional[np.ndarray]:
        assert self.config.precomputed_edge_dir is not None
        candidates = [
            self.config.precomputed_edge_dir / image_path.name,
            self.config.precomputed_edge_dir / f"{image_path.stem}_hed.png",
            self.config.precomputed_edge_dir / f"{image_path.stem}.png",
        ]
        for candidate in candidates:
            if candidate.exists():
                edge = cv2.imread(str(candidate), cv2.IMREAD_GRAYSCALE)
                if edge is None:
                    continue
                return edge.astype(np.float32) / 255.0
        return None

    def _load_ground_truth_edge(self, image_path: Path) -> Optional[np.ndarray]:
        assert self.config.ground_truth_dir is not None
        candidates = [
            self.config.ground_truth_dir / image_path.name,
            self.config.ground_truth_dir / f"{image_path.stem}.png",
            self.config.ground_truth_dir / f"{image_path.stem}.jpg",
        ]
        for candidate in candidates:
            if candidate.exists():
                gt = cv2.imread(str(candidate), cv2.IMREAD_GRAYSCALE)
                if gt is None:
                    continue
                normalized = gt.astype(np.float32) / 255.0
                return normalized
        return None

    def _apply_postprocessing(self, base_edge: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        edge = base_edge.astype(np.float32).copy()
        sigma = float(params.get("blur_sigma", 0.0))
        if sigma > 0:
            ksize = max(3, int(2 * round(3 * sigma) + 1))
            if ksize % 2 == 0:
                ksize += 1
            edge = cv2.GaussianBlur(edge, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
        threshold = float(params.get("threshold", 0.5))
        _, edge_bin = cv2.threshold(edge, threshold, 1.0, cv2.THRESH_BINARY)
        morph_radius = max(0, int(round(params.get("morphology_radius", 1.0))))
        if morph_radius > 0:
            kernel_size = 2 * morph_radius + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            edge_bin = cv2.morphologyEx(edge_bin, cv2.MORPH_CLOSE, kernel)
        nms_strength = float(params.get("nms_strength", 0.0))
        nms_strength = max(0.0, min(1.0, nms_strength))
        if nms_strength > 0:
            iterations = max(1, int(round(nms_strength * 2)))
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            edge_bin = cv2.erode(edge_bin, kernel, iterations=iterations)
        return edge_bin.astype(np.float32)

    @staticmethod
    def _f1_score(prediction: np.ndarray, target: np.ndarray) -> float:
        pred = (prediction > 0.5).astype(np.uint8)
        tgt = (target > 0.5).astype(np.uint8)
        tp = int(np.logical_and(pred == 1, tgt == 1).sum())
        fp = int(np.logical_and(pred == 1, tgt == 0).sum())
        fn = int(np.logical_and(pred == 0, tgt == 1).sum())
        if tp == 0:
            return 0.0
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        if precision + recall == 0:
            return 0.0
        return float(2 * precision * recall / (precision + recall))


__all__ = ["HEDPostProcessEnv", "HEDPostProcessConfig", "ParameterSpec"]
