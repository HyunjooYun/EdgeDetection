"""Utilities for running the HED edge detector via OpenCV's DNN module."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class HedConfig:
    """Configuration for loading and executing the HED DNN."""

    prototxt_path: Path
    caffemodel_path: Path
    input_size: Optional[Tuple[int, int]] = None
    mean_bgr: Tuple[float, float, float] = (104.00698793, 116.66876762, 122.67891434)
    scale: float = 1.0

    def validate(self) -> None:
        if not self.prototxt_path.exists():
            raise FileNotFoundError(f"HED prototxt not found: {self.prototxt_path}")
        if not self.caffemodel_path.exists():
            raise FileNotFoundError(f"HED caffemodel not found: {self.caffemodel_path}")


class HEDModel:
    """Thin wrapper around the OpenCV DNN HED implementation."""

    def __init__(self, config: HedConfig):
        config.validate()
        net = cv2.dnn.readNetFromCaffe(str(config.prototxt_path), str(config.caffemodel_path))
        self._net = net
        self.config = config

    @functools.lru_cache(maxsize=32)
    def _blob_size(self, width: int, height: int) -> Tuple[int, int]:
        if self.config.input_size is None:
            return width, height
        return self.config.input_size

    def infer(self, image_bgr: np.ndarray) -> np.ndarray:
        """Run HED on a BGR image and return a float32 edge map in [0, 1]."""

        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("HEDModel expects a 3-channel BGR image")

        h, w = image_bgr.shape[:2]
        blob_w, blob_h = self._blob_size(w, h)
        blob = cv2.dnn.blobFromImage(
            image_bgr,
            scalefactor=self.config.scale,
            size=(blob_w, blob_h),
            mean=self.config.mean_bgr,
            swapRB=False,
            crop=False,
        )
        self._net.setInput(blob)
        edges = self._net.forward()
        edge_map = edges[0, 0]
        edge_map = cv2.resize(edge_map, (w, h))
        edge_map = np.clip(edge_map, 0.0, 1.0)
        return edge_map.astype(np.float32)


def load_image_bgr(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")
    return image


def infer_hed_edges(model: HEDModel, image_path: Path) -> np.ndarray:
    """Load an image, run HED, and return the edge map."""

    image = load_image_bgr(image_path)
    return model.infer(image)
