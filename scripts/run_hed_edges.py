"""Generate HED edge maps for a directory of images."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from hed_rl import HEDModel, HedConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the HED model on a folder of images")
    parser.add_argument("--prototxt", type=Path, required=True, help="Path to deploy.prototxt for HED")
    parser.add_argument("--caffemodel", type=Path, required=True, help="Path to hed_pretrained_bsds.caffemodel")
    parser.add_argument("--image-dir", type=Path, default=Path("imgs/test"), help="Directory with images to process")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/hed"), help="Directory to store edge maps")
    parser.add_argument("--width", type=int, default=0, help="Optional width to resize input (0 keeps original)")
    parser.add_argument("--height", type=int, default=0, help="Optional height to resize input (0 keeps original)")
    return parser.parse_args()


def _collect_images(image_dir: Path) -> list[Path]:
    supported = {".png", ".jpg", ".jpeg", ".bmp"}
    return sorted(
        path for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in supported
    )


def main() -> None:
    args = parse_args()
    input_size = None
    if args.width > 0 and args.height > 0:
        input_size = (args.width, args.height)

    config = HedConfig(
        prototxt_path=args.prototxt,
        caffemodel_path=args.caffemodel,
        input_size=input_size,
    )
    model = HEDModel(config)

    images = _collect_images(args.image_dir)
    if not images:
        raise ValueError(f"No images found in {args.image_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in images:
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[WARN] Failed to load {image_path}")
            continue
        edges = model.infer(bgr)
        edges_uint8 = (np.clip(edges, 0.0, 1.0) * 255).astype(np.uint8)
        out_path = args.output_dir / f"{image_path.stem}_hed.png"
        cv2.imwrite(str(out_path), edges_uint8)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
