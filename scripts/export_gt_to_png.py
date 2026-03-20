"""Export BSDS-style ground-truth edge maps to PNG files.

For each .mat file in the GT directory, this script:
- loads the BSDS groundTruth structure
- merges all "Boundaries" into a single edge map
- saves it as a 0/255 uint8 PNG image

Defaults are aligned with this repo:
- Input GT:  inputs/GT/test/*.mat
- Output:   outputs/GT/test_png/<image_id>_gt.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import scipy.io as sio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export BSDS GT edge maps to PNG")
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=Path("inputs/GT/test"),
        help="Directory with BSDS ground-truth .mat files",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/GT/test_png"),
        help="Directory to save GT edge PNGs",
    )
    return parser.parse_args()


def load_bsds_ground_truth(mat_path: Path) -> np.ndarray:
    """Load a BSDS-style ground truth .mat file as a float32 array in [0, 1]."""

    data = sio.loadmat(str(mat_path))
    gt_array = None

    # BSDS500 convention: data["groundTruth"][0] is a list of structs,
    # each with a "Boundaries" field (H x W, 0/1 or [0,1] float).
    if "groundTruth" in data:
        try:
            gt_cells = data["groundTruth"][0]
            if len(gt_cells) > 0:
                # Each entry is a struct with a "Boundaries" field. In BSDS,
                # this field is often stored as a 1x1 object array whose single
                # element is the actual 2D boundary map.
                def extract_boundaries(cell_entry) -> np.ndarray:
                    b_field = cell_entry[0]["Boundaries"]
                    if isinstance(b_field, np.ndarray) and b_field.dtype == object:
                        # Take the first element and convert to float array.
                        return np.asarray(b_field.flat[0], dtype=np.float32)
                    return np.asarray(b_field, dtype=np.float32)

                first_boundaries = extract_boundaries(gt_cells[0])
                acc = np.zeros_like(first_boundaries, dtype=np.float32)
                for gt_entry in gt_cells:
                    boundaries = extract_boundaries(gt_entry)
                    acc = np.maximum(acc, boundaries)
                gt_array = acc
        except Exception:  # noqa: BLE001
            gt_array = None

    # Fallback: some .mat files may store a direct edge map.
    if gt_array is None:
        for key in ("edge", "edges", "gt"):
            if key in data:
                arr = np.asarray(data[key], dtype=np.float32)
                if arr.ndim >= 2:
                    gt_array = arr
                    break

    if gt_array is None:
        raise ValueError(f"Could not extract ground truth edge map from {mat_path}")

    gt_array = np.asarray(gt_array, dtype=np.float32)
    gt_array = np.clip(gt_array, 0.0, 1.0)
    return gt_array


def export_gt_pngs(gt_dir: Path, out_dir: Path) -> None:
    mats = sorted(gt_dir.glob("*.mat"))
    if not mats:
        raise SystemExit(f"No .mat files found in {gt_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for mat_path in mats:
        image_id = mat_path.stem
        try:
            gt = load_bsds_ground_truth(mat_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Failed to load GT from {mat_path}: {exc} - skipping")
            continue

        # Convert [0,1] float to 0/255 uint8
        gt_uint8 = (np.clip(gt, 0.0, 1.0) * 255.0).astype(np.uint8)

        out_path = out_dir / f"{image_id}_gt.png"
        cv2.imwrite(str(out_path), gt_uint8)
        count += 1

    print(f"Exported {count} GT PNGs to {out_dir}")


def main() -> None:
    args = parse_args()
    print(f"Exporting GT maps from {args.gt_dir} to {args.out_dir}")
    export_gt_pngs(args.gt_dir, args.out_dir)


if __name__ == "__main__":
    main()
