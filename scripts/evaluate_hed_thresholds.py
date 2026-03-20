"""Sweep HED thresholds on the BSDS test set and save binary maps.

For each test image, this script:
- loads the BSDS ground-truth edge map from a .mat file
- loads the corresponding HED edge probability map (0-255 PNG)
- evaluates pixel-wise F1 scores for thresholds in [0.1, 0.9] (step 0.1)
- selects the threshold with the highest dataset-level F1
- saves binary HED edge maps for that best threshold

Default paths are aligned with this repository:
- Ground truth: inputs/GT/test/*.mat
- HED maps:    outputs/hed/test_baseline/*_hed.png
- Output bin:  outputs/hed/test_baseline_binary/
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
import scipy.io as sio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate HED thresholds on the test set")
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=Path("inputs/GT/test"),
        help="Directory with BSDS ground-truth .mat files",
    )
    parser.add_argument(
        "--hed-dir",
        type=Path,
        default=Path("outputs/hed/test_baseline"),
        help="Directory with HED edge maps (_hed.png)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/hed/test_baseline_binary"),
        help="Directory to save binary HED maps for the chosen threshold",
    )
    parser.add_argument(
        "--method",
        choices=["f1", "hist", "fixed"],
        default="f1",
        help=(
            "Threshold selection method: "
            "'f1' = dataset-level F1 vs BSDS GT (0.1~0.9), "
            "'hist' = histogram-based (global Otsu on HED maps, ignores GT), "
            "'fixed' = use a user-specified fixed threshold."
        ),
    )
    parser.add_argument(
        "--fixed-threshold",
        type=float,
        default=0.5,
        help=(
            "Fixed threshold in [0,1] to use when --method fixed is selected. "
            "This bypasses GT and histogram-based selection."
        ),
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


def compute_f1(tp: int, fp: int, fn: int) -> float:
    if tp == 0:
        return 0.0
    precision = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0.0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def collect_hed_files(hed_dir: Path) -> List[Path]:
    files = sorted(hed_dir.glob("*_hed.png"))
    if not files:
        raise SystemExit(f"No HED edge maps found in {hed_dir}")
    return files


def compute_global_otsu_threshold(hed_dir: Path) -> float:
    """Compute a global Otsu threshold from all HED maps in a directory.

    This builds a single 256-bin histogram over all pixels of all
    *_hed.png images and applies Otsu's between-class variance
    maximization to obtain a threshold in [0,1].
    """

    hed_files = collect_hed_files(hed_dir)
    if not hed_files:
        raise SystemExit(f"No HED edge maps found in {hed_dir}")

    hist = np.zeros(256, dtype=np.float64)
    for hed_path in hed_files:
        img = cv2.imread(str(hed_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[WARN] Failed to read HED map {hed_path} - skipping")
            continue
        # Accumulate 256-bin histogram over [0, 255]
        h = cv2.calcHist([img], [0], None, [256], [0, 256])
        hist += h[:, 0]

    total = hist.sum()
    if total == 0:
        raise SystemExit("Global histogram is empty; cannot compute Otsu threshold")

    # Normalize to probabilities
    prob = hist / total
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256, dtype=np.float64))
    mu_t = mu[-1]

    # Between-class variance: sigma_b^2 = (mu_t * omega - mu)^2 / (omega * (1 - omega))
    denom = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b2 = (mu_t * omega - mu) ** 2 / denom
        sigma_b2[denom == 0.0] = 0.0

    best_idx = int(np.argmax(sigma_b2))
    return best_idx / 255.0


def evaluate_thresholds(
    gt_dir: Path,
    hed_dir: Path,
    thresholds: Iterable[float],
) -> Tuple[Dict[float, float], int]:
    """Return dataset-level F1 for each threshold and number of images used."""

    hed_files = collect_hed_files(hed_dir)
    thresholds = list(thresholds)

    tp_sum = {t: 0 for t in thresholds}
    fp_sum = {t: 0 for t in thresholds}
    fn_sum = {t: 0 for t in thresholds}

    used_images = 0

    for hed_path in hed_files:
        image_id = hed_path.stem.replace("_hed", "")
        mat_path = gt_dir / f"{image_id}.mat"
        if not mat_path.exists():
            print(f"[WARN] Missing GT .mat for {image_id} ({mat_path}) - skipping")
            continue

        hed_img = cv2.imread(str(hed_path), cv2.IMREAD_GRAYSCALE)
        if hed_img is None:
            print(f"[WARN] Failed to read HED map {hed_path} - skipping")
            continue
        hed_prob = hed_img.astype(np.float32) / 255.0

        try:
            gt = load_bsds_ground_truth(mat_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Failed to load GT from {mat_path}: {exc} - skipping")
            continue

        # Ensure same spatial size (defensive; they should already match).
        if gt.shape != hed_prob.shape:
            gt = cv2.resize(gt, (hed_prob.shape[1], hed_prob.shape[0]), interpolation=cv2.INTER_NEAREST)

        gt_bin = (gt >= 0.5).astype(np.uint8)

        used_images += 1

        for t in thresholds:
            pred_bin = (hed_prob >= t).astype(np.uint8)
            tp = int(((pred_bin == 1) & (gt_bin == 1)).sum())
            fp = int(((pred_bin == 1) & (gt_bin == 0)).sum())
            fn = int(((pred_bin == 0) & (gt_bin == 1)).sum())
            tp_sum[t] += tp
            fp_sum[t] += fp
            fn_sum[t] += fn

    if used_images == 0:
        raise SystemExit("No images with both HED map and ground truth could be evaluated")

    f1_scores: Dict[float, float] = {}
    for t in thresholds:
        f1_scores[t] = compute_f1(tp_sum[t], fp_sum[t], fn_sum[t])
    return f1_scores, used_images


def save_binary_maps(hed_dir: Path, out_dir: Path, threshold: float) -> None:
    hed_files = collect_hed_files(hed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for hed_path in hed_files:
        hed_img = cv2.imread(str(hed_path), cv2.IMREAD_GRAYSCALE)
        if hed_img is None:
            print(f"[WARN] Failed to read {hed_path} - skipping")
            continue
        hed_prob = hed_img.astype(np.float32) / 255.0
        pred_bin = (hed_prob >= threshold).astype(np.uint8) * 255

        image_id = hed_path.stem.replace("_hed", "")
        out_path = out_dir / f"{image_id}_hed_bin.png"
        cv2.imwrite(str(out_path), pred_bin)

    print(f"Saved binary HED maps to {out_dir}")


def main() -> None:
    args = parse_args()
    print(f"Evaluating thresholds on HED dir: {args.hed_dir}")

    if args.method == "f1":
        thresholds = [round(0.1 * i, 1) for i in range(1, 10)]  # 0.1, 0.2, ..., 0.9

        print(f"Using F1-based method with GT dir: {args.gt_dir}")
        print(f"Candidate thresholds: {thresholds}")

        f1_scores, used_images = evaluate_thresholds(args.gt_dir, args.hed_dir, thresholds)

        print(f"\nUsed images: {used_images}")
        print("Threshold vs F1 (dataset-level):")
        for t in thresholds:
            print(f"  t={t:.1f}: F1={f1_scores[t]:.6f}")
        best_threshold = max(thresholds, key=lambda t: f1_scores[t])
        print(f"\nBest threshold (F1-based): {best_threshold:.3f} (F1={f1_scores[best_threshold]:.6f})")
    elif args.method == "hist":
        print("Using histogram-based method (global Otsu, ignores GT)")
        best_threshold = compute_global_otsu_threshold(args.hed_dir)
        print(f"Global Otsu threshold: {best_threshold:.6f} (in [0,1] scale)")
    else:  # args.method == "fixed"
        best_threshold = float(args.fixed_threshold)
        print("Using fixed threshold (ignores GT and hist-based selection)")
        print(f"Fixed threshold: {best_threshold:.6f} (in [0,1] scale)")

    # Save binary maps for the chosen threshold
    save_binary_maps(args.hed_dir, args.out_dir, best_threshold)


if __name__ == "__main__":
    main()
