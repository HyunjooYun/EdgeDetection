"""Compare GT vs HED, DQN, PPO edge maps on the BSDS test set.

Metrics:
- Pixelwise F1-score
- Pixelwise IoU

Setup (defaults assume this repo layout):
- Test RGB images: inputs/images/test/*.jpg
- BSDS GT edges:   inputs/GT/test/*.mat ("groundTruth" format)
- HED base edges:  outputs/hed/test_baseline/*_hed.png  (0-255)
- HED threshold:   global scalar (default: 0.1)

DQN/PPO predictions are recomputed using the trained models and the
HEDPostProcessEnv (with synthetic internal ground truth, but that does not
affect the predicted edge maps). Evaluation against BSDS GT is done here.

The script also logs per-image F1/IoU curves to TensorBoard so you can
inspect them as graphs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import csv

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from stable_baselines3 import DQN, PPO
from torch.utils.tensorboard import SummaryWriter

from hed_rl import HEDPostProcessConfig, HEDPostProcessEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare GT vs HED/DQN/PPO edge maps (pixel metrics)")
    parser.add_argument("--image-dir", type=Path, default=Path("inputs/images/test"), help="Test RGB images directory")
    parser.add_argument("--gt-dir", type=Path, default=Path("inputs/GT/test"), help="BSDS ground-truth .mat directory")
    parser.add_argument(
        "--hed-dir",
        type=Path,
        default=Path("outputs/hed/test_baseline"),
        help="Directory with HED base edge maps (_hed.png, 0-255)",
    )
    parser.add_argument(
        "--hed-threshold",
        type=float,
        default=0.1,
        help="Global threshold for HED base edges (in [0,1])",
    )
    parser.add_argument("--dqn-model", type=Path, help="Path to trained DQN .zip model")
    parser.add_argument("--ppo-model", type=Path, help="Path to trained PPO .zip model")
    parser.add_argument(
        "--tensorboard-log",
        type=Path,
        default=Path("runs/edge_comparison_test200"),
        help="Directory for TensorBoard logs",
    )
    return parser.parse_args()


# --- Ground truth loading (BSDS .mat) ---------------------------------------------------------


def load_bsds_ground_truth(mat_path: Path) -> np.ndarray:
    """Load a BSDS-style ground truth .mat file as float32 array in [0, 1]."""

    data = sio.loadmat(str(mat_path))
    gt_array = None

    if "groundTruth" in data:
        try:
            gt_cells = data["groundTruth"][0]
            if len(gt_cells) > 0:
                # Each entry is a struct with a "Boundaries" field. Often this
                # is stored as a 1x1 object array whose single element is the
                # actual 2D boundary map.
                def extract_boundaries(cell_entry) -> np.ndarray:
                    b_field = cell_entry[0]["Boundaries"]
                    if isinstance(b_field, np.ndarray) and b_field.dtype == object:
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


# --- Pixel metrics ---------------------------------------------------------------------------


def confusion_from_binary(pred: np.ndarray, gt: np.ndarray) -> Tuple[int, int, int]:
    """Return (tp, fp, fn) for binary maps with values 0 or 1."""

    pred_bin = (pred > 0).astype(np.uint8)
    gt_bin = (gt > 0).astype(np.uint8)

    tp = int(((pred_bin == 1) & (gt_bin == 1)).sum())
    fp = int(((pred_bin == 1) & (gt_bin == 0)).sum())
    fn = int(((pred_bin == 0) & (gt_bin == 1)).sum())
    return tp, fp, fn


def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    if tp == 0:
        return 0.0
    precision = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0.0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def iou_from_counts(tp: int, fp: int, fn: int) -> float:
    denom = tp + fp + fn
    if denom == 0:
        return 0.0
    return float(tp / float(denom))


# --- Helpers -------------------------------------------------------------------------------


def collect_image_ids(gt_dir: Path) -> List[str]:
    mats = sorted(gt_dir.glob("*.mat"))
    if not mats:
        raise SystemExit(f"No .mat files found in {gt_dir}")
    return [m.stem for m in mats]


def load_hed_prob_map(hed_dir: Path, image_id: str) -> np.ndarray:
    hed_path = hed_dir / f"{image_id}_hed.png"
    if not hed_path.exists():
        raise FileNotFoundError(f"HED map not found for {image_id}: {hed_path}")
    hed_img = cv2.imread(str(hed_path), cv2.IMREAD_GRAYSCALE)
    if hed_img is None:
        raise RuntimeError(f"Failed to read HED map at {hed_path}")
    return hed_img.astype(np.float32) / 255.0


def evaluate_hed(
    image_ids: Sequence[str],
    gt_dir: Path,
    hed_dir: Path,
    hed_threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-image (F1, IoU) for HED thresholded maps."""

    f1_list: List[float] = []
    iou_list: List[float] = []

    for image_id in image_ids:
        mat_path = gt_dir / f"{image_id}.mat"
        gt = load_bsds_ground_truth(mat_path)
        gt_bin = (gt >= 0.5).astype(np.uint8)

        hed_prob = load_hed_prob_map(hed_dir, image_id)
        pred_bin = (hed_prob >= hed_threshold).astype(np.uint8)

        tp, fp, fn = confusion_from_binary(pred_bin, gt_bin)
        f1_list.append(f1_from_counts(tp, fp, fn))
        iou_list.append(iou_from_counts(tp, fp, fn))

    return np.asarray(f1_list, dtype=np.float32), np.asarray(iou_list, dtype=np.float32)


def make_env(image_dir: Path, hed_dir: Path) -> HEDPostProcessEnv:
    """Construct an environment that uses precomputed HED edges.

    Ground truth inside the environment is synthetic (for reward only);
    external BSDS GT is used for evaluation metrics instead.
    """

    cfg = HEDPostProcessConfig(
        image_dir=image_dir,
        hed_config=None,
        ground_truth_dir=None,
        precomputed_edge_dir=hed_dir,
        max_steps=30,
        random_seed=42,
        cache_edges=True,
        cycle_images=False,
    )
    return HEDPostProcessEnv(cfg)


def evaluate_agent(
    image_ids: Sequence[str],
    gt_dir: Path,
    image_dir: Path,
    hed_dir: Path,
    model_path: Path,
    algo: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-image (F1, IoU) for DQN or PPO agent."""

    if algo not in {"dqn", "ppo"}:
        raise ValueError("algo must be 'dqn' or 'ppo'")

    EnvCls = make_env
    env = EnvCls(image_dir, hed_dir)

    try:
        if algo == "dqn":
            model = DQN.load(str(model_path), env=env)
        else:
            model = PPO.load(str(model_path), env=env)

        f1_list: List[float] = []
        iou_list: List[float] = []

        for image_id in image_ids:
            image_name = f"{image_id}.jpg"  # test set uses .jpg names
            obs, _ = env.reset(options={"image_name": image_name})
            terminated = False
            truncated = False
            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, _reward, terminated, truncated, _info = env.step(action)

            if env.current_image is None:
                continue

            base_edge = env._get_base_edge(env.current_image)
            pred_edge = env._apply_postprocessing(base_edge, env.current_params)
            pred_bin = (pred_edge >= 0.5).astype(np.uint8)

            mat_path = gt_dir / f"{image_id}.mat"
            gt = load_bsds_ground_truth(mat_path)
            gt_bin = (gt >= 0.5).astype(np.uint8)

            tp, fp, fn = confusion_from_binary(pred_bin, gt_bin)
            f1_list.append(f1_from_counts(tp, fp, fn))
            iou_list.append(iou_from_counts(tp, fp, fn))

        return np.asarray(f1_list, dtype=np.float32), np.asarray(iou_list, dtype=np.float32)
    finally:
        if hasattr(env, "close"):
            env.close()


def log_to_tensorboard(
    writer: SummaryWriter,
    image_ids: Sequence[str],
    method_name: str,
    f1: np.ndarray,
    iou: np.ndarray,
) -> None:
    for idx, (img_id, f, j) in enumerate(zip(image_ids, f1, iou)):
        writer.add_scalar(f"{method_name}/f1", float(f), idx)
        writer.add_scalar(f"{method_name}/iou", float(j), idx)
        writer.add_text(f"{method_name}/image_id", img_id, idx)

    writer.add_scalar(f"summary/{method_name}_f1_mean", float(f1.mean()), 0)
    writer.add_scalar(f"summary/{method_name}_iou_mean", float(iou.mean()), 0)


def save_csv_and_plots(
    image_ids: Sequence[str],
    hed_f1: np.ndarray,
    hed_iou: np.ndarray,
    dqn_f1: Optional[np.ndarray],
    dqn_iou: Optional[np.ndarray],
    ppo_f1: Optional[np.ndarray],
    ppo_iou: Optional[np.ndarray],
) -> None:
    """Save per-image metrics to CSV and plot F1/IoU curves as PNG.

    This is a complement to TensorBoard so you can visually inspect
    all 200 test images even if TensorBoard behaves oddly.
    """

    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    csv_path = artifacts_dir / "edge_comparison_metrics_test200.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "image_id",
                "hed_f1",
                "hed_iou",
                "dqn_f1",
                "dqn_iou",
                "ppo_f1",
                "ppo_iou",
            ]
        )
        for idx, img_id in enumerate(image_ids):
            def safe_get(arr: Optional[np.ndarray]) -> float:
                if arr is None or idx >= arr.shape[0]:
                    return float("nan")
                return float(arr[idx])

            writer.writerow(
                [
                    img_id,
                    float(hed_f1[idx]),
                    float(hed_iou[idx]),
                    safe_get(dqn_f1),
                    safe_get(dqn_iou),
                    safe_get(ppo_f1),
                    safe_get(ppo_iou),
                ]
            )

    # Simple line plots over image index (0..N-1)
    x = np.arange(len(image_ids))

    # F1 plot
    plt.figure(figsize=(10, 5))
    plt.plot(x, hed_f1, label="HED", color="tab:blue")
    if dqn_f1 is not None and dqn_f1.size > 0:
        plt.plot(x[: dqn_f1.size], dqn_f1, label="DQN", color="tab:green")
    if ppo_f1 is not None and ppo_f1.size > 0:
        plt.plot(x[: ppo_f1.size], ppo_f1, label="PPO", color="tab:orange")
    plt.xlabel("Image index (0-199)")
    plt.ylabel("F1 score")
    plt.title("Pixelwise F1 over test images")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    f1_path = artifacts_dir / "edge_comparison_f1_test200.png"
    plt.savefig(f1_path, dpi=150)
    plt.close()

    # IoU plot
    plt.figure(figsize=(10, 5))
    plt.plot(x, hed_iou, label="HED", color="tab:blue")
    if dqn_iou is not None and dqn_iou.size > 0:
        plt.plot(x[: dqn_iou.size], dqn_iou, label="DQN", color="tab:green")
    if ppo_iou is not None and ppo_iou.size > 0:
        plt.plot(x[: ppo_iou.size], ppo_iou, label="PPO", color="tab:orange")
    plt.xlabel("Image index (0-199)")
    plt.ylabel("IoU")
    plt.title("Pixelwise IoU over test images")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    iou_path = artifacts_dir / "edge_comparison_iou_test200.png"
    plt.savefig(iou_path, dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()

    image_ids = collect_image_ids(args.gt_dir)
    print(f"Found {len(image_ids)} test images from GT dir {args.gt_dir}")

    args.tensorboard_log.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(args.tensorboard_log))

    # HED baseline (global threshold)
    print(f"\n[HED] Evaluating with global threshold {args.hed_threshold:.3f}")
    hed_f1, hed_iou = evaluate_hed(image_ids, args.gt_dir, args.hed_dir, args.hed_threshold)
    print(
        f"HED: mean F1={hed_f1.mean():.6f}, std F1={hed_f1.std():.6f}, "
        f"mean IoU={hed_iou.mean():.6f}, std IoU={hed_iou.std():.6f}"
    )
    log_to_tensorboard(writer, image_ids, "hed", hed_f1, hed_iou)

    # DQN
    dqn_f1: Optional[np.ndarray] = None
    dqn_iou: Optional[np.ndarray] = None
    if args.dqn_model and args.dqn_model.exists():
        print(f"\n[DQN] Evaluating model at {args.dqn_model}")
        dqn_f1, dqn_iou = evaluate_agent(
            image_ids=image_ids,
            gt_dir=args.gt_dir,
            image_dir=args.image_dir,
            hed_dir=args.hed_dir,
            model_path=args.dqn_model,
            algo="dqn",
        )
        print(
            f"DQN: mean F1={dqn_f1.mean():.6f}, std F1={dqn_f1.std():.6f}, "
            f"mean IoU={dqn_iou.mean():.6f}, std IoU={dqn_iou.std():.6f}"
        )
        log_to_tensorboard(writer, image_ids, "dqn", dqn_f1, dqn_iou)
    else:
        print("\n[DQN] Model not provided or path does not exist; skipping DQN evaluation.")

    # PPO
    ppo_f1: Optional[np.ndarray] = None
    ppo_iou: Optional[np.ndarray] = None
    if args.ppo_model and args.ppo_model.exists():
        print(f"\n[PPO] Evaluating model at {args.ppo_model}")
        ppo_f1, ppo_iou = evaluate_agent(
            image_ids=image_ids,
            gt_dir=args.gt_dir,
            image_dir=args.image_dir,
            hed_dir=args.hed_dir,
            model_path=args.ppo_model,
            algo="ppo",
        )
        print(
            f"PPO: mean F1={ppo_f1.mean():.6f}, std F1={ppo_f1.std():.6f}, "
            f"mean IoU={ppo_iou.mean():.6f}, std IoU={ppo_iou.std():.6f}"
        )
        log_to_tensorboard(writer, image_ids, "ppo", ppo_f1, ppo_iou)
    else:
        print("\n[PPO] Model not provided or path does not exist; skipping PPO evaluation.")

    # Save CSV + PNG plots for direct inspection
    save_csv_and_plots(image_ids, hed_f1, hed_iou, dqn_f1, dqn_iou, ppo_f1, ppo_iou)

    writer.flush()
    writer.close()
    print(f"\nTensorBoard logs written to: {args.tensorboard_log}")
    print("Saved per-image metrics CSV and plots under artifacts/.")


if __name__ == "__main__":
    main()
