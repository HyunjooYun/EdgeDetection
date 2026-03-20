"""Compare GT vs HED, DQN, PPO using boundary-based F1 (BPR-style).

This uses the same boundary F1 implementation as the training environment
(`HEDPostProcessEnv._f1_score`), which:
- binarizes prediction and target at 0.5,
- matches boundaries with a small spatial tolerance via distance transforms.

Setup (defaults):
- Test images: inputs/images/test/*.jpg
- BSDS GT:     inputs/GT/test/*.mat
- HED maps:    outputs/hed/test_baseline/*_hed.png (0-255)
- DQN model:   artifacts/dqn_bpr_200k_0305.zip (example)
- PPO model:   artifacts/ppo_train_0305.zip (example)

Outputs:
- Prints mean boundary F1 for HED / DQN / PPO over 200 test images.
- Writes per-image F1 values to CSV + a simple line plot PNG in artifacts/.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import csv

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from stable_baselines3 import DQN, PPO
from torch.utils.tensorboard import SummaryWriter

from hed_rl import HEDPostProcessConfig, HEDPostProcessEnv


# -----------------------------------------------------------------------------
# Args
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare HED/DQN/PPO using boundary F1 (BPR-style)")
    parser.add_argument("--image-dir", type=Path, default=Path("inputs/images/test"), help="Test images directory")
    parser.add_argument("--gt-dir", type=Path, default=Path("inputs/GT/test"), help="BSDS GT .mat directory")
    parser.add_argument(
        "--hed-dir",
        type=Path,
        default=Path("outputs/hed/test_baseline"),
        help="Directory with HED base edge maps (_hed.png, 0-255) used as precomputed edges for agents",
    )
    parser.add_argument(
        "--hed-binary-dir",
        type=Path,
        default=Path("outputs/hed/test_baseline_binary_hist"),
        help=(
            "Directory with HED binary edge maps (e.g. *_hed_bin.png) "
            "for baseline HED vs GT evaluation (typically histogram-based)."
        ),
    )
    parser.add_argument("--dqn-model", type=Path, help="Path to trained DQN .zip model")
    parser.add_argument("--ppo-model", type=Path, help="Path to trained PPO .zip model")
    parser.add_argument(
        "--tensorboard-log",
        type=Path,
        default=Path("runs/edge_comparison_bpr_test200"),
        help="Directory for TensorBoard logs",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help=(
            "Optional tag to append to output CSV/plot filenames "
            "(e.g., '0321' -> edge_comparison_bpr_metrics_test200_0321.csv)."
        ),
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# GT loader (BSDS groundTruth format)
# -----------------------------------------------------------------------------


def load_bsds_ground_truth(mat_path: Path) -> np.ndarray:
    """Load BSDS groundTruth .mat as float32 array in [0,1]."""

    data = sio.loadmat(str(mat_path))
    gt_array = None

    if "groundTruth" in data:
        try:
            gt_cells = data["groundTruth"][0]
            if len(gt_cells) > 0:
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


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def collect_image_ids(gt_dir: Path) -> List[str]:
    mats = sorted(gt_dir.glob("*.mat"))
    if not mats:
        raise SystemExit(f"No .mat files found in {gt_dir}")
    return [m.stem for m in mats]


def load_hed_binary_map(hed_binary_dir: Path, image_id: str) -> np.ndarray:
    """Load a thresholded HED binary map as float32 in [0,1].

    Expects files named like ``<image_id>_hed_bin.png`` in ``hed_binary_dir``.
    """

    hed_path = hed_binary_dir / f"{image_id}_hed_bin.png"
    if not hed_path.exists():
        raise FileNotFoundError(f"HED binary map not found for {image_id}: {hed_path}")
    img = cv2.imread(str(hed_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Failed to read HED binary map: {hed_path}")
    return (img.astype(np.float32) / 255.0)


# -----------------------------------------------------------------------------
# Boundary F1 evaluation
# -----------------------------------------------------------------------------


def evaluate_hed_bpr(image_ids: Sequence[str], gt_dir: Path, hed_binary_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-image boundary F1 and IoU (BPR-style) for HED vs BSDS GT.

    IoU is derived from F1 using the standard relation for a single
    confusion matrix: IoU = F1 / (2 - F1). This yields an IoU that is
    consistent with the boundary-based precision/recall used by
    :meth:`HEDPostProcessEnv._f1_score`.
    """

    f1_list: List[float] = []
    iou_list: List[float] = []
    for image_id in image_ids:
        mat_path = gt_dir / f"{image_id}.mat"
        gt = load_bsds_ground_truth(mat_path)
        hed_bin = load_hed_binary_map(hed_binary_dir, image_id)

        # Defensive resize if needed
        if gt.shape != hed_bin.shape:
            gt = cv2.resize(gt, (hed_bin.shape[1], hed_bin.shape[0]), interpolation=cv2.INTER_NEAREST)

        f1 = float(HEDPostProcessEnv._f1_score(hed_bin, gt))
        if f1 <= 0.0:
            iou = 0.0
        else:
            iou = float(f1 / (2.0 - f1))
        f1_list.append(f1)
        iou_list.append(iou)

    return np.asarray(f1_list, dtype=np.float32), np.asarray(iou_list, dtype=np.float32)


def make_env(image_dir: Path, hed_dir: Path) -> HEDPostProcessEnv:
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


def evaluate_agent_bpr(
    image_ids: Sequence[str],
    gt_dir: Path,
    image_dir: Path,
    hed_dir: Path,
    model_path: Path,
    algo: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-image boundary F1 and IoU for a DQN or PPO agent vs BSDS GT."""

    if algo not in {"dqn", "ppo"}:
        raise ValueError("algo must be 'dqn' or 'ppo'")

    env = make_env(image_dir, hed_dir)

    try:
        if algo == "dqn":
            model = DQN.load(str(model_path), env=env)
        else:
            model = PPO.load(str(model_path), env=env)

        f1_list: List[float] = []
        iou_list: List[float] = []

        for image_id in image_ids:
            image_name = f"{image_id}.jpg"
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

            mat_path = gt_dir / f"{image_id}.mat"
            gt = load_bsds_ground_truth(mat_path)
            if gt.shape != pred_edge.shape:
                gt = cv2.resize(gt, (pred_edge.shape[1], pred_edge.shape[0]), interpolation=cv2.INTER_NEAREST)

            f1 = float(HEDPostProcessEnv._f1_score(pred_edge, gt))
            if f1 <= 0.0:
                iou = 0.0
            else:
                iou = float(f1 / (2.0 - f1))
            f1_list.append(f1)
            iou_list.append(iou)

        return np.asarray(f1_list, dtype=np.float32), np.asarray(iou_list, dtype=np.float32)
    finally:
        if hasattr(env, "close"):
            env.close()


# -----------------------------------------------------------------------------
# Logging helpers
# -----------------------------------------------------------------------------


def log_to_tensorboard(writer: SummaryWriter, tag: str, f1: np.ndarray, iou: np.ndarray) -> None:
    for idx, (s, j) in enumerate(zip(f1, iou)):
        writer.add_scalar(f"{tag}/bpr_f1", float(s), idx)
        writer.add_scalar(f"{tag}/bpr_iou", float(j), idx)
    writer.add_scalar(f"summary/{tag}_bpr_f1_mean", float(f1.mean()), 0)
    writer.add_scalar(f"summary/{tag}_bpr_iou_mean", float(iou.mean()), 0)


def save_csv_and_plot(
    image_ids: Sequence[str],
    hed_f1: np.ndarray,
    hed_iou: np.ndarray,
    dqn_f1: Optional[np.ndarray],
    dqn_iou: Optional[np.ndarray],
    ppo_f1: Optional[np.ndarray],
    ppo_iou: Optional[np.ndarray],
    tag: str = "",
) -> None:
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_{tag}" if tag else ""
    csv_path = artifacts_dir / f"edge_comparison_bpr_metrics_test200{suffix}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "image_id",
                "hed_bpr_f1",
                "hed_bpr_iou",
                "dqn_bpr_f1",
                "dqn_bpr_iou",
                "ppo_bpr_f1",
                "ppo_bpr_iou",
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

    x = np.arange(len(image_ids))
    plt.figure(figsize=(10, 5))
    plt.plot(x, hed_f1, label="HED", color="tab:blue")
    if dqn_f1 is not None and dqn_f1.size > 0:
        plt.plot(x[: dqn_f1.size], dqn_f1, label="DQN", color="tab:green")
    if ppo_f1 is not None and ppo_f1.size > 0:
        plt.plot(x[: ppo_f1.size], ppo_f1, label="PPO", color="tab:orange")
    plt.xlabel("Image index (0-199)")
    plt.ylabel("Boundary F1 (BPR-style)")
    plt.title("Boundary F1 over test images (BSDS GT)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path_f1 = artifacts_dir / f"edge_comparison_bpr_f1_test200{suffix}.png"
    plt.savefig(out_path_f1, dpi=150)
    plt.close()

    # IoU plot
    plt.figure(figsize=(10, 5))
    plt.plot(x, hed_iou, label="HED", color="tab:blue")
    if dqn_iou is not None and dqn_iou.size > 0:
        plt.plot(x[: dqn_iou.size], dqn_iou, label="DQN", color="tab:green")
    if ppo_iou is not None and ppo_iou.size > 0:
        plt.plot(x[: ppo_iou.size], ppo_iou, label="PPO", color="tab:orange")
    plt.xlabel("Image index (0-199)")
    plt.ylabel("Boundary IoU (BPR-style)")
    plt.title("Boundary IoU over test images (BSDS GT)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path_iou = artifacts_dir / f"edge_comparison_bpr_iou_test200{suffix}.png"
    plt.savefig(out_path_iou, dpi=150)
    plt.close()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    image_ids = collect_image_ids(args.gt_dir)
    print(f"Found {len(image_ids)} test images from GT dir {args.gt_dir}")

    args.tensorboard_log.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(args.tensorboard_log))

    # HED baseline (using histogram-based binary maps by default)
    print("\n[HED] Evaluating boundary F1/IoU vs BSDS GT (binary maps)")
    hed_f1, hed_iou = evaluate_hed_bpr(image_ids, args.gt_dir, args.hed_binary_dir)
    print(
        f"HED: mean BPR F1={hed_f1.mean():.6f}, std={hed_f1.std():.6f}, "
        f"mean BPR IoU={hed_iou.mean():.6f}, std={hed_iou.std():.6f}"
    )
    log_to_tensorboard(writer, "hed_bpr", hed_f1, hed_iou)

    # DQN agent
    dqn_f1: Optional[np.ndarray] = None
    dqn_iou: Optional[np.ndarray] = None
    if args.dqn_model and args.dqn_model.exists():
        print(f"\n[DQN] Evaluating boundary F1/IoU for model at {args.dqn_model}")
        dqn_f1, dqn_iou = evaluate_agent_bpr(
            image_ids=image_ids,
            gt_dir=args.gt_dir,
            image_dir=args.image_dir,
            hed_dir=args.hed_dir,
            model_path=args.dqn_model,
            algo="dqn",
        )
        print(
            f"DQN: mean BPR F1={dqn_f1.mean():.6f}, std={dqn_f1.std():.6f}, "
            f"mean BPR IoU={dqn_iou.mean():.6f}, std={dqn_iou.std():.6f}"
        )
        log_to_tensorboard(writer, "dqn_bpr", dqn_f1, dqn_iou)
    else:
        print("\n[DQN] Model not provided or does not exist; skipping DQN.")

    # PPO agent
    ppo_f1: Optional[np.ndarray] = None
    ppo_iou: Optional[np.ndarray] = None
    if args.ppo_model and args.ppo_model.exists():
        print(f"\n[PPO] Evaluating boundary F1/IoU for model at {args.ppo_model}")
        ppo_f1, ppo_iou = evaluate_agent_bpr(
            image_ids=image_ids,
            gt_dir=args.gt_dir,
            image_dir=args.image_dir,
            hed_dir=args.hed_dir,
            model_path=args.ppo_model,
            algo="ppo",
        )
        print(
            f"PPO: mean BPR F1={ppo_f1.mean():.6f}, std={ppo_f1.std():.6f}, "
            f"mean BPR IoU={ppo_iou.mean():.6f}, std={ppo_iou.std():.6f}"
        )
        log_to_tensorboard(writer, "ppo_bpr", ppo_f1, ppo_iou)
    else:
        print("\n[PPO] Model not provided or does not exist; skipping PPO.")

    save_csv_and_plot(image_ids, hed_f1, hed_iou, dqn_f1, dqn_iou, ppo_f1, ppo_iou, tag=args.tag)

    writer.flush()
    writer.close()
    print(f"\nTensorBoard logs written to: {args.tensorboard_log}")
    print("Saved BPR F1 CSV and plot under artifacts/.")


if __name__ == "__main__":
    main()
