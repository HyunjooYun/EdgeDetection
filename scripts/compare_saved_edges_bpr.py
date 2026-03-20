"""Compare saved GT, HED, DQN, PPO edge maps using BPR-based F1/IoU.

This script assumes that all maps are already saved as PNGs:
- GT maps:   outputs/GT/test_png/<image_id>_gt.png
- HED maps:  outputs/hed/test_baseline_binary_hist/<image_id>_hed_bin.png
- DQN maps:  outputs/dqn/test/<image_id>_dqn.png
- PPO maps:  outputs/ppo/test/<image_id>_ppo.png

For each image_id, it:
- loads GT and prediction maps as float32 in [0,1]
- computes boundary F1 using HEDPostProcessEnv._f1_score
- derives IoU from F1 via IoU = F1 / (2 - F1)
- writes per-image metrics to a CSV under artifacts/.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from hed_rl import HEDPostProcessEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare saved GT/HED/DQN/PPO edge maps using BPR F1/IoU")
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=Path("outputs/GT/test_png"),
        help="Directory with GT PNGs (<image_id>_gt.png)",
    )
    parser.add_argument(
        "--hed-dir",
        type=Path,
        default=Path("outputs/hed/test_baseline_binary_hist"),
        help="Directory with HED binary PNGs (<image_id>_hed_bin.png)",
    )
    parser.add_argument(
        "--dqn-dir",
        type=Path,
        default=Path("outputs/dqn/test"),
        help="Directory with DQN post-processed PNGs (<image_id>_dqn.png)",
    )
    parser.add_argument(
        "--ppo-dir",
        type=Path,
        default=Path("outputs/ppo/test"),
        help="Directory with PPO post-processed PNGs (<image_id>_ppo.png)",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("artifacts/edge_comparison_bpr_metrics_saved_png.csv"),
        help="Path to save per-image BPR F1/IoU metrics as CSV",
    )
    return parser.parse_args()


def collect_image_ids(gt_dir: Path) -> List[str]:
    pngs = sorted(gt_dir.glob("*_gt.png"))
    if not pngs:
        raise SystemExit(f"No GT PNG files found in {gt_dir}")
    ids: List[str] = []
    for path in pngs:
        stem = path.stem
        if stem.endswith("_gt"):
            ids.append(stem[:-3])
        else:
            ids.append(stem)
    return ids


def load_gray01(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return img.astype(np.float32) / 255.0


def compute_f1_iou_for_methods(
    image_ids: List[str],
    gt_dir: Path,
    hed_dir: Path,
    dqn_dir: Path,
    ppo_dir: Path,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
    hed_f1: Dict[str, float] = {}
    hed_iou: Dict[str, float] = {}
    dqn_f1: Dict[str, float] = {}
    dqn_iou: Dict[str, float] = {}
    ppo_f1: Dict[str, float] = {}
    ppo_iou: Dict[str, float] = {}

    for image_id in image_ids:
        gt_path = gt_dir / f"{image_id}_gt.png"
        if not gt_path.exists():
            print(f"[WARN] Missing GT PNG for {image_id}: {gt_path} - skipping")
            continue
        gt = load_gray01(gt_path)

        # HED
        hed_path = hed_dir / f"{image_id}_hed_bin.png"
        if hed_path.exists():
            hed = load_gray01(hed_path)
            if hed.shape != gt.shape:
                gt_resized = cv2.resize(gt, (hed.shape[1], hed.shape[0]), interpolation=cv2.INTER_NEAREST)
            else:
                gt_resized = gt
            f1 = float(HEDPostProcessEnv._f1_score(hed, gt_resized))
            iou = 0.0 if f1 <= 0.0 else float(f1 / (2.0 - f1))
            hed_f1[image_id] = f1
            hed_iou[image_id] = iou
        else:
            print(f"[WARN] Missing HED PNG for {image_id}: {hed_path}")

        # DQN
        dqn_path = dqn_dir / f"{image_id}_dqn.png"
        if dqn_path.exists():
            dqn = load_gray01(dqn_path)
            if dqn.shape != gt.shape:
                gt_resized = cv2.resize(gt, (dqn.shape[1], dqn.shape[0]), interpolation=cv2.INTER_NEAREST)
            else:
                gt_resized = gt
            f1 = float(HEDPostProcessEnv._f1_score(dqn, gt_resized))
            iou = 0.0 if f1 <= 0.0 else float(f1 / (2.0 - f1))
            dqn_f1[image_id] = f1
            dqn_iou[image_id] = iou
        else:
            print(f"[WARN] Missing DQN PNG for {image_id}: {dqn_path}")

        # PPO
        ppo_path = ppo_dir / f"{image_id}_ppo.png"
        if ppo_path.exists():
            ppo = load_gray01(ppo_path)
            if ppo.shape != gt.shape:
                gt_resized = cv2.resize(gt, (ppo.shape[1], ppo.shape[0]), interpolation=cv2.INTER_NEAREST)
            else:
                gt_resized = gt
            f1 = float(HEDPostProcessEnv._f1_score(ppo, gt_resized))
            iou = 0.0 if f1 <= 0.0 else float(f1 / (2.0 - f1))
            ppo_f1[image_id] = f1
            ppo_iou[image_id] = iou
        else:
            print(f"[WARN] Missing PPO PNG for {image_id}: {ppo_path}")

    return hed_f1, hed_iou, dqn_f1, dqn_iou, ppo_f1, ppo_iou


def save_csv(
    image_ids: List[str],
    out_csv: Path,
    hed_f1: Dict[str, float],
    hed_iou: Dict[str, float],
    dqn_f1: Dict[str, float],
    dqn_iou: Dict[str, float],
    ppo_f1: Dict[str, float],
    ppo_iou: Dict[str, float],
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
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
        for image_id in image_ids:
            def get(d: Dict[str, float]) -> float:
                return float(d.get(image_id, float("nan")))

            writer.writerow(
                [
                    image_id,
                    get(hed_f1),
                    get(hed_iou),
                    get(dqn_f1),
                    get(dqn_iou),
                    get(ppo_f1),
                    get(ppo_iou),
                ]
            )


def main() -> None:
    args = parse_args()
    image_ids = collect_image_ids(args.gt_dir)
    print(f"Found {len(image_ids)} GT images in {args.gt_dir}")

    hed_f1, hed_iou, dqn_f1, dqn_iou, ppo_f1, ppo_iou = compute_f1_iou_for_methods(
        image_ids=image_ids,
        gt_dir=args.gt_dir,
        hed_dir=args.hed_dir,
        dqn_dir=args.dqn_dir,
        ppo_dir=args.ppo_dir,
    )

    save_csv(image_ids, args.out_csv, hed_f1, hed_iou, dqn_f1, dqn_iou, ppo_f1, ppo_iou)

    # Print simple summary
    def mean_from_dict(d: Dict[str, float]) -> float:
        if not d:
            return float("nan")
        return float(np.mean(list(d.values())))

    print(
        "HED: mean BPR F1=%.6f, mean BPR IoU=%.6f"
        % (mean_from_dict(hed_f1), mean_from_dict(hed_iou))
    )
    print(
        "DQN: mean BPR F1=%.6f, mean BPR IoU=%.6f"
        % (mean_from_dict(dqn_f1), mean_from_dict(dqn_iou))
    )
    print(
        "PPO: mean BPR F1=%.6f, mean BPR IoU=%.6f"
        % (mean_from_dict(ppo_f1), mean_from_dict(ppo_iou))
    )


if __name__ == "__main__":
    main()
