"""Export DQN/PPO post-processed edge maps for the BSDS test set.

For each test image, this script:
- loads the precomputed HED edge map
- runs a trained DQN or PPO agent in the HEDPostProcessEnv
- applies the final post-processing parameters to get the predicted edge map
- saves the predicted edge map as a PNG under outputs/.

Defaults are aligned with this repository:
- Test RGB images: inputs/images/test/*.jpg
- Precomputed HED: outputs/hed/test_baseline/*_hed.png
- DQN model:      artifacts/dqn_bpr_200k_0305.zip
- PPO model:      artifacts/ppo_train_0305.zip
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence

import cv2
import numpy as np
from stable_baselines3 import DQN, PPO

from hed_rl import HEDPostProcessConfig, HEDPostProcessEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export DQN/PPO post-processed edge maps for test images")
    parser.add_argument("--image-dir", type=Path, default=Path("inputs/images/test"), help="Test RGB images directory")
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=Path("inputs/GT/test"),
        help="Directory with BSDS ground-truth .mat files (used only to define test image IDs)",
    )
    parser.add_argument(
        "--hed-dir",
        type=Path,
        default=Path("outputs/hed/test_baseline"),
        help="Directory with precomputed HED edge maps (_hed.png, 0-255)",
    )
    parser.add_argument("--dqn-model", type=Path, default=Path("artifacts/dqn_bpr_200k_0305.zip"))
    parser.add_argument("--ppo-model", type=Path, default=Path("artifacts/ppo_train_0305.zip"))
    parser.add_argument(
        "--out-dqn",
        type=Path,
        default=Path("outputs/dqn/test"),
        help="Output directory for DQN post-processed edge PNGs",
    )
    parser.add_argument(
        "--out-ppo",
        type=Path,
        default=Path("outputs/ppo/test"),
        help="Output directory for PPO post-processed edge PNGs",
    )
    return parser.parse_args()


def collect_image_ids(gt_dir: Path) -> List[str]:
    mats = sorted(gt_dir.glob("*.mat"))
    if not mats:
        raise SystemExit(f"No .mat files found in {gt_dir}")
    return [m.stem for m in mats]


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


def run_agent_and_save(
    image_ids: Sequence[str],
    image_dir: Path,
    hed_dir: Path,
    model_path: Path,
    algo: str,
    out_dir: Path,
) -> None:
    if algo not in {"dqn", "ppo"}:
        raise ValueError("algo must be 'dqn' or 'ppo'")

    env = make_env(image_dir, hed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        if algo == "dqn":
            model = DQN.load(str(model_path), env=env)
        else:
            model = PPO.load(str(model_path), env=env)

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

            # Convert [0,1] float to 0-255 uint8 for saving
            pred_uint8 = (np.clip(pred_edge, 0.0, 1.0) * 255.0).astype(np.uint8)
            out_path = out_dir / f"{image_id}_{algo}.png"
            cv2.imwrite(str(out_path), pred_uint8)

        print(f"Saved {algo.upper()} post-processed edges to {out_dir}")
    finally:
        if hasattr(env, "close"):
            env.close()


def main() -> None:
    args = parse_args()

    image_ids = collect_image_ids(args.gt_dir)
    print(f"Found {len(image_ids)} test images from GT dir {args.gt_dir}")

    if args.dqn_model and args.dqn_model.exists():
        print(f"Exporting DQN post-processed edges from {args.dqn_model}")
        run_agent_and_save(
            image_ids=image_ids,
            image_dir=args.image_dir,
            hed_dir=args.hed_dir,
            model_path=args.dqn_model,
            algo="dqn",
            out_dir=args.out_dqn,
        )
    else:
        print("[WARN] DQN model not found; skipping DQN export")

    if args.ppo_model and args.ppo_model.exists():
        print(f"Exporting PPO post-processed edges from {args.ppo_model}")
        run_agent_and_save(
            image_ids=image_ids,
            image_dir=args.image_dir,
            hed_dir=args.hed_dir,
            model_path=args.ppo_model,
            algo="ppo",
            out_dir=args.out_ppo,
        )
    else:
        print("[WARN] PPO model not found; skipping PPO export")


if __name__ == "__main__":
    main()
