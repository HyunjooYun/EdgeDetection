"""Compute exact episode-level metrics from stable-baselines3 Monitor logs.

Reads all ``*.monitor.csv`` files under given run directories
(e.g., runs/0311/DQN/monitor, runs/0311/PPO/monitor) and computes:
- episode_reward_mean, episode_reward_std, episode_reward_max
- episode_length_mean, episode_length_std

This uses ALL recorded episodes during training, not just TensorBoard
summary points.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np


@dataclass
class EpisodeStats:
    episodes: int
    reward_mean: float
    reward_std: float
    reward_max: float
    length_mean: float
    length_std: float


def load_monitor_file(path: Path) -> Tuple[List[float], List[int]]:
    rewards: List[float] = []
    lengths: List[int] = []

    with path.open("r", encoding="utf-8") as f:
        # First line is a comment with metadata, second line is header "r,l,t"
        first = f.readline()
        if not first.startswith("#"):
            # If format differs, rewind so DictReader still sees header
            f.seek(0)
        # Next line should be the CSV header
        reader = csv.DictReader(f)
        for row in reader:
            try:
                r = float(row["r"])
                l = int(row["l"])
            except (KeyError, ValueError):
                continue
            rewards.append(r)
            lengths.append(l)

    return rewards, lengths


def load_monitor_dir(root: Path) -> EpisodeStats:
    all_rewards: List[float] = []
    all_lengths: List[int] = []

    monitor_files = sorted(root.rglob("*.monitor.csv"))
    if not monitor_files:
        raise FileNotFoundError(f"No .monitor.csv files found under {root}")

    for mf in monitor_files:
        rewards, lengths = load_monitor_file(mf)
        all_rewards.extend(rewards)
        all_lengths.extend(lengths)

    if not all_rewards or not all_lengths:
        raise ValueError(f"No episode data found in monitor files under {root}")

    rewards_np = np.asarray(all_rewards, dtype=np.float32)
    lengths_np = np.asarray(all_lengths, dtype=np.float32)

    return EpisodeStats(
        episodes=int(rewards_np.shape[0]),
        reward_mean=float(rewards_np.mean()),
        reward_std=float(rewards_np.std(ddof=0)),
        reward_max=float(rewards_np.max()),
        length_mean=float(lengths_np.mean()),
        length_std=float(lengths_np.std(ddof=0)),
    )


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    # Adjust these to point to the 200k training runs you want to analyze.
    runs_root = project_root / "runs" / "0311"
    dqn_monitor_root = runs_root / "DQN" / "monitor"
    ppo_monitor_root = runs_root / "PPO" / "monitor"

    for algo, root in [("DQN", dqn_monitor_root), ("PPO", ppo_monitor_root)]:
        print(f"\n=== {algo} training episodes from {root} ===")
        if not root.exists():
            print("[WARN] monitor directory does not exist")
            continue
        try:
            stats = load_monitor_dir(root)
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] Failed to load episode stats: {exc}")
            continue

        print(f"Episodes: {stats.episodes}")
        print(f"Episode Reward Mean: {stats.reward_mean:.4f}")
        print(f"Episode Reward Std: {stats.reward_std:.4f}")
        print(f"Episode Reward Max: {stats.reward_max:.4f}")
        print(f"Episode Length Mean: {stats.length_mean:.4f}")
        print(f"Episode Length Std: {stats.length_std:.4f}")


if __name__ == "__main__":
    main()
