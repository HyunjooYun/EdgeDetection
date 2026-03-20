"""Plot episode reward trajectories for DQN and PPO (0311 runs).

Reads stable-baselines3 Monitor logs under runs/0311/DQN/monitor and
runs/0311/PPO/monitor, and plots episode reward vs episode index for
both algorithms on the same figure, including a moving-average curve.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def load_monitor_file(path: Path) -> Tuple[List[float], List[int]]:
    rewards: List[float] = []
    lengths: List[int] = []

    with path.open("r", encoding="utf-8") as f:
        first = f.readline()
        if not first.startswith("#"):
            f.seek(0)
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


def load_all_rewards(monitor_root: Path) -> np.ndarray:
    all_rewards: List[float] = []
    monitor_files = sorted(monitor_root.rglob("*.monitor.csv"))
    if not monitor_files:
        raise FileNotFoundError(f"No .monitor.csv files found under {monitor_root}")
    for mf in monitor_files:
        rewards, _ = load_monitor_file(mf)
        all_rewards.extend(rewards)
    if not all_rewards:
        raise ValueError(f"No rewards found in monitor logs under {monitor_root}")
    return np.asarray(all_rewards, dtype=np.float32)


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    if values.size < window:
        return values
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(values, kernel, mode="valid")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    runs_root = project_root / "runs" / "0311"

    dqn_rewards = load_all_rewards(runs_root / "DQN" / "monitor")
    ppo_rewards = load_all_rewards(runs_root / "PPO" / "monitor")

    dqn_ma = moving_average(dqn_rewards, window=200)
    ppo_ma = moving_average(ppo_rewards, window=200)

    plt.figure(figsize=(10, 5))
    x_dqn = np.arange(dqn_rewards.size)
    x_ppo = np.arange(ppo_rewards.size)

    plt.plot(x_dqn, dqn_rewards, color="tab:green", alpha=0.15, linewidth=0.5, label="DQN (raw)")
    plt.plot(np.arange(dqn_ma.size) + 200 - 1, dqn_ma, color="tab:green", linewidth=2.0, label="DQN (MA, 200 ep)")

    plt.plot(x_ppo, ppo_rewards, color="tab:orange", alpha=0.15, linewidth=0.5, label="PPO (raw)")
    plt.plot(np.arange(ppo_ma.size) + 200 - 1, ppo_ma, color="tab:orange", linewidth=2.0, label="PPO (MA, 200 ep)")

    plt.xlabel("Episode index")
    plt.ylabel("Episode reward")
    plt.title("Training episode rewards (0311, 200k steps)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_dir = project_root / "artifacts" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "episode_rewards_0311.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
