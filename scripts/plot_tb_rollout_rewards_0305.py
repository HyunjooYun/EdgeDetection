"""Plot rollout/ep_rew_mean curves for 0305 DQN and PPO runs.

Reads TensorBoard event files under runs/0305/DQN and runs/0305/PPO
and plots rollout/ep_rew_mean vs training step for each run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


TAG = "rollout/ep_rew_mean"


def load_scalars(run_dir: Path, tag: str) -> List[tuple[int, float]]:
    events: List[tuple[int, float]] = []
    event_files = sorted(run_dir.glob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No event files found in {run_dir}")
    for ef in event_files:
        acc = EventAccumulator(str(ef))
        acc.Reload()
        if tag not in acc.Tags().get("scalars", []):
            continue
        for ev in acc.Scalars(tag):
            events.append((int(ev.step), float(ev.value)))
    return events


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    runs_root = project_root / "runs" / "0305"

    runs: Dict[str, Path] = {
        "DQN_1": runs_root / "DQN" / "DQN_1",
        "DQN_2": runs_root / "DQN" / "DQN_2",
        "DQN_3": runs_root / "DQN" / "DQN_3",
        "PPO_1": runs_root / "PPO" / "PPO_1",
        "PPO_2": runs_root / "PPO" / "PPO_2",
    }

    plt.figure(figsize=(10, 5))

    colors = {
        "DQN_1": "tab:green",
        "DQN_2": "tab:olive",
        "DQN_3": "tab:cyan",
        "PPO_1": "tab:orange",
        "PPO_2": "tab:red",
    }

    for name, run_dir in runs.items():
        if not run_dir.exists():
            continue
        try:
            data = load_scalars(run_dir, TAG)
        except Exception:
            continue
        if not data:
            continue
        steps, values = zip(*data)
        plt.plot(steps, values, label=name, color=colors.get(name), alpha=0.9)

    plt.xlabel("Step")
    plt.ylabel(TAG)
    plt.title("0305 runs: rollout/ep_rew_mean")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_dir = project_root / "artifacts" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "rollout_ep_rew_mean_0305.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
