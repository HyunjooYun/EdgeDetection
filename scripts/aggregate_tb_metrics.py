"""Aggregate scalar metrics from TensorBoard logs for selected runs.

We focus on the 0305 experiments (DQN_3, PPO_2) and compute
simple averages over the whole training for available tags.

Tags of interest (only those that actually exist will be reported):
- rollout/ep_rew_mean       (episode reward mean)
- rollout/ep_len_mean       (episode length mean)
- eval/reward_mean          (evaluation reward mean)
- eval/reward_std           (evaluation reward std)
- eval/reward_max           (evaluation reward max)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


TAGS_OF_INTEREST: List[str] = [
    "rollout/ep_rew_mean",
    "rollout/ep_len_mean",
    "eval/reward_mean",
    "eval/reward_std",
    "eval/reward_max",
]


def load_scalars(run_dir: Path) -> Dict[str, List[float]]:
    """Load all scalar values for tags of interest from a run directory."""

    scalars: Dict[str, List[float]] = {tag: [] for tag in TAGS_OF_INTEREST}

    event_files = sorted(run_dir.glob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No event files found in {run_dir}")

    for ef in event_files:
        acc = EventAccumulator(str(ef))
        acc.Reload()
        available_tags = set(acc.Tags().get("scalars", []))
        for tag in TAGS_OF_INTEREST:
            if tag not in available_tags:
                continue
            for ev in acc.Scalars(tag):
                scalars[tag].append(float(ev.value))

    # Drop tags that ended up with no values at all
    return {tag: values for tag, values in scalars.items() if values}


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    runs_root = project_root / "runs" / "0305"

    runs = {
        "DQN_3": runs_root / "DQN" / "DQN_3",
        "PPO_2": runs_root / "PPO" / "PPO_2",
    }

    for name, run_dir in runs.items():
        print(f"\n=== {name} ({run_dir}) ===")
        if not run_dir.exists():
            print("[WARN] Run directory does not exist")
            continue
        try:
            scalars = load_scalars(run_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] Failed to load scalars: {exc}")
            continue

        if not scalars:
            print("No scalar tags of interest were found.")
            continue

        for tag, values in scalars.items():
            avg = sum(values) / len(values)
            print(f"{tag}: mean over {len(values)} points = {avg:.4f}")


if __name__ == "__main__":
    main()
