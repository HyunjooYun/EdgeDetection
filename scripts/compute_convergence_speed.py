"""Compute convergence speed for DQN and PPO from TensorBoard logs.

For each run, we:
1. Read the scalar `rollout/ep_rew_mean` from the event file.
2. Take the mean of the last K points as the "final" value.
3. Define threshold = final_value * FRACTION (e.g. 0.9).
4. Find the earliest training step where the scalar >= threshold.

This gives a simple estimate of convergence speed in terms of env steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


@dataclass
class RunStats:
    name: str
    final_value: float
    threshold_fraction: float
    threshold_value: float
    convergence_step: int | None


def load_scalar(path: Path, tag: str) -> List[Tuple[int, float]]:
    acc = EventAccumulator(str(path))
    acc.Reload()
    events = acc.Scalars(tag)
    return [(e.step, float(e.value)) for e in events]


def compute_convergence(
    events: List[Tuple[int, float]], threshold_fraction: float = 0.9, tail_points: int = 20
) -> RunStats:
    if not events:
        return RunStats("", 0.0, threshold_fraction, 0.0, None)

    steps, values = zip(*events)
    values = list(values)

    k = min(len(values), max(1, tail_points))
    tail = values[-k:]
    final_value = sum(tail) / len(tail)
    threshold = final_value * threshold_fraction

    conv_step: int | None = None
    for step, value in events:
        if value >= threshold:
            conv_step = int(step)
            break

    return RunStats("", final_value, threshold_fraction, threshold, conv_step)


def analyze_run(run_dir: Path, tag: str = "rollout/ep_rew_mean") -> RunStats:
    event_files = sorted(run_dir.glob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No event files found in {run_dir}")

    events: List[Tuple[int, float]] = []
    for ef in event_files:
        events.extend(load_scalar(ef, tag))

    stats = compute_convergence(events)
    stats.name = run_dir.name
    return stats


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    print("Using tag: rollout/ep_rew_mean, threshold: 90% of final 20 points")

    # Analyze 0305 runs (multiple seeds) which correspond to
    # artifacts/dqn_bpr_200k_0305.zip and artifacts/ppo_train_0305.zip.
    runs_root_0305 = project_root / "runs" / "0305"
    dqn_root_0305 = runs_root_0305 / "DQN"
    ppo_root_0305 = runs_root_0305 / "PPO"

    for algo, root in [("DQN", dqn_root_0305), ("PPO", ppo_root_0305)]:
        if not root.exists():
            print(f"[WARN] Root directory not found for {algo} (0305): {root}")
            continue

        run_dirs = sorted(d for d in root.iterdir() if d.is_dir() and not d.name.startswith("rollout"))
        if not run_dirs:
            print(f"[WARN] No run subdirectories found for {algo} in {root}")
            continue

        for run_dir in run_dirs:
            try:
                stats = analyze_run(run_dir)
            except Exception as exc:  # noqa: BLE001
                print(f"[ERROR] Failed to analyze {algo} at {run_dir}: {exc}")
                continue

            print(f"\n=== 0305 {algo} ({run_dir.name}) ===")
            print(f"Final mean (tail avg): {stats.final_value:.4f}")
            print(f"Threshold (90%): {stats.threshold_value:.4f}")
            if stats.convergence_step is None:
                print("Convergence step: never reached threshold")
            else:
                print(f"Convergence step: {stats.convergence_step}")


if __name__ == "__main__":
    main()
