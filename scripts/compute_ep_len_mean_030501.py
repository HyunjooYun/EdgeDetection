import os
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


RUNS_ROOT = Path("runs/030501")
TAG = "rollout/ep_len_mean"
TAIL_POINTS = 20


def load_scalars(run_dir: Path, tag: str):
    ea = EventAccumulator(str(run_dir))
    ea.Reload()
    try:
        scalars = ea.Scalars(tag)
    except KeyError:
        available = ea.Tags().get("scalars", [])
        raise SystemExit(f"Tag '{tag}' not found in {run_dir}. Available scalar tags: {available}")
    return scalars


def summarize_run(label: str, run_dir: Path):
    scalars = load_scalars(run_dir, TAG)
    if not scalars:
        print(f"{label}: no scalars for tag {TAG}")
        return

    steps = [s.step for s in scalars]
    values = [s.value for s in scalars]

    final_step = steps[-1]
    final_value = values[-1]

    tail_vals = values[-TAIL_POINTS:] if len(values) >= TAIL_POINTS else values
    tail_mean = sum(tail_vals) / len(tail_vals)

    global_mean = sum(values) / len(values)

    print(f"=== {label} ({run_dir}) ===")
    print(f"Num points: {len(values)}")
    print(f"Final step: {final_step}")
    print(f"Final rollout/ep_len_mean: {final_value:.4f}")
    print(f"Last {len(tail_vals)} points mean: {tail_mean:.4f}")
    print(f"Global mean over all points: {global_mean:.4f}")
    print()


def main():
    dqn_run = RUNS_ROOT / "DQN" / "DQN_1"
    ppo_run = RUNS_ROOT / "PPO" / "PPO_1"

    if not dqn_run.exists():
        raise SystemExit(f"DQN run dir not found: {dqn_run}")
    if not ppo_run.exists():
        raise SystemExit(f"PPO run dir not found: {ppo_run}")

    summarize_run("030501 DQN_1", dqn_run)
    summarize_run("030501 PPO_1", ppo_run)


if __name__ == "__main__":
    main()
