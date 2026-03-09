"""Convert evaluation JSON result files to CSV.

Looks for files like eval_results*.json under the artifacts directory
(and its subdirectories) and writes a CSV with the same stem in the
same directory.

Assumes JSON structure like:
{
  "results": [
    {"model": "dqn", "mean_reward": ...},
    {"model": "ppo", "mean_reward": ...}
  ]
}
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, List, Dict, Any


def find_eval_json_files(root: Path) -> List[Path]:
    patterns = [
        "eval_results.json",
        "eval_results_test20.json",
        "eval_results_test20_0305.json",
        "eval_test_results.json",
        "eval_val_results.json",
        "eval_results_test200_030501.json",
        "eval_results_test20_030501.json",
        "eval_results_train_030501.json",
        "eval_results_val_030501.json",
    ]
    files: List[Path] = []
    for pattern in patterns:
        files.extend(root.rglob(pattern))
    return sorted(set(files))


def load_results(json_path: Path) -> List[Dict[str, Any]]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "results" in data and isinstance(data["results"], list):
        return list(data["results"])
    raise ValueError(f"Unexpected JSON format in {json_path}")


def collect_fieldnames(results: Iterable[Dict[str, Any]]) -> List[str]:
    field_set = set()
    for row in results:
        field_set.update(row.keys())
    # Provide a stable, sensible ordering: common keys first
    preferred_order = [
        "model",
        "model_path",
        "episodes",
        "mean_reward",
        "std_reward",
        "min_reward",
        "max_reward",
        "mean_episode_length",
        "std_episode_length",
    ]
    ordered: List[str] = []
    for key in preferred_order:
        if key in field_set:
            ordered.append(key)
            field_set.remove(key)
    # Append any remaining keys in sorted order
    ordered.extend(sorted(field_set))
    return ordered


def write_csv(json_path: Path) -> Path:
    results = load_results(json_path)
    if not results:
        raise ValueError(f"No results found in {json_path}")

    fieldnames = collect_fieldnames(results)

    csv_path = json_path.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    return csv_path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    artifacts_root = project_root / "artifacts"

    if not artifacts_root.exists():
        raise SystemExit(f"artifacts directory not found at {artifacts_root}")

    json_files = find_eval_json_files(artifacts_root)
    if not json_files:
        raise SystemExit("No eval_results*.json files found under artifacts/")

    print("Converting JSON evaluation results to CSV:")
    for json_file in json_files:
        csv_path = write_csv(json_file)
        rel_json = json_file.relative_to(project_root)
        rel_csv = csv_path.relative_to(project_root)
        print(f"  {rel_json} -> {rel_csv}")


if __name__ == "__main__":
    main()
