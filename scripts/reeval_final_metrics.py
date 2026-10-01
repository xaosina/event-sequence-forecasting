#!/usr/bin/env python3
import argparse
import csv
import shutil
import subprocess
from pathlib import Path

import yaml

HORIZONS = (4, 16, 32, 64)
REPO_ROOT = Path(__file__).resolve().parents[1]


def read_metric_map(csv_path: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("") or "").strip()
            if not name:
                continue
            mean_raw = row.get("mean")
            if mean_raw in (None, ""):
                continue
            metrics[name] = float(mean_raw)
    return metrics


def compare_metrics(
    old: dict[str, float],
    new: dict[str, float],
    tolerance: float,
) -> tuple[int, list[str], list[str], list[tuple[str, float, float]]]:
    old_only = sorted(set(old) - set(new))
    new_only = sorted(set(new) - set(old))
    changed: list[tuple[str, float, float]] = []

    for key in sorted(set(old) & set(new)):
        old_val = old[key]
        new_val = new[key]
        if abs(old_val - new_val) > tolerance:
            changed.append((key, old_val, new_val))

    return len(set(old) & set(new)), old_only, new_only, changed


def infer_run_name(run_path: Path, dataset_root: Path) -> str:
    run_path = run_path.resolve()
    dataset_root = dataset_root.resolve()

    try:
        return run_path.relative_to(dataset_root).as_posix()
    except ValueError:
        pass

    # Fallback: split by "<dataset>/<...>" when path cannot be relativized directly.
    parts = list(run_path.parts)
    if dataset_root.name in parts:
        idx = parts.index(dataset_root.name)
        if idx + 1 < len(parts):
            return Path(*parts[idx + 1 :]).as_posix()

    raise ValueError(f"cannot infer run_name from {run_path}")


def reeval_one_horizon(dataset_root: Path, model: str, horizon: int, device: str, tolerance: float) -> None:
    exp_dir = dataset_root / f"horizon_{horizon}" / model
    meta_path = exp_dir / "meta.yaml"
    final_path = exp_dir / "final.csv"

    if not meta_path.is_file() or not final_path.is_file():
        print(f"[h={horizon}] skip {exp_dir}: missing meta.yaml or final.csv")
        return

    with open(meta_path, encoding="utf-8") as f:
        meta = yaml.safe_load(f) or {}

    run_path_raw = meta.get("path")
    if not run_path_raw:
        print(f"[h={horizon}] skip {exp_dir}: meta.yaml has no 'path'")
        return

    run_path = Path(run_path_raw).resolve()
    config_path = run_path / "seed_0" / "config.yaml"
    results_path = run_path / "results.csv"

    if not config_path.is_file():
        print(f"[h={horizon}] skip {exp_dir}: missing config at {config_path}")
        return

    old_metrics = read_metric_map(final_path)
    run_name = infer_run_name(run_path, dataset_root)
    dataset = dataset_root.name

    cmd = [
        "python",
        "main.py",
        "--config_path",
        str(config_path),
        "--run_name",
        run_name,
        "--runner.name",
        "GenerationEvaluator",
        "--trainer.verbose",
        "True",
        "--overwrite_factory",
        f"[metrics/with_detection/{dataset},horizon/{horizon}]",
        "--device",
        device,
    ]
    print(f"[h={horizon}] run_name={run_name}")
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))

    if not results_path.is_file():
        raise FileNotFoundError(f"[h={horizon}] expected results after rerun: {results_path}")

    new_metrics = read_metric_map(results_path)
    common_count, old_only, new_only, changed = compare_metrics(old_metrics, new_metrics, tolerance)
    print(
        f"[h={horizon}] debug: common={common_count}, changed={len(changed)}, "
        f"old_only={len(old_only)}, new_only={len(new_only)}"
    )
    if changed:
        print(f"[h={horizon}] changed metrics (up to 10):")
        for key, old_val, new_val in changed[:10]:
            print(f"  - {key}: old={old_val:.10g} new={new_val:.10g} diff={abs(old_val - new_val):.3g}")
    if old_only:
        print(f"[h={horizon}] only in old final.csv (up to 10): {old_only[:10]}")
    if new_only:
        print(f"[h={horizon}] only in new results.csv (up to 10): {new_only[:10]}")

    shutil.copy(results_path, final_path)
    print(f"[h={horizon}] updated {final_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rerun evaluator for selected runs from meta.yaml and refresh final.csv."
    )
    parser.add_argument("dataset_root", help="e.g. log/gen-paper/age")
    parser.add_argument("--model", required=True, help="model folder name under each horizon")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tolerance", type=float, default=1e-9)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    for horizon in HORIZONS:
        reeval_one_horizon(
            dataset_root=dataset_root,
            model=args.model,
            horizon=horizon,
            device=args.device,
            tolerance=args.tolerance,
        )


if __name__ == "__main__":
    main()
