#!/usr/bin/env python3
import argparse
import re
import subprocess
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_ROOT = REPO_ROOT / "log" / "gen-paper"
# python scripts/rerun_with_seeds.py log/gen-paper/age/horizon_32/ldm/vae --n_runs 3 --device_list 1,2,5

def infer_dataset_and_horizon(exp_dir: Path) -> tuple[str, int]:
    exp_dir = exp_dir.resolve()
    rel = exp_dir.relative_to(PAPER_ROOT.resolve())
    parts = rel.parts
    if len(parts) < 2:
        raise ValueError(f"unexpected experiment path layout: {exp_dir}")

    dataset = parts[0]
    horizon_part = parts[1]
    match = re.fullmatch(r"horizon_(\d+)", horizon_part)
    if not match:
        raise ValueError(f"cannot parse horizon from '{horizon_part}'")
    horizon = int(match.group(1))
    return dataset, horizon


def infer_run_name(run_path: Path, dataset_root: Path) -> str:
    run_path = run_path.resolve()
    dataset_root = dataset_root.resolve()
    return run_path.relative_to(dataset_root).as_posix()


def parse_device_list(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    if raw.strip().lower() == "none":
        return None
    device_ids = [device_id.strip() for device_id in raw.split(",") if device_id.strip()]
    if not device_ids:
        return None
    if not all(device_id.isdigit() for device_id in device_ids):
        raise ValueError("--device_list must contain only numeric ids, e.g. 0,1,2")
    return [f"cuda:{device_id}" for device_id in device_ids]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rerun best experiment from meta.yaml with multiple seeds. "
            "Writes one output CSV next to meta.yaml."
        )
    )
    parser.add_argument(
        "exp_path",
        help="Path to experiment directory containing meta.yaml (e.g. log/gen-paper/age/horizon_4/ldm/vae)",
    )
    parser.add_argument("--n_runs", type=int, default=5, help="Number of runs (default: 5)")
    parser.add_argument(
        "--device_list",
        required=True,
        help="Comma-separated CUDA ids (e.g. 0,1,2).",
    )
    args = parser.parse_args()

    exp_dir = Path(args.exp_path).resolve()
    meta_path = exp_dir / "meta.yaml"
    if not meta_path.is_file():
        raise FileNotFoundError(f"missing meta.yaml: {meta_path}")

    with open(meta_path, encoding="utf-8") as f:
        meta = yaml.safe_load(f) or {}
    run_path_raw = meta.get("path")
    if not run_path_raw:
        raise ValueError(f"{meta_path} does not contain 'path'")

    run_path = Path(run_path_raw).resolve()
    config_path = run_path / "seed_0" / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing config: {config_path}")

    dataset, horizon = infer_dataset_and_horizon(exp_dir)
    dataset_root = PAPER_ROOT / dataset
    exp_rel_path = exp_dir.resolve().relative_to(dataset_root.resolve()).as_posix()
    run_name = f"{exp_rel_path}/{args.n_runs}_runs"
    device_list = parse_device_list(args.device_list)
    if device_list is None:
        raise ValueError("--device_list must be provided with at least one numeric CUDA id")
    cmd = [
        "python", "main.py",
        "--config_path", str(config_path),
        "--run_name", run_name,
        "--trainer.ckpt_resume", "null",
        "--runner.name", "GenerationTrainer",
        "--runner.run_type", "simple",
        "--runner.params.n_runs", str(args.n_runs),
        "--trainer.verbose", "True",
        "--overwrite_factory", f"[metrics/with_detection/{dataset},horizon/{horizon}]",
        "--runner.device_list", f"[{','.join(device_list)}]",
        "--runner.params.n_workers", str(len(device_list)),
    ]
    print(f"running evaluator for {run_name} with n_runs={args.n_runs}")
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))

if __name__ == "__main__":
    main()
