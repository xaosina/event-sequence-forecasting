#!/usr/bin/env python3
import argparse
import subprocess
import time
from pathlib import Path

from grid_cross_horizon import REPO_ROOT, load_run_info, write_meta_and_final
from run_claim import claim_run, release_run

HORIZONS = (4, 16, 32, 64)
DEFAULT_MODELS = ("ldm/vae", "cdiff", "ar", "multitoken", "detpp")
MODELS_TEMPERATURE_ONE = frozenset({"ar", "multitoken", "detpp"})
# Examples:
# python scripts/grid_booster.py log/gen-paper/age --device cuda:1
# python scripts/grid_booster.py log/gen-paper/gender --device cuda:2
# python scripts/grid_booster.py log/gen-paper/alphabattle --device cuda:3


def parse_csv_values(raw: str | None, cast):
    if raw is None:
        return None
    values = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(cast(chunk))
    return values


def load_seed_run_info(experiment_root: Path, source_run: str, seed: int):
    base_path = experiment_root / source_run / f"seed_{seed}"
    config_file = base_path / "config.yaml"
    ckpt_dir = base_path / "ckpt"
    ckpts = sorted(ckpt_dir.glob("*.ckpt"))
    if not config_file.is_file():
        raise FileNotFoundError(f"missing config for seed {seed}: {config_file}")
    if len(ckpts) != 1:
        raise FileNotFoundError(
            f"expected exactly one checkpoint for seed {seed} in {ckpt_dir}, got {len(ckpts)}"
        )
    return config_file, ckpts[0]


def detect_seeds(experiment_root: Path, source_run: str) -> list[int]:
    source_dir = experiment_root / source_run
    if not source_dir.is_dir():
        return []
    seeds: list[int] = []
    for seed_dir in sorted(source_dir.glob("seed_*")):
        suffix = seed_dir.name.removeprefix("seed_")
        if suffix.isdigit():
            seeds.append(int(suffix))
    return seeds


def run_booster_grid(
    dataset_root: Path,
    device: str = "cuda:0",
    horizons: list[int] | None = None,
    models: list[str] | None = None,
    booster: str = "wasserstein_barybooster1",
    source_run: str | None = None,
    seeds: list[int] | None = None,
    output_root: Path | None = None,
    overwrite_factory: str | None = None,
    extra_args: list[str] | None = None,
) -> None:
    dataset_root = Path(dataset_root).resolve()
    dataset = dataset_root.name
    output_root = Path(output_root).resolve() if output_root else dataset_root
    horizons = list(HORIZONS) if horizons is None else horizons
    models = list(DEFAULT_MODELS) if models is None else models
    for horizon in horizons:
        for model in models:
            experiment_root = dataset_root / f"horizon_{horizon}" / model
            if not experiment_root.exists():
                print(f"skip missing experiment {experiment_root}")
                continue

            if source_run is None:
                run_seeds = [0] if seeds is None else seeds
            else:
                run_seeds = detect_seeds(experiment_root, source_run) if seeds is None else seeds
                if not run_seeds:
                    print(f"skip {experiment_root}: no seed_* found in {source_run}")
                    continue

            for seed in run_seeds:
                if source_run is None:
                    if seed != 0:
                        raise ValueError(
                            "seed != 0 requires --source_run (e.g. --source_run 3_runs)"
                        )
                    config_file, ckpt_file = load_run_info(experiment_root)
                else:
                    config_file, ckpt_file = load_seed_run_info(experiment_root, source_run, seed)

                if source_run is None and seed == 0:
                    run_name = f"horizon_{horizon}/{model}/booster/{booster}"
                else:
                    run_name = f"horizon_{horizon}/{model}/{source_run}_booster/{booster}/seed_{seed}"

                out_dir = output_root / run_name
                if not claim_run(out_dir):
                    print(f"skip {out_dir}")
                    continue

                if overwrite_factory is None:
                    factory = f"[metrics/with_detection/{dataset},metrics/boosters/{booster}]"
                else:
                    factory = overwrite_factory.format(
                        dataset=dataset, horizon=horizon, booster=booster
                    )

                print(f"run {run_name}")
                try:
                    cmd = [
                        "python", "main.py",
                        "--config_path", str(config_file),
                        "--trainer.ckpt_resume", str(ckpt_file),
                        "--run_name", run_name,
                        "--log_dir", str(output_root),
                        "--runner.name", "GenerationEvaluator",
                        "--runner.run_type", "simple",
                        "--runner.params.n_runs", "1",
                        "--trainer.verbose", "True",
                        "--overwrite_factory", factory,
                        "--device", device,
                        "--runner.device_list", f"[{device}]",
                    ]
                    if model in MODELS_TEMPERATURE_ONE:
                        cmd.extend(
                            [
                                "--evaluator.topk", "-1",
                                "--evaluator.temperature", "1",
                            ]
                        )
                    if extra_args:
                        cmd.extend(extra_args)
                    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
                    write_meta_and_final(out_dir)
                finally:
                    release_run(out_dir)
                time.sleep(2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", help="e.g. log/gen-paper/age")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--horizon",
        default=None,
        help="Comma-separated horizon values, e.g. --horizon 4,16,64",
    )
    parser.add_argument(
        "--model",
        default=None,
        help='Comma-separated model names; default includes ldm/vae, cdiff, ar, multitoken, detpp',
    )
    parser.add_argument(
        "--booster",
        default="wasserstein_barybooster1",
        help="Booster config name from configs_paper/metrics/boosters",
    )
    parser.add_argument(
        "--source_run",
        default=None,
        help="Run folder under each model for multirun checkpoints, e.g. 3_runs; auto-detects seed_* by default",
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated seed ids to run, e.g. --seeds 1 or --seeds 0,2",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Write runs here instead of dataset_root (e.g. log/ablations/mean_amount/age)",
    )
    parser.add_argument(
        "--overwrite-factory",
        default=None,
        help="Format string with {dataset},{horizon},{booster} for --overwrite_factory",
    )
    args = parser.parse_args()

    horizons = parse_csv_values(args.horizon, cast=int)
    models = parse_csv_values(args.model, cast=str)
    seeds = parse_csv_values(args.seeds, cast=int)
    run_booster_grid(
        args.dataset_root,
        device=args.device,
        horizons=horizons,
        models=models,
        booster=args.booster,
        source_run=args.source_run,
        seeds=seeds,
        output_root=Path(args.output_root) if args.output_root else None,
        overwrite_factory=args.overwrite_factory,
    )


if __name__ == "__main__":
    main()
