import argparse
import subprocess
import time
from pathlib import Path

import yaml

HORIZONS = [4, 16, 32, 64]
TEMPS = [0, 0.1, 0.4, 0.5, 0.6, 1., 1.2, 1.5, 1.6, 2.]

GLOBAL_MODELS = ["ar"]
HORIZON_MODELS = ["multitoken", "detpp"]


def load_run_info(experiment_root):
    with open(experiment_root / "meta.yaml") as f:
        meta = yaml.safe_load(f)

    base_path = Path(meta["path"])
    config_file = base_path / "seed_0" / "config.yaml"
    ckpt_dir = base_path / "seed_0" / "ckpt"
    ckpts = list(ckpt_dir.glob("*.ckpt"))

    if not ckpts:
        raise FileNotFoundError(f"no checkpoint in {ckpt_dir}")
    assert len(ckpts) == 1, len(ckpts)

    return config_file, ckpts[0]


def get_experiment_root(dataset_root, model, horizon):
    if model in GLOBAL_MODELS:
        return dataset_root / model
    if model in HORIZON_MODELS:
        return dataset_root / f"horizon_{horizon}" / model
    raise ValueError(model)


def parse_csv_values(raw, cast=float):
    if raw is None:
        return None
    vals = []
    for x in raw.split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(cast(x))
    return vals


def run_temp_grid(dataset_root, model, device="cuda:0", horizons=None, temps=None):
    dataset_root = Path(dataset_root)
    dataset = dataset_root.name
    horizons = HORIZONS if horizons is None else horizons
    temps = TEMPS if temps is None else temps

    for horizon in horizons:
        experiment_root = get_experiment_root(dataset_root, model, horizon)
        if not experiment_root.exists():
            print(f"skip {experiment_root}")
            continue

        config_file, ckpt_file = load_run_info(experiment_root)

        for temp in temps:
            run_name = f"horizon_{horizon}/{model}/temp/{temp}"
            out_dir = dataset_root / run_name
            if out_dir.exists():
                print(f"skip existing {out_dir}")
                continue

            print(f"run {run_name}")

            cmd = [
                "python", "main.py",
                "--config_path", str(config_file),
                "--trainer.ckpt_resume", str(ckpt_file),
                "--run_name", run_name,
                "--runner.name", "GenerationEvaluator",
                "--trainer.verbose", "True",
                "--evaluator.topk", "-1",
                "--evaluator.temperature", str(temp),
                "--overwrite_factory", f"[metrics/with_detection/{dataset},horizon/{horizon}]",
                "--device", device,
            ]
            subprocess.run(cmd, check=True)
            time.sleep(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root")
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--horizon",
        default=None,
        help="Comma-separated horizon values, e.g. --horizon 64 or --horizon 4,16",
    )
    parser.add_argument(
        "--temp",
        default=None,
        help="Comma-separated temperature values, e.g. --temp 0.4 or --temp 0.1,0.4",
    )
    args = parser.parse_args()

    horizons = parse_csv_values(args.horizon, cast=int)
    temps = parse_csv_values(args.temp, cast=float)
    run_temp_grid(
        args.dataset_root,
        args.model,
        args.device,
        horizons=horizons,
        temps=temps,
    )


if __name__ == "__main__":
    main()