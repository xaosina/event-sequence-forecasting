#!/usr/bin/env python3
import argparse
import subprocess
import time
from pathlib import Path

import yaml

PAPER_ROOT = "log/gen-paper/"
CFGS = (0, 1, 1.2, 1.3, 1.4, 1.5, 2, 3, 4, 5)

def parse_cfgs(raw):
    if raw is None:
        return CFGS
    vals = []
    for x in raw.split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(float(x))
    return tuple(vals)


def load_run_root(experiment_root):
    with open(experiment_root / "meta.yaml") as f:
        meta = yaml.safe_load(f)
    return Path(meta["path"])

def run_cfg_grid(root, device="cuda:0", cfgs=None):
    root = Path(root)
    exp_root = load_run_root(root)
    cfgs = CFGS if cfgs is None else tuple(cfgs)

    marker = f"{PAPER_ROOT.rstrip('/')}/"
    root_str = str(root).rstrip("/")
    if marker not in root_str:
        raise ValueError(f"path must contain '{PAPER_ROOT}': {root}")
    rel_path = root_str.split(marker, 1)[1]
    dataset, _, tail = rel_path.partition("/")

    config_path = exp_root / "seed_0/config.yaml"
    ckpt_dir = exp_root / "seed_0/ckpt"
    ckpts = sorted(ckpt_dir.glob("*.ckpt"))
    assert len(ckpts) == 1, (ckpts, ckpt_dir)
    ckpt = ckpts[0]

    dataset_root = Path(PAPER_ROOT.rstrip("/")) / dataset
    for cfg in cfgs:
        prefix = f"{tail}/" if tail else ""
        run_name = f"{prefix}cfg/{cfg}"
        out_dir = dataset_root / run_name
        if out_dir.exists():
            print(f"skip existing {out_dir}")
            continue

        print("Running:", run_name)

        subprocess.run(
            [
                "python", "main.py",
                "--config_path", str(config_path),
                "--trainer.ckpt_resume", str(ckpt),
                "--run_name", run_name,
                "--runner.name", "GenerationEvaluator",
                "--runner.run_type", "simple",
                "--runner.params.n_runs", "1",
                "--overwrite_factory", f"[metrics/with_detection/{dataset},metrics/cfg/{cfg}]",
                "--device", device,
            ],
            check=True,
        )
        time.sleep(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_root")
    parser.add_argument("device", nargs="?", default="cuda:0")
    parser.add_argument(
        "--cfg",
        default=None,
        help="Comma-separated cfg values, e.g. --cfg 3 or --cfg 1,3,5",
    )
    args = parser.parse_args()
    run_cfg_grid(args.experiment_root, device=args.device, cfgs=parse_cfgs(args.cfg))


if __name__ == "__main__":
    main()
