#!/usr/bin/env python3
import argparse
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAIRS = ((4, 16), (4, 64), (16, 64))


def load_run_info(experiment_root):
    with open(experiment_root / "meta.yaml", encoding="utf-8") as f:
        meta = yaml.safe_load(f)

    base_path = Path(meta["path"])
    config_file = base_path / "seed_0" / "config.yaml"
    with open(config_file, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    ckpt_dir = base_path / "seed_0" / "ckpt"
    ckpts = list(ckpt_dir.glob("*.ckpt"))
    ckpt_from_cfg = (cfg.get("trainer", {})).get("ckpt_resume")

    assert len(ckpts) <= 1, len(ckpts)
    assert not (ckpts and ckpt_from_cfg), "checkpoint defined in both places"

    if ckpts:
        ckpt_file = ckpts[0]
    elif ckpt_from_cfg:
        ckpt_file = Path(ckpt_from_cfg)
    else:
        raise FileNotFoundError(f"no checkpoint in {ckpt_dir} and no trainer.ckpt_resume")

    if not ckpt_file.is_absolute():
        ckpt_file = (REPO_ROOT / ckpt_file).resolve()
    if not ckpt_file.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_file}")

    return config_file, ckpt_file


def get_experiment_root(dataset_root, model, horizon):
    return dataset_root / f"horizon_{horizon}" / model


def parse_pairs(raw: str | None) -> list[tuple[int, int]]:
    if raw is None:
        return list(DEFAULT_PAIRS)
    out: list[tuple[int, int]] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        a, _, b = chunk.partition(":")
        if not b:
            raise ValueError(f"bad pair {chunk!r}, expected train:eval")
        out.append((int(a.strip()), int(b.strip())))
    return out


def write_meta_and_final(exp_dir: Path) -> None:
    """Same pattern as scripts/run_baselines.sh: results.csv -> final.csv + meta.yaml."""
    exp_dir = exp_dir.resolve()
    src = exp_dir / "results.csv"
    if not src.is_file():
        raise FileNotFoundError(f"expected {src} after run")
    shutil.copy(src, exp_dir / "final.csv")

    date = datetime.now(ZoneInfo("Europe/Moscow")).strftime("%Y-%m-%d %H:%M:%S")
    meta_path = exp_dir / "meta.yaml"
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(
            f'path: "{exp_dir.as_posix()}/"\n'
            f'date: "{date}"\n'
            f"final: true\n"
        )


def run_cross_grid(
    dataset_root: Path,
    model: str,
    device: str = "cuda:0",
    pairs: list[tuple[int, int]] | None = None,
):
    dataset_root = Path(dataset_root).resolve()
    dataset = dataset_root.name
    pairs = pairs or list(DEFAULT_PAIRS)

    for train_h, eval_h in pairs:
        if eval_h <= train_h:
            print(f"skip pair ({train_h}, {eval_h}): eval horizon must exceed train")
            continue

        exp_root = get_experiment_root(dataset_root, model, train_h)
        if not exp_root.exists():
            print(f"skip missing experiment {exp_root}")
            continue

        config_file, ckpt_file = load_run_info(exp_root)

        run_name = f"horizon_{eval_h}/{model}_h{train_h}"
        out_dir = (dataset_root / run_name).resolve()
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
            "--overwrite_factory", f"[metrics/with_detection/{dataset},horizon/{eval_h}]",
            "--device", device,
        ]
        subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
        write_meta_and_final(out_dir)
        time.sleep(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root")
    parser.add_argument("--model", required=True, help="detpp | multitoken | ldm/vae")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--pairs",
        default=None,
        help="Comma-separated train:eval, e.g. 4:16,4:64,16:64",
    )
    args = parser.parse_args()

    pairs = parse_pairs(args.pairs)
    run_cross_grid(
        args.dataset_root,
        args.model,
        device=args.device,
        pairs=pairs,
    )


if __name__ == "__main__":
    main()
