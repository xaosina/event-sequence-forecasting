#!/usr/bin/env python3
"""Cross-model generation figures: `run` generates samples, `plot` draws them.

`run` evaluates every model on a single batch (configs_paper/metrics/visual/) and
keeps the samples parquet; `plot` stacks Original + one row per model. The test
loader is unshuffled with a fixed seed, so a pinned batch_size gives every model
the same users -- which is what makes the rows comparable.

  python scripts/visuals.py run log/gen-paper/age --horizon 32 --device cuda:4
  python scripts/visuals.py plot log/gen-paper/age --horizon 32 --users 40377
"""

import argparse
import hashlib
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from dacite import Config, from_dict

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from generation.data.data_types import DataConfig  # noqa: E402
from generation.metrics.metrics import draw_sequence, visual_cat2col  # noqa: E402
from grid_cross_horizon import load_run_info  # noqa: E402
from run_claim import claim_run, release_run  # noqa: E402

MODELS = {
    "ldm/vae": "LDM",
    "ldm/vae/booster/wasserstein_barybooster1": "LDM-WB",
    "detpp": "DEF",
    "cdiff": "DDM",
    "multitoken": "MT",
    "ar": "AR",
    "baselines/mode": "Mode",
    "baselines/repeat": "Repeat",
    "baselines/hist_sampler": "Hist.",
}

BATCH_SIZE = 32

# The client shown in the paper figure; pass --users all to browse the whole batch.
DEFAULT_USER = 20132


def resolve_run(experiment_root: Path) -> tuple[Path, Path | None]:
    """Baseline meta.yaml points at itself and has no ckpt/, so load_run_info fails."""
    if "baselines/" in experiment_root.as_posix():
        config_file = experiment_root / "seed_0" / "config.yaml"
        if not config_file.is_file():
            raise FileNotFoundError(f"missing baseline config: {config_file}")
        return config_file, None
    return load_run_info(experiment_root)


def run_models(dataset_root: Path, horizon: int, device: str, models: list[str]) -> None:
    for model in models:
        source, _, booster = model.partition("/booster/")
        experiment_root = dataset_root / f"horizon_{horizon}" / source
        if not experiment_root.exists():
            print(f"skip missing experiment {experiment_root}")
            continue
        try:
            config_file, ckpt_file = resolve_run(experiment_root)
        except FileNotFoundError as exc:
            print(f"skip {model}: {exc}")
            continue

        run_name = f"visuals/horizon_{horizon}/{model}"
        out_dir = dataset_root / run_name
        if not claim_run(out_dir):
            print(f"skip {out_dir}")
            continue

        factory = f"metrics/visual/{dataset_root.name}"
        if booster:
            factory += f",metrics/boosters/{booster}"

        print(f"run {run_name}")
        try:
            cmd = ["python", "main.py", "--config_path", str(config_file)]
            if ckpt_file is not None:
                cmd += ["--trainer.ckpt_resume", str(ckpt_file)]
            cmd += [
                "--run_name", run_name,
                "--log_dir", str(dataset_root),
                "--runner.name", "GenerationEvaluator",
                "--runner.run_type", "simple",
                "--runner.params.n_runs", "1",
                "--trainer.verbose", "True",
                "--overwrite_factory", f"[{factory}]",
                "--device", device,
                "--runner.device_list", f"[{device}]",
                "--data_conf.batch_size", str(BATCH_SIZE),
            ]
            subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
        finally:
            release_run(out_dir)


def test_samples(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Val is evaluated before test, so the second evaluation dir holds test."""
    keyed = []
    for path in run_dir.glob("evaluation*"):
        match = re.fullmatch(r"evaluation(?:\((\d+)\))?", path.name)
        if match and path.is_dir():
            keyed.append((int(match.group(1) or 0), path))
    if len(keyed) < 2:
        raise FileNotFoundError(f"no test evaluation in {run_dir}")
    base = sorted(keyed)[1][1] / "samples"
    return pd.read_parquet(base / "gt"), pd.read_parquet(base / "gen")


def fingerprint(df: pd.DataFrame, data_conf: DataConfig) -> str:
    """float64 normalises dtype noise, so a mismatch means the values differ."""
    parts = [np.asarray(df.index, dtype=np.int64).tobytes()]
    for col in (data_conf.time_name, data_conf.target_token):
        parts += [np.asarray(arr, dtype=np.float64).tobytes() for arr in df[col]]
    return hashlib.sha1(b"".join(parts)).hexdigest()


def check_same_users(model, ref_model, gt_ref, gt, gen, data_conf) -> None:
    assert gen.index.equals(gt.index), (
        f"{model}: generated and ground-truth client_ids disagree within the run"
    )
    assert gt.index.is_unique, (
        f"{model}: duplicate client_id in the batch (val_resamples > 1?); "
        "plotting would concatenate several sequences for one user"
    )
    assert gt_ref.index.equals(gt.index), (
        f"{model} evaluated different users (or a different order) than {ref_model}: "
        f"{gt_ref.index.tolist()[:5]}... vs {gt.index.tolist()[:5]}... -- "
        "check that batch_size and the dataset config match across runs"
    )
    assert fingerprint(gt_ref, data_conf) == fingerprint(gt, data_conf), (
        f"{model} has the same client_ids as {ref_model} but different ground-truth "
        "sequences -- the runs used different data configs (horizon? random_end?)"
    )


def collect(dataset_root: Path, horizon: int, models: list[str]):
    gt_ref = data_conf = ref_model = None
    frames: dict[str, pd.DataFrame] = {}

    for model in models:
        run_dir = dataset_root / "visuals" / f"horizon_{horizon}" / model / "seed_0"
        if not run_dir.is_dir():
            print(f"skip {model}: no run at {run_dir}")
            continue
        try:
            gt, gen = test_samples(run_dir)
        except FileNotFoundError as exc:
            print(f"skip {model}: {exc}")
            continue

        gt, gen = gt.set_index("client_id"), gen.set_index("client_id")
        if gt_ref is None:
            with open(run_dir / "config.yaml", encoding="utf-8") as f:
                data_conf = from_dict(
                    DataConfig, yaml.safe_load(f)["data_conf"], Config(strict=False)
                )
            gt_ref, ref_model = gt, model
        check_same_users(model, ref_model, gt_ref, gt, gen, data_conf)
        frames[model] = gen

    if gt_ref is None:
        raise SystemExit("no runs found; run `visuals.py run` first")
    print(f"verified {len(frames)} models on the same {len(gt_ref)} users")
    return gt_ref, frames, data_conf


def plot_users(dataset_root: Path, horizon: int, models: list[str], users, fmt: str):
    gt, frames, data_conf = collect(dataset_root, horizon, models)
    users = users or gt.index.tolist()
    missing = [u for u in users if u not in gt.index]
    if missing:
        raise SystemExit(f"client_id not in the evaluated batch: {missing}")

    out_dir = dataset_root / "visuals" / f"horizon_{horizon}" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 16, "axes.labelsize": 20, "axes.titlesize": 20})
    cat2col = visual_cat2col(gt, data_conf)
    panels = [("Original", gt)] + [(MODELS.get(m, m), f) for m, f in frames.items()]
    nrows = -(-len(panels) // 2)

    for user in users:
        fig, axs = plt.subplots(
            figsize=(18, 2.0 * nrows), nrows=nrows, ncols=2, sharex=True, sharey=True
        )
        flat = list(axs.T.flat)  # column-major, so the left column fills first
        for ax, (label, df) in zip(flat, panels):
            draw_sequence(ax, df, user, data_conf, cat2col)
            ax.set_title(label)
        for ax in flat[len(panels) :]:
            ax.axis("off")
        for col in range(2):
            axs[-1, col].set_xlabel("Time")
        fig.suptitle(
            f"Sample visualization for {data_conf.dataset_name.upper()}, horizon={data_conf.generation_len}",
            size=26,
        )
        fig.supylabel("Log amount", size=20)
        fig.tight_layout()
        out = out_dir / f"{user}.{fmt}"
        fig.savefig(out, bbox_inches="tight", **({"dpi": 150} if fmt == "png" else {}))
        plt.close(fig)
        print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "plot"))
    parser.add_argument("dataset_root", help="e.g. log/gen-paper/age")
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model", default=None, help="Comma-separated; default is all")
    parser.add_argument(
        "--users", default=None, help="Comma-separated client_id, or 'all'"
    )
    parser.add_argument("--format", default="pdf", choices=("png", "pdf"))
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    models = args.model.split(",") if args.model else list(MODELS)
    if args.command == "run":
        run_models(dataset_root, args.horizon, args.device, models)
    else:
        if args.users == "all":
            users = None
        else:
            users = [int(u) for u in args.users.split(",")] if args.users else [DEFAULT_USER]
        plot_users(dataset_root, args.horizon, models, users, args.format)


if __name__ == "__main__":
    main()
