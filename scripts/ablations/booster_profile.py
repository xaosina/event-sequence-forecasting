#!/usr/bin/env python3
"""BaryBooster cost ablation (rebuttal RPG6-W2). Two separate experiments,
both on the full test set (no batch limit), for LDM at horizon 32.

  sweep  -- TMS (OTD) and cDFS (Detection) vs number of booster samples N.
            Plain production booster; answers "how much gain survives at
            lower N" (Pareto).
  time   -- a single N=30 run (the paper setting) with the booster profiler on,
            splitting wall-time into base diffusion sampling vs the
            Hungarian / aggregation overhead.

  python scripts/ablations/booster_profile.py sweep --dataset age --device cuda:0
  python scripts/ablations/booster_profile.py time  --dataset age --device cuda:0
  python scripts/ablations/booster_profile.py collect --dataset age,gender,alphabattle
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "ablations"))

from run_claim import claim_run, release_run  # noqa: E402
from common import PAPER_ROOT, resolve_ckpt_resume, write_meta  # noqa: E402

SWEEP_BASE = REPO_ROOT / "log" / "ablations" / "booster" / "sweep"
TIME_BASE = REPO_ROOT / "log" / "ablations" / "booster" / "time"
MODEL = "ldm/vae"
BOOSTER = "wasserstein_barybooster1"
BOOSTER_PROFILE = "wasserstein_barybooster_profile"
N_GRID = (1, 5, 10, 15, 30)
PAPER_N = 30

# TMS = OTD (plain + log-amount variant per dataset).
OTD_METRICS = {
    "age": ["OTD", {"OTD": {"log_cols": ["amount_rur"]}}],
    "gender": ["OTD", {"OTD": {"log_cols": ["amount"]}}],
    "alphabattle": ["OTD"],
}


def parse_csv(raw: str) -> list[str]:
    return [c.strip() for c in raw.split(",") if c.strip()]


def sweep_metrics(dataset: str, horizon: int) -> list:
    """OTD (TMS) plus Detection (cDFS) so the Pareto shows both axes vs N."""
    detection = {"Detection": {"report_std": True, "condition_len": horizon, "verbose": False}}
    return OTD_METRICS[dataset] + [detection]


def _ldm_config(dataset: str, horizon: int):
    meta_path = PAPER_ROOT / dataset / f"horizon_{horizon}" / MODEL / "meta.yaml"
    if not meta_path.is_file():
        return None, None, None
    source_path = Path(str(yaml.safe_load(meta_path.read_text())["path"])).resolve()
    config_path = source_path / "seed_0" / "config.yaml"
    ckpt_resume = resolve_ckpt_resume(config_path, source_path)
    return source_path, config_path, ckpt_resume


def _run(out_dir: Path, log_dir: Path, run_name: str, config_path: Path,
         ckpt_resume, factory: str, metrics: list, n_samples: int,
         device: str, source_path: Path, extra_meta: dict,
         blim: int | None = None) -> None:
    print(f"run {run_name} -> {out_dir}")
    ckpt_args = ["--trainer.ckpt_resume", ckpt_resume] if ckpt_resume else []
    blim_args = ["--evaluator.blim", str(blim)] if blim else []
    try:
        subprocess.run(
            [
                "python", "main.py",
                "--config_path", str(config_path),
                "--log_dir", str(log_dir),
                "--run_name", run_name,
                "--runner.name", "GenerationEvaluator",
                "--runner.run_type", "simple",
                "--runner.params.n_runs", "1",
                "--runner.device_list", f"[{device}]",
                "--trainer.verbose", "True",
                "--overwrite_factory", factory,
                "--evaluator.booster.n_samples", str(n_samples),
                "--evaluator.metrics", json.dumps(metrics),
                "--evaluator.devices", f"[{device},{device},{device}]",
                "--device", device,
                *ckpt_args,
                *blim_args,
            ],
            check=True,
            cwd=str(REPO_ROOT),
        )
        results = out_dir / "results.csv"
        if not results.is_file():
            raise FileNotFoundError(f"expected results: {results}")
        shutil.copy(results, out_dir / "final.csv")
        write_meta(out_dir, source_path, extra=extra_meta)
    finally:
        release_run(out_dir)


def cmd_sweep(args: argparse.Namespace) -> None:
    n_grid = [int(n) for n in parse_csv(args.n)] if args.n else list(N_GRID)
    for dataset in parse_csv(args.dataset):
        source_path, config_path, ckpt = _ldm_config(dataset, args.horizon)
        if source_path is None:
            print(f"skip {dataset}: no ldm/vae meta")
            continue
        for n in n_grid:
            out_dir = SWEEP_BASE / dataset / f"N_{n}"
            if not claim_run(out_dir):
                print(f"skip {out_dir}")
                continue
            _run(
                out_dir, SWEEP_BASE / dataset, f"N_{n}", config_path, ckpt,
                f"[metrics/boosters/{BOOSTER}]", sweep_metrics(dataset, args.horizon),
                n, args.device, source_path, {"n_samples": n},
            )


def cmd_time(args: argparse.Namespace) -> None:
    for dataset in parse_csv(args.dataset):
        source_path, config_path, ckpt = _ldm_config(dataset, args.horizon)
        if source_path is None:
            print(f"skip {dataset}: no ldm/vae meta")
            continue
        run_name = f"N_{PAPER_N}" if args.horizon == 32 else f"h{args.horizon}_N_{PAPER_N}"
        out_dir = TIME_BASE / dataset / run_name
        if not claim_run(out_dir):
            print(f"skip {out_dir}")
            continue
        _run(
            out_dir, TIME_BASE / dataset, run_name, config_path, ckpt,
            f"[metrics/boosters/{BOOSTER_PROFILE}]", ["OTD"],
            PAPER_N, args.device, source_path, {"n_samples": PAPER_N, "blim": args.blim,
                                                "horizon": args.horizon},
            blim=args.blim,
        )


def cmd_collect(args: argparse.Namespace) -> None:
    for dataset in parse_csv(args.dataset):
        _collect_sweep(dataset)
        _collect_time(dataset)


def _collect_sweep(dataset: str) -> None:
    root = SWEEP_BASE / dataset
    rows: list[dict] = []
    for n_dir in sorted(root.glob("N_*")):
        final = n_dir / "final.csv"
        if not final.is_file():
            continue
        metrics = pd.read_csv(final, index_col=0)["mean"]
        row = {"N": int(n_dir.name.removeprefix("N_"))}
        for key in metrics.index:
            if key.startswith("OTD") or key.startswith("Detection"):
                row[key] = float(metrics[key])
        rows.append(row)
    if not rows:
        print(f"[{dataset}] sweep: nothing to collect")
        return
    df = pd.DataFrame(rows).sort_values("N")
    out = root / "summary.csv"
    df.to_csv(out, index=False)
    print(f"[{dataset}] sweep -> {out} ({len(df)} rows)")
    _plot(df, dataset, root / "pareto.png")


def _collect_time(dataset: str) -> None:
    prof = TIME_BASE / dataset / f"N_{PAPER_N}" / "booster_profile.csv"
    if not prof.is_file():
        return
    p = pd.read_csv(prof).iloc[0]
    overhead = p["total"] - p["sampling"]
    breakdown = {
        "dataset": dataset,
        "n_samples": PAPER_N,
        "sec_per_sample": p["sampling"] / p["n_generate"] if p["n_generate"] else 0.0,
        "t_sampling": p["sampling"],
        "t_overhead": overhead,
        "t_matching": p["matching"],
        "overhead_frac": overhead / p["total"] if p["total"] else 0.0,
        "t_total": p["total"],
    }
    out = TIME_BASE / dataset / "timing.csv"
    pd.DataFrame([breakdown]).to_csv(out, index=False)
    print(
        f"[{dataset}] time -> {out}: sampling {p['sampling']:.1f}s "
        f"({100 * (1 - overhead / p['total']):.1f}%), overhead {overhead:.1f}s "
        f"(matching {p['matching']:.2f}s)"
    )


def _plot(df: pd.DataFrame, dataset: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if "OTD" not in df.columns:
        return
    det = next((c for c in df.columns if c.startswith("Detection")), None)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(df["N"], df["OTD"], "o-", color="#2563eb", label="TMS (OTD)")
    ax.set_xlabel("N samples")
    ax.set_ylabel("TMS (OTD)", color="#2563eb")
    if det is not None:
        ax2 = ax.twinx()
        ax2.plot(df["N"], df[det], "s--", color="#dc2626", label="cDFS")
        ax2.set_ylabel("cDFS", color="#dc2626")
    ax.set_title(f"BaryBooster vs N — {dataset}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[{dataset}] plot -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="BaryBooster cost / TMS-vs-N ablation.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_sweep = sub.add_parser("sweep", help="TMS/cDFS vs N on full test (Pareto)")
    p_sweep.add_argument("--dataset", default="age")
    p_sweep.add_argument("--device", default="cuda:0")
    p_sweep.add_argument("--horizon", type=int, default=32)
    p_sweep.add_argument("--n", default=None, help="N grid (default 1,5,10,15,30)")
    p_sweep.set_defaults(func=cmd_sweep)

    p_time = sub.add_parser("time", help="Single N=30 profiled run (stage timing)")
    p_time.add_argument("--dataset", default="age")
    p_time.add_argument("--device", default="cuda:0")
    p_time.add_argument("--horizon", type=int, default=32)
    p_time.add_argument("--blim", type=int, default=10)
    p_time.set_defaults(func=cmd_time)

    p_collect = sub.add_parser("collect", help="Collect sweep summary/plot + timing")
    p_collect.add_argument("--dataset", default="age")
    p_collect.set_defaults(func=cmd_collect)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
