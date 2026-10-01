#!/usr/bin/env python3
"""Clean inference-time measurement for Table 8 (rebuttal RPG6-W2 / MUu1-W4).

Re-runs evaluation for every paper model (plus LDM-WB) one job per GPU and
reads the wall time between the ``validation started`` and ``Sampling done.``
log markers, so the numbers are free of the multi-experiment-per-GPU bias of
the original table.

  python scripts/ablations/eval_time.py run --dataset age,gender,alphabattle --device cuda:0
  python scripts/ablations/eval_time.py collect
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "ablations"))

from run_claim import claim_run, release_run  # noqa: E402
from common import PAPER_ROOT, resolve_ckpt_resume, write_meta  # noqa: E402

BASE = REPO_ROOT / "log" / "ablations" / "eval_time"
DATASETS = ("age", "gender", "alphabattle")
HORIZON = 64
BOOSTER_N = 30
# run name -> (meta dir under horizon_<h>, booster factory or None)
MODELS = {
    "ar": ("ar", None),
    "multitoken": ("multitoken", None),
    "detpp": ("detpp", None),
    "cdiff": ("cdiff", None),
    "ldm": ("ldm/vae", None),
    "ldm_wb": ("ldm/vae", "wasserstein_barybooster1"),
}
TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")


def parse_csv(raw: str) -> list[str]:
    return [c.strip() for c in raw.split(",") if c.strip()]


def _model_config(dataset: str, meta_dir: str):
    meta_path = PAPER_ROOT / dataset / f"horizon_{HORIZON}" / meta_dir / "meta.yaml"
    if not meta_path.is_file():
        return None, None, None
    source_path = Path(str(yaml.safe_load(meta_path.read_text())["path"])).resolve()
    config_path = source_path / "seed_0" / "config.yaml"
    ckpt_resume = resolve_ckpt_resume(config_path, source_path)
    return source_path, config_path, ckpt_resume


def cmd_run(args: argparse.Namespace) -> None:
    for dataset in parse_csv(args.dataset):
        for run_name, (meta_dir, booster) in MODELS.items():
            source_path, config_path, ckpt = _model_config(dataset, meta_dir)
            if source_path is None:
                print(f"skip {dataset}/{run_name}: no meta")
                continue
            out_dir = BASE / dataset / run_name
            if not claim_run(out_dir):
                print(f"skip {out_dir}")
                continue
            print(f"run {dataset}/{run_name} -> {out_dir}")
            cmd = [
                "python", "main.py",
                "--config_path", str(config_path),
                "--log_dir", str(BASE / dataset),
                "--run_name", run_name,
                "--runner.name", "GenerationEvaluator",
                "--runner.run_type", "simple",
                "--runner.params.n_runs", "1",
                "--runner.device_list", f"[{args.device}]",
                "--trainer.verbose", "True",
                "--evaluator.metrics", json.dumps(["OTD"]),
                "--device", args.device,
            ]
            if ckpt:
                cmd += ["--trainer.ckpt_resume", ckpt]
            if booster:
                cmd += ["--overwrite_factory", f"[metrics/boosters/{booster}]",
                        "--evaluator.booster.n_samples", str(BOOSTER_N)]
            try:
                subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
                results = out_dir / "results.csv"
                if not results.is_file():
                    raise FileNotFoundError(f"expected results: {results}")
                shutil.copy(results, out_dir / "final.csv")
                write_meta(out_dir, source_path,
                           extra={"horizon": HORIZON, "booster": booster})
            finally:
                release_run(out_dir)


def _sampling_seconds(log_file: Path) -> float | None:
    start = done = None
    for line in log_file.read_text(errors="ignore").splitlines():
        if "validation started" in line:
            m = TS_RE.search(line)
            start = m.group(1) if m else start
        elif "Sampling done." in line:
            m = TS_RE.search(line)
            done = m.group(1) if m else done
    if not (start and done):
        return None
    fmt = "%Y-%m-%d %H:%M:%S,%f"
    return (datetime.strptime(done, fmt) - datetime.strptime(start, fmt)).total_seconds()


def cmd_collect(args: argparse.Namespace) -> None:
    rows = []
    for dataset in parse_csv(args.dataset):
        for run_name in MODELS:
            log_file = BASE / dataset / run_name / "seed_0" / "log"
            if not log_file.is_file():
                print(f"[{dataset}/{run_name}] no log")
                continue
            sec = _sampling_seconds(log_file)
            if sec is None:
                print(f"[{dataset}/{run_name}] markers not found")
                continue
            rows.append({"dataset": dataset, "model": run_name,
                         "seconds": round(sec, 1), "minutes": round(sec / 60, 2)})
    if not rows:
        print("nothing collected")
        return
    df = pd.DataFrame(rows)
    out = BASE / "summary.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"-> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean per-model inference timing.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="evaluate every model, one job per GPU")
    p_run.add_argument("--dataset", default=",".join(DATASETS))
    p_run.add_argument("--device", default="cuda:0")
    p_run.set_defaults(func=cmd_run)

    p_collect = sub.add_parser("collect", help="parse logs into summary.csv")
    p_collect.add_argument("--dataset", default=",".join(DATASETS))
    p_collect.set_defaults(func=cmd_collect)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
