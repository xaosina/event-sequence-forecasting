#!/usr/bin/env python3
"""cDFS detector-architecture ablation (rebuttal RPG6-W3 / rDYr-W3 / MUu1-W5).

Re-evaluates the paper's generators against several detector architectures at
once (see ``configs_paper/metrics/ablations/detector/*``: gru_512_2l reference,
gru_1024_2l, transformer_512_2l, transformer_1024_3l) to show the model ranking
by cDFS is stable under a stronger / different discriminator. Horizon 32 by
default, all datasets. The Wasserstein booster runs on ``ldm/vae`` only.

  python scripts/ablations/detector_ablation.py --dataset age --device cuda:0
  python scripts/ablations/detector_ablation.py --dataset gender --models ldm/vae
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "ablations"))

from run_claim import claim_run, release_run  # noqa: E402
from common import PAPER_ROOT, resolve_ckpt_resume, write_meta  # noqa: E402

OUTPUT_BASE = REPO_ROOT / "log" / "ablations" / "detector"

BASE_MODELS = (
    "ldm/vae",
    "cdiff",
    "ar",
    "multitoken",
    "detpp",
    "baselines/gt",
    "baselines/hist_sampler",
    "baselines/mode",
    "baselines/repeat",
)
BOOSTER = "wasserstein_barybooster1"
BOOSTER_MODELS = ("ldm/vae",)


def reeval_one(
    dataset: str,
    horizon: int,
    model: str,
    device: str,
    booster: str | None = None,
) -> None:
    """Re-evaluate a gen-paper run with the detector-ablation metric set."""
    factory = f"[metrics/ablations/detector/{dataset},horizon/{horizon}"
    if booster:
        run_name = f"horizon_{horizon}/{model}/booster/{booster}"
        factory += f",metrics/boosters/{booster}]"
    else:
        run_name = f"horizon_{horizon}/{model}"
        factory += "]"

    out_dir = OUTPUT_BASE / dataset / run_name
    if not claim_run(out_dir):
        print(f"[h={horizon}] skip {out_dir}")
        return

    meta_path = PAPER_ROOT / dataset / f"horizon_{horizon}" / model / "meta.yaml"
    if not meta_path.is_file():
        print(f"[h={horizon}] skip {model}: no meta.yaml")
        release_run(out_dir)
        return

    with open(meta_path, encoding="utf-8") as f:
        run_path_raw = (yaml.safe_load(f) or {}).get("path")
    if not run_path_raw:
        print(f"[h={horizon}] skip {model}: meta.yaml has no path")
        release_run(out_dir)
        return

    source_path = Path(str(run_path_raw)).resolve()
    config_path = source_path / "seed_0" / "config.yaml"
    if not config_path.is_file():
        print(f"[h={horizon}] skip {model}: missing config at {config_path}")
        release_run(out_dir)
        return

    ckpt_resume = resolve_ckpt_resume(config_path, source_path)
    ckpt_args = ["--trainer.ckpt_resume", ckpt_resume] if ckpt_resume else []

    label = f"{model}/booster/{booster}" if booster else model
    print(f"[h={horizon}] start {label} -> {out_dir}")
    try:
        subprocess.run(
            [
                "python", "main.py",
                "--config_path", str(config_path),
                "--log_dir", str(OUTPUT_BASE / dataset),
                "--run_name", run_name,
                "--runner.name", "GenerationEvaluator",
                "--trainer.verbose", "True",
                "--overwrite_factory", factory,
                "--device", device,
                *ckpt_args,
            ],
            check=True,
            cwd=str(REPO_ROOT),
        )
        results_path = out_dir / "results.csv"
        if not results_path.is_file():
            raise FileNotFoundError(f"expected results after rerun: {results_path}")
        shutil.copy(results_path, out_dir / "final.csv")
        write_meta(out_dir, source_path)
        print(f"[h={horizon}] updated {out_dir / 'final.csv'}")
    finally:
        release_run(out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="cDFS detector-architecture ablation.")
    parser.add_argument("--dataset", default="age")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument(
        "--models",
        default=",".join(BASE_MODELS),
        help="comma-separated base models (booster runs on ldm/vae regardless)",
    )
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for model in models:
        reeval_one(args.dataset, args.horizon, model, args.device)
    for model in BOOSTER_MODELS:
        if model in models:
            reeval_one(args.dataset, args.horizon, model, args.device, booster=BOOSTER)


if __name__ == "__main__":
    main()
