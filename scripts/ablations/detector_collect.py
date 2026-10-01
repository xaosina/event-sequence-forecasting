#!/usr/bin/env python3
"""Collect the detector-architecture ablation into a cDFS table + ranking check.

Reads ``log/ablations/detector/<dataset>/horizon_32/<model>/final.csv`` and
builds a models x detectors matrix of cDFS. The rebuttal claim (RPG6-W3 /
rDYr-W3 / MUu1-W5) is that the *ranking* of models is stable across detector
capacity/architecture, so we also report Spearman rank correlation of every
detector column against the reference (gru_512_2l).

  python scripts/ablations/detector_collect.py --dataset age
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "ablations"))
from common import label_model  # noqa: E402

BASE = REPO_ROOT / "log" / "ablations" / "detector"
REFERENCE = "gru_512_2l"
DETECTOR_ORDER = (
    "gru_128_1l", "gru_512_1l", "gru_512_2l", "gru_1024_3l",
    "transformer_128_1l", "transformer_512_1l", "transformer_512_2l", "transformer_1024_3l",
)
DET_RE = re.compile(r"Detection detector_ablation/(\S+) score \(\d+ hist\) mean$")


def model_name(final_path: Path, dataset_root: Path) -> str:
    return "/".join(final_path.parent.relative_to(dataset_root).parts)


def collect_auroc(dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-detector train/val AUROC from each detector's own results.csv.

    Path: ``<model>/seed_0/evaluation/detection/<detector>/results.csv``.
    """
    root = BASE / dataset / "horizon_32"
    train, val = {}, {}
    for res in sorted(root.glob("**/evaluation/detection/*/results.csv")):
        parts = res.relative_to(root).parts
        model = "/".join(parts[: parts.index("seed_0")])
        det = res.parent.name
        s = pd.read_csv(res, index_col=0)["mean"]
        train.setdefault(model, {})[det] = float(s["train_MulticlassAUROC"])
        val.setdefault(model, {})[det] = float(s["MulticlassAUROC"])
    cols = [d for d in DETECTOR_ORDER if any(d in v for v in val.values())]
    return (
        pd.DataFrame(train).T.reindex(columns=cols),
        pd.DataFrame(val).T.reindex(columns=cols),
    )


def collect(dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = BASE / dataset / "horizon_32"
    means, stds = {}, {}
    for final in sorted(root.glob("**/final.csv")):
        model = model_name(final, root)
        s = pd.read_csv(final, index_col=0)["mean"]
        m_row, sd_row = {}, {}
        for idx, val in s.items():
            hit = DET_RE.search(str(idx))
            if hit:
                m_row[hit.group(1)] = float(val)
                sd_row[hit.group(1)] = float(s.get(str(idx).replace(") mean", ") std")))
        if m_row:
            means[model] = m_row
            stds[model] = sd_row
    cols = [d for d in DETECTOR_ORDER if any(d in m for m in means.values())]
    mean_df = pd.DataFrame(means).T.reindex(columns=cols)
    std_df = pd.DataFrame(stds).T.reindex(columns=cols)
    return mean_df, std_df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="age")
    args = ap.parse_args()

    mean_df, std_df = collect(args.dataset)
    mean_df.index = [label_model(m) for m in mean_df.index]
    std_df.index = mean_df.index

    out = BASE / args.dataset / "cdfs_table.csv"
    mean_df.to_csv(out)
    print(f"# cDFS (mean) — {args.dataset} — horizon 32   [wrote {out}]\n")
    print(mean_df.round(3).to_string())

    print("\n# Spearman rank corr of model ordering vs reference "
          f"({REFERENCE}):\n")
    shown = mean_df.round(3)
    ref = shown[REFERENCE]
    for col in shown.columns:
        rho = shown[col].corr(ref, method="spearman")
        tag = "  <- reference" if col == REFERENCE else ""
        print(f"  {col:22s} rho={rho:+.3f}{tag}")

    train_df, val_df = collect_auroc(args.dataset)
    print("\n# Detector fit (AUROC averaged over models) — overfitting check:\n")
    summary = pd.DataFrame({
        "train_AUROC": train_df.mean(),
        "val_AUROC": val_df.mean(),
        "gap": (train_df - val_df).mean(),
    }).round(3)
    print(summary.to_string())


if __name__ == "__main__":
    main()
