"""Helpers shared by the ablation scripts."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_ROOT = REPO_ROOT / "log" / "gen-paper"

MODEL_LABELS: dict[str, str] = {
    "ar": "AR",
    "baselines/gt": "GT",
    "baselines/hist_sampler": "Hist.",
    "baselines/mode": "Mode",
    "baselines/repeat": "Repeat",
    "ldm/vae": "LDM",
    "detpp": "DEF",
    "cdiff": "DDM",
    "multitoken": "MT",
    "ldm/vae/booster/wasserstein_barybooster1": "LDM-WB",
    "baselines/gt/booster/wasserstein_barybooster1": "GT-WB",
}


def label_model(model: str) -> str:
    return MODEL_LABELS.get(model, model.replace("baselines/", ""))


def write_meta(
    out_dir: Path,
    source_path: Path,
    *,
    extra: dict | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    date = datetime.now(ZoneInfo("Europe/Moscow")).strftime("%Y-%m-%d %H:%M:%S")
    payload: dict = {
        "path": f"{out_dir.as_posix()}/",
        "source_path": f"{source_path.as_posix()}/",
        "date": date,
        "final": True,
    }
    if extra:
        payload.update(extra)
    with open(out_dir / "meta.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, default_flow_style=False, allow_unicode=True)


def _best_checkpoint(ckpt_dir: Path) -> Path | None:
    """Pick the best checkpoint in a run's ``ckpt`` dir.

    Filenames look like ``epoch__0036_-_loss__-79.92.ckpt``. With
    ``ckpt_replace: true`` only the best ckpt is kept, so usually there is a
    single file; if several exist we prefer the lowest tracked ``loss``.
    """
    ckpts = sorted(ckpt_dir.glob("*.ckpt"))
    if not ckpts:
        return None
    if len(ckpts) == 1:
        return ckpts[0]

    def loss_key(p: Path) -> float:
        m = re.search(r"loss__(-?\d+(?:\.\d+)?)", p.name)
        return float(m.group(1)) if m else float("inf")

    return min(ckpts, key=loss_key)


def resolve_ckpt_resume(config_path: Path, source_path: Path) -> str | None:
    """Return a ckpt path to inject when the run config lacks ``ckpt_resume``.

    Some gen-paper "final" runs are training runs that stored their weights in
    their own ``seed_0/ckpt/`` dir but left ``ckpt_resume: null`` in the config
    (e.g. every ``detpp`` run). Re-evaluating such a config would silently score
    a randomly-initialised model, so we recover the in-run checkpoint here.
    Returns ``None`` when the config already resumes weights or the model needs
    none (deterministic baselines have no ``ckpt`` dir).
    """
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if (cfg.get("trainer") or {}).get("ckpt_resume"):
        return None
    ckpt = _best_checkpoint(source_path / "seed_0" / "ckpt")
    return str(ckpt) if ckpt else None
