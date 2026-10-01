"""Atomic skip/claim for grid ablation runs (parallel-safe).

Lock is a sibling file ``<out_dir>.lock`` — the run directory itself is NOT
created on claim, so a parallel launcher never sees an empty folder without a
lock and mistakes it for "not started".
"""

import os
from pathlib import Path


def should_skip_run(out_dir: Path) -> bool:
    if (out_dir / "final.csv").is_file():
        return True
    # Pre-lock runs (or main.py output) may leave the dir without final.csv;
    # skip so main.py does not create <run>(1), etc.
    return out_dir.is_dir()


def lock_path(out_dir: Path) -> Path:
    return Path(f"{out_dir}.lock")


def claim_run(out_dir: Path) -> bool:
    """Return True if this process should run; False to skip."""
    if should_skip_run(out_dir):
        return False
    lock = lock_path(out_dir)
    if lock.exists():
        return False
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        return True
    except FileExistsError:
        return False


def release_run(out_dir: Path) -> None:
    lock_path(out_dir).unlink(missing_ok=True)
