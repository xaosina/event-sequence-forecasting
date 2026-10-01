#!/usr/bin/env python3
import argparse
import csv
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_ROOT = REPO_ROOT / "log" / "gen-paper"
TIME_FMT = "%Y-%m-%d %H:%M:%S,%f"
ALLOWED_MODELS = {"ldm/vae", "cdiff", "multitoken", "detpp", "ar"}
DATASET_ORDER = ("age", "gender", "alphabattle")
MODEL_ORDER = ("ldm/vae", "cdiff", "detpp",  "multitoken", "ar")
MODEL_LABELS = {
    "ar": "AR",
    "cdiff": "DDM",
    "detpp": "DEF",
    "ldm/vae": "LDM",
    "multitoken": "MT-AR",
}
DATASET_LABELS = {
    "age": "Age",
    "gender": "Gender",
    "alphabattle": "Alphabattle",
}

TRAIN_START_RE = re.compile(r"Epoch 0001:\s+train started")
TRAIN_END_RE = re.compile(r"<All keys matched successfully>")
INFER_START_RE = re.compile(r"Epoch \d{4}:\s+validation started")
INFER_END_RE = re.compile(r"Sampling done\.")
TIMESTAMP_RE = re.compile(r" - (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - ")


@dataclass
class RunStats:
    dataset: str
    horizon: int
    model: str
    run_root: Path
    log_path: Path
    train_seconds: float | None
    inference_seconds_max: float | None
    inference_pairs: int
    memory_after: float | None


def parse_timestamp(line: str) -> datetime | None:
    match = TIMESTAMP_RE.search(line)
    if not match:
        return None
    return datetime.strptime(match.group(1), TIME_FMT)


def parse_log_times(log_path: Path) -> tuple[float | None, float | None, int]:
    train_start: datetime | None = None
    train_end: datetime | None = None
    inference_phase_started = False
    inference_starts: list[datetime] = []
    inference_durations: list[float] = []

    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        ts = parse_timestamp(line)
        if ts is None:
            continue

        if train_start is None and TRAIN_START_RE.search(line):
            train_start = ts
            continue

        if train_start is not None and train_end is None and TRAIN_END_RE.search(line):
            train_end = ts
            inference_phase_started = True
            continue

        if inference_phase_started and INFER_START_RE.search(line):
            inference_starts.append(ts)
            continue

        if inference_phase_started and INFER_END_RE.search(line) and inference_starts:
            start_ts = inference_starts.pop(0)
            inference_durations.append((ts - start_ts).total_seconds())
            if len(inference_durations) >= 3:
                break

    train_seconds = (train_end - train_start).total_seconds() if train_start and train_end else None
    inference_seconds_max = max(inference_durations) if inference_durations else None
    return train_seconds, inference_seconds_max, len(inference_durations)


def parse_memory_after(final_csv_path: Path) -> float | None:
    if not final_csv_path.is_file():
        return None

    with final_csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            if row[0].strip() != "memory_after":
                continue
            try:
                # Prefer mean if present, otherwise fallback to run-0 value.
                if len(row) > 2 and row[2].strip():
                    return float(row[2])
                return float(row[1])
            except ValueError:
                return None
    return None


def parse_run_metadata(log_path: Path) -> tuple[str, int, str, Path]:
    rel = log_path.relative_to(PAPER_ROOT)
    parts = rel.parts
    if len(parts) < 5:
        raise ValueError(f"unexpected log path: {log_path}")

    dataset = parts[0]
    lr_idx = parts.index("lr")
    if len(parts) > 1 and parts[1].startswith("horizon_"):
        horizon = int(parts[1].removeprefix("horizon_"))
        model = "/".join(parts[2:lr_idx])
        run_root = PAPER_ROOT.joinpath(*parts[:lr_idx])
    else:
        # AR runs are stored as dataset/ar/lr/3.e-4/seed_0/log (without horizon folder).
        horizon = 64
        model = "/".join(parts[1:lr_idx])
        run_root = PAPER_ROOT.joinpath(*parts[:lr_idx])
    return dataset, horizon, model, run_root


def collect_stats() -> list[RunStats]:
    stats: list[RunStats] = []
    horizon_logs = set(PAPER_ROOT.glob("*/horizon_64/**/lr/3.e-4/seed_0/log"))
    ar_logs = set(PAPER_ROOT.glob("*/ar/lr/3.e-4/seed_0/log"))
    for log_path in sorted(horizon_logs | ar_logs):
        dataset, horizon, model, run_root = parse_run_metadata(log_path)
        if model not in ALLOWED_MODELS:
            continue
        train_seconds, inference_seconds_max, inference_pairs = parse_log_times(log_path)
        memory_after = parse_memory_after(run_root / "final.csv")
        stats.append(
            RunStats(
                dataset=dataset,
                horizon=horizon,
                model=model,
                run_root=run_root,
                log_path=log_path,
                train_seconds=train_seconds,
                inference_seconds_max=inference_seconds_max,
                inference_pairs=inference_pairs,
                memory_after=memory_after,
            )
        )
    return stats


def write_csv(rows: list[RunStats], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "dataset",
                "horizon",
                "model",
                "train_seconds",
                "inference_seconds_max",
                "inference_pairs",
                "memory_after",
                "memory_after_gb",
                "run_root",
                "log_path",
            ]
        )
        for row in rows:
            memory_after_gb = (row.memory_after / 1024.0) if row.memory_after is not None else None
            writer.writerow(
                [
                    row.dataset,
                    row.horizon,
                    row.model,
                    row.train_seconds,
                    row.inference_seconds_max,
                    row.inference_pairs,
                    row.memory_after,
                    memory_after_gb,
                    row.run_root.as_posix(),
                    row.log_path.as_posix(),
                ]
            )


def fmt_value(value: float | None, ndigits: int = 1) -> str:
    if value is None:
        return "-"
    return f"{value:.{ndigits}f}"


def write_latex_tables(rows: list[RunStats], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[RunStats]] = {dataset: [] for dataset in DATASET_ORDER}
    for row in rows:
        grouped.setdefault(row.dataset, []).append(row)

    with output_path.open("w", encoding="utf-8") as f:
        f.write("\\section{Runtime and memory consumption}\n")
        f.write("\\label{app:runtime-h64}\n\n")
        f.write(
            "This appendix summarizes training time, inference time, and post-evaluation GPU memory "
            "usage for the main learned models at horizon $L=64$. Memory is reported in GB.\n\n"
        )

        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{llrrr}\n")
        f.write("\\toprule\n")
        f.write("Dataset & Model & Train (min) & Inference (min) & Memory (GB) \\\\\n")
        f.write("\\midrule\n")
        for dataset in DATASET_ORDER:
            dataset_rows = grouped.get(dataset, [])
            dataset_rows.sort(key=lambda x: MODEL_ORDER.index(x.model) if x.model in MODEL_ORDER else 999)
            dataset_label = DATASET_LABELS.get(dataset, dataset)
            for row in dataset_rows:
                model_label = MODEL_LABELS.get(row.model, row.model)
                train_min = (row.train_seconds / 60.0) if row.train_seconds is not None else None
                infer_min = (row.inference_seconds_max / 60.0) if row.inference_seconds_max is not None else None
                memory_gb = (row.memory_after / 1024.0) if row.memory_after is not None else None
                train_s = fmt_value(train_min, ndigits=1)
                infer_s = fmt_value(infer_min, ndigits=1)
                memory = fmt_value(memory_gb, ndigits=2)
                f.write(f"{dataset_label} & {model_label} & {train_s} & {infer_s} & {memory} \\\\\n")
            if dataset != DATASET_ORDER[-1]:
                f.write("\\midrule\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Runtime and memory summary for Age, Gender, and Alphabattle at horizon $L=64$.}\n")
        f.write("\\label{tab:runtime-h64}\n")
        f.write("\\end{table}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect training/inference times and memory_after for horizon_64 runs."
    )
    parser.add_argument(
        "--output",
        default="log/gen-paper/horizon_64_time_memory_summary.csv",
        help="Output CSV path relative to repo root.",
    )
    parser.add_argument(
        "--latex-output",
        default="log/gen-paper/horizon_64_time_memory_summary.tex",
        help="LaTeX output path relative to repo root.",
    )
    args = parser.parse_args()

    rows = collect_stats()
    output_path = (REPO_ROOT / args.output).resolve()
    latex_output_path = (REPO_ROOT / args.latex_output).resolve()
    write_csv(rows, output_path)
    write_latex_tables(rows, latex_output_path)
    print(f"Collected {len(rows)} runs")
    print(f"Wrote: {output_path}")
    print(f"Wrote: {latex_output_path}")


if __name__ == "__main__":
    main()
