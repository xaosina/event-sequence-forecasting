import ast
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from IPython.display import HTML, clear_output, display

_RE_LOG_TRAIN = re.compile(r"Epoch\s+(\d+):\s+avg train loss = ([\d.]+)")
_RE_LOG_METRICS = re.compile(r"Epoch\s+(\d+):\s+metrics:\s+(.+)$")
_RE_LOG_EMA = re.compile(r"Epoch\s+(\d+):\s+EMA metrics:\s+(.+)$")


def parse_log_file(p):
    d = {}
    with open(p) as f:
        for l in f:
            if "finished successfully" in l:
                break
            if m := _RE_LOG_TRAIN.search(l):
                d.setdefault(int(m[1]), {})["train_loss"] = float(m[2])
            elif m := _RE_LOG_METRICS.search(l):
                d.setdefault(int(m[1]), {}).update(ast.literal_eval(m[2]))
            elif m := _RE_LOG_EMA.search(l):
                for k, v in ast.literal_eval(m[2]).items():
                    d.setdefault(int(m[1]), {})[f"ema_{k}"] = v
    
    return (
        pd.DataFrame(d)
        .T.reset_index(names="epoch")
        .sort_values("epoch")
        .reset_index(drop=True)
    )


def parse_val(s):
    # "3.e-4(foo)" -> (3e-4, "foo")
    if "(" in s:
        base, suffix = s.split("(", 1)
        suffix = suffix[:-1]  # remove ")"
    else:
        base, suffix = s, ""

    return float(base), suffix


def sort_vals(vals):
    return sorted(vals, key=lambda x: parse_val(x))


class Exp:
    DEFAULT_GRIDS = {
        "lr": ("3.e-3", "1.e-3", "3.e-4", "1.e-4", "3.e-5", "1.e-5"),
        "temp": ("0", "0.1", "0.4", "1.0", "1.2", "1.5"),
        "cfg": (),
    }
    SWEEP_KINDS = tuple(DEFAULT_GRIDS.keys())

    def __init__(self, path, base="/home/transaction-generation/log/gen-paper/"):
        self.path = Path(path)
        try:
            self.exp_name = self.path.relative_to(base)
        except ValueError:
            self.exp_name = self.path
        self.dataset = self.path.parts[0]
        self.exp_name = Path(*self.exp_name.parts[1:])

    def _grid(self, name):
        root = self.path / name
        defaults = list(self.DEFAULT_GRIDS.get(name, ()))
        if not root.exists():
            return sort_vals(defaults)

        vals = [p.name for p in root.iterdir() if p.is_dir()]
        base = set(vals) | set(defaults)
        return sort_vals(base)

    def grid_dir(self, name):
        return self.path / name

    def grid_status(self, name):
        root = self.grid_dir(name)
        grid = self._grid(name)

        out = {}
        for v in grid:
            d = root / v
            has_results = (d / "results.csv").exists()
            failed = (d / "FAILED.txt").exists()
            out[v] = {
                "exists": d.exists(),
                "failed": failed,
                "done": has_results,
                "in_progress": d.exists() and not has_results and not failed,
            }
        return grid, out

    def is_done(self, grid_name="lr"):
        _, s = self.grid_status(grid_name)
        return all(v["done"] or v["failed"] for v in s.values())

    def summary(self, grid_name="lr"):
        grid, s = self.grid_status(grid_name)
        done = all(v["done"] or v["failed"] for v in s.values())
        if not done:
            self.show_summary(grid_name=grid_name, grid=grid, status=s)
        return done

    def show_summary(self, grid_name="lr", status=None, grid=None):
        if grid is None or status is None:
            g, s0 = self.grid_status(grid_name)
            grid = g if grid is None else grid
            status = s0 if status is None else status
        s = status

        headers = [""] + list(grid)
        marks = ["ok"] + [
            "⚠"
            if s[c]["failed"]
            else (
                "✓"
                if s[c]["done"]
                else ("…" if s[c]["in_progress"] else "×")
            )
            for c in grid
        ]

        rows = [headers, marks]
        widths = [max(len(str(row[i])) for row in rows) for i in range(len(headers))]

        for row in rows:
            print("  ".join(str(v).ljust(widths[i]) for i, v in enumerate(row)))

    @staticmethod
    def _read_results_csv(path):
        """Load only index + mean; full parse is costly for wide metric tables."""
        hdr = pd.read_csv(path, nrows=0)
        if "mean" in hdr.columns:
            idx_col = hdr.columns[0]
            return pd.read_csv(path, index_col=0, usecols=[idx_col, "mean"])
        return pd.read_csv(path, index_col=0)

    @staticmethod
    def metric_getter(path, metric=None, get_metric=False):
        df = Exp._read_results_csv(path)
        fallback = ["Paired overall", "test_GenOTD"]
        if metric is None:
            for f in fallback:
                if f in df.index:
                    metric = f
                    break
        if metric is None:
            return df["mean"]
        if metric not in df.index:
            return 0
        if get_metric:
            return df.loc[metric, "mean"], metric
        return df.loc[metric, "mean"]

    def best_sweep_metric(self, grid_name, metric=None, get_all=False, grid=None):
        if grid is None:
            grid, _ = self.grid_status(grid_name)
        root = self.grid_dir(grid_name)

        res = []
        for cell in grid:
            f = root / cell / "results.csv"
            if f.exists():
                value, metric = self.metric_getter(f, metric, get_metric=True)
                res.append((cell, value))

        if not res:
            return ([], None) if get_all else (None, None)

        res = sorted(res, key=lambda x: x[1], reverse=True)
        return (res, metric) if get_all else res[0]

    def plot_sweep(self, grid_name, metric=None):
        grid, s = self.grid_status(grid_name)
        scores, metric = self.best_sweep_metric(
            grid_name, metric, get_all=True, grid=grid
        )
        scores = dict(scores)

        ys = [scores.get(c, np.nan) for c in grid]
        xs = range(len(grid))
        plt.figure(figsize=(5, 3))
        yd = [y for y in ys if not np.isnan(y)]
        lo, hi = (min(yd), max(yd)) if yd else (0, 1)
        pad = (hi - lo) * 0.2 + 1e-6
        ymin, ymax = lo - pad, hi + pad

        bar_vals = [ymin if np.isnan(y) else y for y in ys]
        heights = [y - ymin for y in bar_vals]

        plt.bar(xs, heights, bottom=ymin)
        ty = ymin + 0.15 * (ymax - ymin)

        for i, cell in enumerate(grid):
            if not np.isnan(ys[i]):
                plt.text(i, ys[i], f"{ys[i]:.4g}", ha="center", va="bottom")
            else:
                if s[cell]["failed"]:
                    msg = "X\n\nFailed"
                elif s[cell]["in_progress"]:
                    msg = "...\n\nRunning"
                else:
                    msg = "X\n\nMissing"
                plt.text(i, ty, msg, ha="center", va="center")

        plt.xticks(xs, grid)
        plt.ylim(ymin, ymax)
        plt.title(f"{str(self.exp_name)} | {grid_name} | {metric}")
        plt.tight_layout()
        plt.show()


    def plot_losses(self, cell, grid_name="lr"):
        p = self.grid_dir(grid_name) / cell / "seed_0" / "log"
        if not p.exists():
            print("no log")
            return

        df = parse_log_file(p)

        if "train_loss" in df:
            df["train_loss"] *= -1

        plt.figure(figsize=(5, 3))

        if "train_loss" in df:
            plt.plot(df["epoch"], df["train_loss"], label="train_loss")

        if "loss" in df:
            plt.plot(df["epoch"], df["loss"], label="loss")
            val_loss_ema = df["loss"].ewm(alpha=0.1, adjust=False).mean()
            plt.plot(
                df["epoch"],
                val_loss_ema,
                label="loss (EMA, alpha=0.1)",
            )

        if "ema_loss" in df:
            plt.plot(df["epoch"], df["ema_loss"], label="ema_loss")

        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title(f"{self.exp_name} | {grid_name}={cell}")
        # plt.yscale("log")
        if df.shape[0] > 50:
            plt.ylim(df["train_loss"][5], None)
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def read_meta(exp_root):
        exp_root = Path(exp_root)
        p = exp_root / "meta.yaml"
        if not p.exists():
            return {}
        with open(p) as f:
            return yaml.safe_load(f) or {}

    @staticmethod
    def meta_final_state(exp_root):
        """Human 'paper lock' from meta.yaml key ``final`` (bool). None if absent."""
        meta = Exp.read_meta(exp_root)
        if "final" not in meta:
            return None
        return bool(meta["final"])

    @staticmethod
    def set_meta_final_approved(exp_root, approved=True):
        """Set ``final`` in meta.yaml; preserves other keys. Caller must validate preconditions."""
        exp_root = Path(exp_root)
        meta_path = exp_root / "meta.yaml"
        data = Exp.read_meta(exp_root)
        data["final"] = bool(approved)
        if approved:
            data["final_approved_at"] = datetime.now(ZoneInfo("Europe/Moscow")).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        with open(meta_path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

    def save_final(self, cell, force=False, grid_name="lr"):
        root = self.grid_dir(grid_name)
        src = root / cell / "results.csv"
        path = root / cell
        if not src.exists():
            print("no results")
            return

        exp_root = root.parent
        dst = exp_root / "final.csv"
        meta = exp_root / "meta.yaml"

        if dst.exists() and not force:
            print("FILES ALREADY EXISTS (use triple click to rewrite)")
            return

        shutil.copy2(src, dst)

        prev = Exp.read_meta(exp_root)
        keep_lock = prev.get("final") is True

        data = {
            "path": str(path),
            "grid": grid_name,
            "cell": str(cell),
            "date": datetime.now(ZoneInfo("Europe/Moscow")).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "final": True if keep_lock else False,
        }
        if keep_lock and "final_approved_at" in prev:
            data["final_approved_at"] = prev["final_approved_at"]

        with open(meta, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

        print(f"saved {path}")


class ExpNavigator:
    def __init__(self, root):
        self.root = Path(root)
        self.cur = self.root
        self.out = widgets.Output()

    @staticmethod
    def sweep_kinds_present(path):
        path = Path(path)
        return [k for k in Exp.SWEEP_KINDS if (path / k).is_dir()]

    @classmethod
    def first_sweep_kind(cls, path):
        kinds = cls.sweep_kinds_present(path)
        return kinds[0] if kinds else "lr"

    def is_exp(self, path):
        return bool(self.sweep_kinds_present(path))

    def children(self, path):
        return sorted(
            [p for p in path.iterdir() if p.is_dir()],
            key=lambda x: (not x.is_dir(), x.name),
        )

    def rel(self, path):
        try:
            return "." if path == self.root else str(path.relative_to(self.root))
        except ValueError:
            return str(path)

    def run(self):
        self.render_tree()
        display(self.out)

    def render_tree(self):
        with self.out:
            clear_output()

            items = []

            title = widgets.HTML(f"<b>{self.rel(self.cur)}</b>")
            items.append(title)

            # if self.cur != self.root:
            up_btn = widgets.Button(
                description="..", layout=widgets.Layout(width="auto")
            )
            up_btn.on_click(lambda _: self.go(self.cur.parent))
            items.append(up_btn)

            for p in self.children(self.cur):
                if self.is_exp(p):
                    label = widgets.HTML(
                        f"📊 <b>{p.name}</b> <span style='color:#666'>(sweeps)</span>"
                    )
                    sweep_btns = []
                    for kind in self.sweep_kinds_present(p):
                        b = widgets.Button(
                            description=kind,
                            layout=widgets.Layout(width="auto"),
                        )
                        b.on_click(
                            lambda _, p=p, kind=kind: self.render_exp_menu(p, kind)
                        )
                        sweep_btns.append(b)
                    row = widgets.HBox(
                        [label] + sweep_btns,
                        layout=widgets.Layout(flex_flow="row wrap", align_items="center"),
                    )
                    items.append(row)
                else:
                    btn = widgets.Button(
                        description=f"📁 {p.name}",
                        layout=widgets.Layout(width="auto"),
                    )
                    btn.on_click(lambda _, p=p: self.go(p))
                    items.append(btn)

            display(widgets.VBox(items))

    def render_exp_menu(self, path, grid_name):
        with self.out:
            clear_output()

            path = Path(path)
            self.cur = path.parent
            exp = Exp(path)

            title = widgets.HTML(
                f"<b>experiment:</b> {self.rel(path)} &nbsp;|&nbsp; <b>grid:</b> {grid_name}"
            )

            path_input = widgets.Text(
                value=str(path),
                layout=widgets.Layout(width="70%"),
            )

            go_btn = widgets.Button(
                description="Go to:",
                layout=widgets.Layout(width="auto"),
            )

            def go_to_path(_):
                p = Path(path_input.value)
                if p.exists():
                    if self.is_exp(p):
                        kinds = self.sweep_kinds_present(p)
                        target_grid = grid_name if grid_name in kinds else self.first_sweep_kind(p)
                        self.render_exp_menu(p, target_grid)
                    else:
                        self.go(p)

            go_btn.on_click(go_to_path)

            btn_back = widgets.Button(
                description="back to tree", layout=widgets.Layout(width="auto")
            )

            info = widgets.Output()

            metrics = []
            root = exp.grid_dir(grid_name)
            grid_cells = exp._grid(grid_name)
            for cell in grid_cells:
                f = root / cell / "results.csv"
                if f.exists():
                    metrics = list(Exp._read_results_csv(f).index)
                    metrics = self.metric_filter(metrics)
                    break

            train_metrics = [m for m in metrics if "train" in m]
            test_metrics = [m for m in metrics if "test" in m or "EMA" in m]
            other_metrics = [
                m
                for m in metrics
                if "train" not in m and "test" not in m and "EMA" not in m
            ]

            def make_row(name, metric_list):
                buttons = []
                for metric in metric_list:
                    btn = widgets.Button(
                        description=str(metric),
                        layout=widgets.Layout(width="auto"),
                    )
                    btn.on_click(
                        lambda _, metric=metric: self._plot_exp(
                            exp, grid_name, metric, info
                        )
                    )
                    buttons.append(btn)

                return widgets.HBox(
                    [widgets.HTML(f"\n<b>{name.upper()}</b>: ")] + buttons,
                    layout=widgets.Layout(flex_flow="row wrap", align_items="center"),
                )

            self._plot_exp(exp, grid_name, None, info)

            btn_back.on_click(lambda _: self.render_tree())

            cell_buttons = []
            for cell in grid_cells:
                btn = widgets.Button(
                    description=f"{cell}",
                    layout=widgets.Layout(width="auto"),
                )
                btn.on_click(
                    lambda _, cell=cell: self._plot_losses(
                        exp, grid_name, cell, info
                    )
                )
                cell_buttons.append(btn)

            losses_row = widgets.HBox(
                [widgets.HTML(f"<b>LOSSES</b> ({grid_name}): ")] + cell_buttons,
                layout=widgets.Layout(flex_flow="row wrap"),
            )

            save_buttons = []
            for cell in grid_cells:
                btn = widgets.Button(
                    description=f"{cell}",
                    layout=widgets.Layout(width="auto"),
                )
                btn._click_times = []

                def handler(_, cell=cell, btn=btn):
                    import time

                    now = time.time()

                    # keep only recent clicks (1 sec window)
                    btn._click_times = [t for t in btn._click_times if now - t < 1.0]
                    btn._click_times.append(now)

                    force = len(btn._click_times) >= 3
                    if force:
                        btn._click_times = []  # reset after trigger

                    self._save_final(exp, grid_name, cell, info, force)

                btn.on_click(handler)
                save_buttons.append(btn)
            save_row = widgets.HBox(
                [widgets.HTML(f"<b>SAVE</b> ({grid_name}): ")] + save_buttons,
                layout=widgets.Layout(flex_flow="row wrap"),
            )

            paper_lock = self._paper_lock_panel(path, grid_name, info)

            display(
                widgets.VBox(
                    [
                        title,
                        widgets.HBox([go_btn, path_input]),
                        make_row("train", train_metrics),
                        widgets.HTML("<hr style='margin:6px 0'>"),
                        make_row("test", test_metrics),
                        widgets.HTML("<hr style='margin:6px 0'>"),
                        make_row("other", other_metrics),
                        widgets.HTML("<hr style='margin:6px 0'>"),
                        losses_row,
                        widgets.HTML("<hr style='margin:6px 0'>"),
                        save_row,
                        paper_lock,
                        widgets.HTML("<hr style='margin:6px 0'>"),
                        btn_back,
                        info,
                    ]
                )
            )

    def _plot_exp(self, exp, grid_name, metric, info):
        with info:
            clear_output()
            exp.plot_sweep(grid_name, metric)

    def _plot_losses(self, exp, grid_name, cell, info):
        with info:
            clear_output()
            exp.plot_losses(cell, grid_name=grid_name)

    def _save_final(self, exp, grid_name, cell, info, force):
        with info:
            clear_output()
            exp.save_final(cell, force=force, grid_name=grid_name)

    def _paper_lock_panel(self, path, grid_name, info):
        """meta.final=true = human paper freeze; guarded by checkbox + typed FREEZE."""
        path = Path(path)

        def meta_details_html(meta):
            meta_file = path / "meta.yaml"
            meta_href = f"file://{meta_file.resolve()}"
            if not meta:
                return (
                    "<div style='margin-top:4px;color:#57606a;font-size:12px'>"
                    "meta details: <code>grid=—</code> ; <code>cell=—</code> ; <code>path=—</code>"
                    f" ; meta file: <a href='{meta_href}' target='_blank'><code>{meta_file}</code></a>"
                    "</div>"
                )
            grid_v = meta.get("grid") or "—"
            cell_v = meta.get("cell") or "—"
            path_v = meta.get("path") or "—"
            return (
                "<div style='margin-top:4px;color:#57606a;font-size:12px'>"
                f"meta details: <code>grid={grid_v}</code> ; "
                f"<code>cell={cell_v}</code> ; "
                f"<code>path={path_v}</code>"
                f" ; meta file: <a href='{meta_href}' target='_blank'><code>{meta_file}</code></a>"
                "</div>"
            )

        def lines_status():
            meta = Exp.read_meta(path)
            mf = Exp.meta_final_state(path)
            at = meta.get("final_approved_at") or ""
            has_csv = (path / "final.csv").exists()
            has_meta = (path / "meta.yaml").exists()
            if mf is True:
                extra = f" <span style='color:#57606a'>(approved {at})</span>" if at else ""
                return (
                    f"<span style='color:#1a7f37;font-weight:600'>meta.final = true</span>"
                    f"{extra} — paper lock is <b>ON</b>."
                )
            if not has_meta:
                return (
                    "<span style='color:#57606a'>No meta.yaml yet.</span> "
                    "Use <b>SAVE</b> above to write <code>final.csv</code> + <code>meta.yaml</code>."
                )
            if not has_csv:
                return (
                    "<span style='color:#cf222e'>meta.yaml without final.csv</span> "
                    "(unexpected); fix paths before locking."
                )
            saved = meta.get("date") or ""
            extra = f" Saved: {saved}." if saved else ""
            return (
                f"<span style='color:#9a6700;font-weight:600'>meta.final = false</span> — "
                f"artifact saved, not paper-locked.{extra}"
            )

        status = widgets.HTML(
            value=(
                f"<div style='max-width:820px'><b>Paper lock</b>: {lines_status()}"
                f"{meta_details_html(Exp.read_meta(path))}</div>"
            )
        )

        def refresh_status():
            status.value = (
                f"<div style='max-width:820px'><b>Paper lock</b>: {lines_status()}"
                f"{meta_details_html(Exp.read_meta(path))}</div>"
            )

        if Exp.meta_final_state(path) is True:
            inner = widgets.VBox(
                [
                    status,
                    widgets.HTML(
                        "<div style='color:#57606a;font-size:12px;max-width:820px'>"
                        "To clear the lock, edit <code>meta.yaml</code> manually (not offered here).</div>"
                    ),
                ]
            )
        else:
            warn = widgets.HTML(
                "<div style='max-width:820px;color:#57606a;font-size:12px;margin-bottom:6px'>"
                "<b>meta.final=true</b> records that this experiment is <b>frozen for the paper</b> "
                "(see also <code>ExpChecker</code> column <code>lock:YES</code>). "
                "Triple-click <b>SAVE</b> overwrite keeps an existing lock.</div>"
            )
            cb = widgets.Checkbox(
                value=False,
                description="I verified final.csv is the sweep cell I want for the paper.",
                indent=False,
                layout=widgets.Layout(width="95%"),
            )
            phrase = widgets.Text(
                description="Confirm:",
                placeholder="type FREEZE in capitals",
                layout=widgets.Layout(width="420px"),
            )
            apply_btn = widgets.Button(
                description="Set meta.final = true",
                button_style="danger",
                layout=widgets.Layout(width="auto"),
            )

            def on_apply(_):
                if Exp.meta_final_state(path) is True:
                    with info:
                        clear_output()
                        print("Already locked.")
                    return
                if not (path / "final.csv").exists() or not (path / "meta.yaml").exists():
                    with info:
                        clear_output()
                        print("Need both final.csv and meta.yaml (use SAVE on a cell first).")
                    return
                if not cb.value:
                    with info:
                        clear_output()
                        print("Enable the checkbox first.")
                    return
                if phrase.value.strip() != "FREEZE":
                    with info:
                        clear_output()
                        print('Confirmation phrase must be exactly FREEZE (all caps).')
                    return
                Exp.set_meta_final_approved(path, True)
                with info:
                    clear_output()
                    print("meta.final=true written. Paper lock ON.")
                self.render_exp_menu(path, grid_name)

            apply_btn.on_click(on_apply)
            inner = widgets.VBox([status, warn, cb, phrase, apply_btn])

        acc = widgets.Accordion(children=[inner])
        acc.set_title(0, "Paper lock (meta.final) — use with care")
        acc.selected_index = None

        def on_accordion_change(change):
            if change.get("name") == "selected_index" and change["new"] == 0:
                refresh_status()

        acc.observe(on_accordion_change, names="selected_index")
        return acc

    def metric_filter(self, metrics):
        res = []
        exclude = [
            "micro",
            "std",
            "Visualization",
            "Cardinality",
            "memory",
            "Reconstruction",
            # "Paired",
        ]
        include = ["Paired"]
        for m in metrics:
            if any(kw in m for kw in exclude):
                continue
            res += [m]
        return res

    def go(self, path):
        self.cur = path
        self.render_tree()


class FinalTables:
    HORIZONS = ("horizon_4", "horizon_16", "horizon_32", "horizon_64")
    MODELS = (
        "baselines/gt",
        "baselines/hist_sampler",
        "baselines/mode",
        "baselines/repeat",
        "ldm/vae",
        # "ldm/vae/booster/wasserstein_barybooster",
        "ldm/vae/booster/wasserstein_barybooster1",
        "cdiff/booster/wasserstein_barybooster1",
        "ldm/vae_h4",
        "ldm/vae_h16",
        "ldm/vae_h32",
        "cdiff",
        "cdiff_h4",
        "cdiff_h16",
        "cdiff_h32",
        "ar",
        "multitoken",
        "multitoken_h4",
        "multitoken_h16",
        "multitoken_h32",
        "detpp",
        "detpp_h4",
        "detpp_h16",
        "detpp_h32",
    )

    def __init__(self, root):
        self.root = Path(root)

    def metric_filter(self, metrics):
        res = []
        exclude = [
            "micro",
            "std",
            "Visualization",
            # "Cardinality",
            "memory",
            "Reconstruction",
            "Paired",
            # "Matched",
            "loss",
        ]
        include = [
            "test_",
            "EMA_"
        ]
        for m in metrics:
            if any(kw in m for kw in exclude):
                continue
            for inc in include:
                if inc in m:
                    res += [m]
        return res

    def exp_path(self, horizon, model):
        return self.root / horizon / model

    def read_final(self, horizon, model):
        p = self.exp_path(horizon, model) / "final.csv"
        if not p.exists():
            return None
        df = Exp._read_results_csv(p)
        df = df.loc[self.metric_filter(df.index)]
        test_m = [n for n in df.index if "test" in n]
        if "EMA_GenOTD" in df.index:
            df = df.drop(test_m)
            df.index = [n.replace("EMA_", "test_") for n in df.index]
        return df

    def read_meta(self, horizon, model):
        p = self.exp_path(horizon, model) / "meta.yaml"
        if not p.exists():
            return {}
        with open(p) as f:
            return yaml.safe_load(f) or {}

    def horizon_table(self, horizon):
        cols = {}
        for model in self.MODELS:
            df = self.read_final(horizon, model)
            if df is None:
                continue
            cols[model] = df["mean"]

        if not cols:
            return pd.DataFrame()

        return pd.DataFrame(cols).round(4)

    def all_horizon_tables(self):
        return {h: self.horizon_table(h) for h in self.HORIZONS}

    def final_flags(self):
        data = {}
        for horizon in self.HORIZONS:
            col = {}
            for model in self.MODELS:
                meta = self.read_meta(horizon, model)
                col[horizon] = "Missing"
            data[horizon] = col

        df = pd.DataFrame(index=self.MODELS, columns=self.HORIZONS)

        for horizon in self.HORIZONS:
            for model in self.MODELS:
                meta = self.read_meta(horizon, model)
                if "final" in meta:
                    df.loc[model, horizon] = "Done" if meta["final"] else "Some result"

        return df


class ExpChecker:
    """Status checker for experiment coverage under log/gen-paper/<dataset>.

    Basic usage:
        checker = ExpChecker("age")
        checker.summary()  # status only
        checker.suggest_commands(device="cuda:0")  # optional, on demand
    """

    LR_GRID = ("1.e-5", "3.e-5", "1.e-4", "3.e-4", "1.e-3", "3.e-3")
    TEMP_GRID = ("0", "0.1", "0.4", "0.5", "0.6", "1.0", "1.2", "1.5", "2.0")
    CFG_GRID = (0, 1, 1.2, 1.3, 1.4, 1.5, 2, 3, 4)
    HORIZONS = (4, 16, 32, 64)  # horizon_32 is debug-only by default
    CROSS_MODELS = ("detpp", "multitoken", "ldm/vae", "cdiff")
    CROSS_PAIRS = ((4, 16), (4, 64), (16, 64), (32, 64), (4, 32), (16, 32))
    ANSI_COLORS = {
        "done": "\033[92m",  # green
        "in_progress": "\033[93m",  # yellow
        "missing": "\033[91m",  # red
    }
    ANSI_RESET = "\033[0m"

    def __init__(self, dataset, root_path="log/gen-paper"):
        self.project_root = Path(__file__).resolve().parents[2]
        self.root_path = self._resolve_root_path(root_path)
        self.dataset_root = self._resolve_dataset_root(dataset)
        self.dataset = self.dataset_root.name
        self.spec = self._build_spec()

    def _resolve_root_path(self, root_path):
        p = Path(root_path).expanduser()
        if p.is_absolute():
            return p
        return (self.project_root / p).resolve()

    def _resolve_dataset_root(self, dataset):
        ds = Path(dataset).expanduser()
        if ds.is_absolute():
            return ds

        if len(ds.parts) == 1:
            return (self.root_path / ds).resolve()

        anchored = (self.project_root / ds).resolve()
        if str(anchored).startswith(str(self.root_path)):
            return anchored
        return (self.root_path / ds).resolve()

    @classmethod
    def _build_spec(cls):
        spec = [
            {"exp": "ae/vae", "checks": ("lr", "final")},
            {"exp": "ar", "checks": ("lr", "final")},
        ]
        for h in cls.HORIZONS:
            prefix = f"horizon_{h}"
            spec += [
                {"exp": f"{prefix}/ar", "checks": ("temp", "final")},
                {"exp": f"{prefix}/detpp", "checks": ("lr", "final")},
                {"exp": f"{prefix}/ldm/vae", "checks": ("lr", "final", "cfg")},
                # {"exp": f"{prefix}/ldm/vae_best_recon", "checks": ("lr", "final", "cfg")},
                {"exp": f"{prefix}/cdiff", "checks": ("lr", "final", "cfg")},
                {"exp": f"{prefix}/multitoken", "checks": ("lr", "final", "temp")},
                {"exp": f"{prefix}/seqvae", "checks": ("lr", "final")},
            ]
            for b in ("gt", "hist_sampler", "mode", "repeat"):
                spec += [{"exp": f"{prefix}/baselines/{b}", "checks": ("final",)}]
        for train_h, eval_h in cls.CROSS_PAIRS:
            for model in cls.CROSS_MODELS:
                spec += [{"exp": f"horizon_{eval_h}/{model}_h{train_h}", "checks": ("final",)}]
        return spec

    def _grid_values(self, grid_name):
        if grid_name == "lr":
            return self.LR_GRID
        if grid_name == "temp":
            return self.TEMP_GRID
        if grid_name == "cfg":
            return self.CFG_GRID
        raise ValueError(f"Unknown grid: {grid_name}")

    @staticmethod
    def _read_meta_grid(exp_root):
        p = exp_root / "meta.yaml"
        if not p.exists():
            return None
        with open(p) as f:
            meta = yaml.safe_load(f) or {}
        g = meta.get("grid")
        return None if g is None else str(g)

    @staticmethod
    def _read_meta_final_flag(exp_root):
        """``True`` / ``False`` from meta.yaml if key present; ``None`` if no meta or no key."""
        return Exp.meta_final_state(exp_root)

    @staticmethod
    def _cell_sweep_dot_state(cell):
        """Dot color for a grid cell: failed (brown) overrides done/run/missing."""
        if cell.get("failed"):
            return "failed"
        if cell["done"]:
            return "done"
        if cell["exists"]:
            return "in_progress"
        return "missing"

    def _grid_state(self, exp_root, grid_name):
        grid_dir = exp_root / grid_name
        cells = self._grid_values(grid_name)
        items = {}
        for cell in cells:
            cell_dir = grid_dir / str(cell)
            has_results = (cell_dir / "results.csv").exists()
            failed = (cell_dir / "FAILED.txt").exists()
            items[str(cell)] = {
                "exists": cell_dir.exists(),
                "failed": failed,
                "done": has_results,
            }

        # Terminal cells (results or FAILED) count toward grid completion → overall DONE.
        done_n = sum(v["done"] or v["failed"] for v in items.values())
        exists_n = sum(v["exists"] for v in items.values())
        total_n = len(items)
        if done_n == total_n:
            state = "done"
        elif done_n > 0 or exists_n > 0:
            state = "in_progress"
        else:
            state = "missing"
        return {
            "state": state,
            "done_n": done_n,
            "exists_n": exists_n,
            "total_n": total_n,
            "items": items,
        }

    def check(self):
        out = []
        for row in self.spec:
            exp_rel = row["exp"]
            checks = row["checks"]
            exp_root = self.dataset_root / exp_rel
            status = {"exp": exp_rel, "path": exp_root, "checks": {}, "overall": "done"}

            for c in checks:
                if c == "final":
                    ok = (exp_root / "final.csv").exists()
                    status["checks"]["final"] = {"state": "done" if ok else "missing", "ok": ok}
                else:
                    status["checks"][c] = self._grid_state(exp_root, c)

            states = [v["state"] for v in status["checks"].values()]
            if all(s == "done" for s in states):
                status["overall"] = "done"
            elif any(s == "in_progress" for s in states):
                status["overall"] = "in_progress"
            else:
                status["overall"] = "missing"
            status["meta_grid"] = self._read_meta_grid(exp_root)
            status["meta_final"] = self._read_meta_final_flag(exp_root)
            out.append(status)
        return out

    @staticmethod
    def _overall_mark(overall_state, meta_grid):
        marks = {"done": "DONE", "in_progress": "RUN", "missing": "MISS"}
        text = marks[overall_state]
        if overall_state == "done" and meta_grid:
            return f"{text}:{meta_grid}"
        return text

    def _state_label(self, state, use_color=True):
        marks = {"done": "DONE", "in_progress": "RUN", "missing": "MISS"}
        text = marks[state]
        if not use_color:
            return text
        if not sys.stdout.isatty():
            return text
        return f"{self.ANSI_COLORS[state]}{text}{self.ANSI_RESET}"

    def _overall_label(self, overall_state, meta_grid=None, use_color=True):
        text = self._overall_mark(overall_state, meta_grid)
        if not use_color:
            return text
        if not sys.stdout.isatty():
            return text
        return f"{self.ANSI_COLORS[overall_state]}{text}{self.ANSI_RESET}"

    @staticmethod
    def _meta_final_short_label(meta_final):
        if meta_final is True:
            return "lock:YES"
        if meta_final is False:
            return "lock:no"
        return "lock:—"

    def summary(self, use_color=True):
        rows = self.check()

        print(f"Dataset: {self.dataset_root} (standard horizons: {self.HORIZONS})")
        print("-" * 100)
        for r in rows:
            checks = []
            for k, v in r["checks"].items():
                if k == "final":
                    checks += [f"final:{self._state_label(v['state'], use_color=use_color)}"]
                else:
                    checks += [
                        f"{k}:{self._state_label(v['state'], use_color=use_color)}({v['done_n']}/{v['total_n']})"
                    ]
            checks.append(self._meta_final_short_label(r.get("meta_final")))
            overall = self._overall_label(
                r["overall"], r.get("meta_grid"), use_color=use_color
            )
            print(f"{overall:12} | {r['exp']:<34} | " + " ; ".join(checks))

        done_n = sum(r["overall"] == "done" for r in rows)
        run_n = sum(r["overall"] == "in_progress" for r in rows)
        miss_n = sum(r["overall"] == "missing" for r in rows)
        print("-" * 100)
        print(f"Totals: done={done_n}, in_progress={run_n}, missing={miss_n}, all={len(rows)}")

    def summary_notebook(self, hide_finished=False):
        rows = self.check()
        if hide_finished:
            rows = [r for r in rows if r.get("meta_final") is not True]
        palette = {"done": "#1a7f37", "in_progress": "#0a7aca", "missing": "#cf222e"}
        marks = {"done": "DONE", "in_progress": "RUN", "missing": "MISS"}
        dot_palette = {
            "done": "#2da44e",
            "in_progress": "#1f6feb",
            "missing": "#cf222e",
            "failed": "#8b5a2b",
        }

        grouped = {"global": []}
        for h in self.HORIZONS:
            grouped[f"horizon_{h}"] = []

        for r in rows:
            h = self._horizon_from_exp(r["exp"])
            key = "global" if h is None else f"horizon_{h}"
            grouped.setdefault(key, []).append(r)

        sections = []
        for section_name, section_rows in grouped.items():
            if not section_rows:
                continue
            section_rows = sorted(section_rows, key=lambda x: x["exp"])
            html_rows = []
            for r in section_rows:
                checks = []
                mf = r.get("meta_final")
                if mf is True:
                    lock_html = (
                        "<span style='color:#1a7f37;font-weight:600' title='meta.yaml final=true'>"
                        "lock:YES</span>"
                    )
                elif mf is False:
                    lock_html = (
                        "<span style='color:#9a6700;font-weight:600' title='meta.yaml final=false'>"
                        "lock:no</span>"
                    )
                else:
                    lock_html = (
                        "<span style='color:#57606a' title='no meta.final key'>lock:—</span>"
                    )
                for k, v in r["checks"].items():
                    state = v["state"]
                    label = marks[state]
                    if k == "final":
                        checks.append(
                            f"<span style='color:{palette[state]};font-weight:600'>{k}:{label}</span>"
                        )
                    else:
                        dots = []
                        for cell in v["items"].values():
                            c_state = self._cell_sweep_dot_state(cell)
                            dots.append(
                                "<span "
                                f"title='{c_state}' "
                                "style='display:inline-block;width:8px;height:8px;border-radius:50%;"
                                f"background:{dot_palette[c_state]};margin:0 2px 0 0;vertical-align:middle'></span>"
                            )
                        dots_html = "".join(dots)
                        checks.append(
                            f"<span style='color:{palette[state]};font-weight:600'>{k}:{label}</span>"
                            f" ({v['done_n']}/{v['total_n']}) "
                            f"<span style='white-space:nowrap'>{dots_html}</span>"
                        )

                overall = r["overall"]
                overall_text = self._overall_mark(overall, r.get("meta_grid"))
                exp_label = r["exp"] if section_name == "global" else r["exp"].split("/", 1)[1]
                html_rows.append(
                    "<tr>"
                    f"<td style='padding:4px 8px;text-align:left;color:{palette[overall]};font-weight:700'>{overall_text}</td>"
                    f"<td style='padding:4px 8px;text-align:left'><code>{exp_label}</code></td>"
                    f"<td style='padding:4px 8px;text-align:left'>{' ; '.join(checks)}</td>"
                    f"<td style='padding:4px 8px;text-align:left;white-space:nowrap'>{lock_html}</td>"
                    "</tr>"
                )

            sec_done = sum(r["overall"] == "done" for r in section_rows)
            sec_run = sum(r["overall"] == "in_progress" for r in section_rows)
            sec_miss = sum(r["overall"] == "missing" for r in section_rows)
            sections.append(
                "<br>"
                "<div style='margin-top:8px;margin-bottom:6px;padding-top:6px;border-top:1px solid #d0d7de'>"
                f"<b style='font-size:15px'>{section_name}</b> "
                f"<span style='color:#57606a'>(done={sec_done}, run={sec_run}, miss={sec_miss}, all={len(section_rows)})</span>"
                "</div>"
                "<table style='border-collapse:collapse;margin-top:6px;text-align:left'>"
                "<thead><tr>"
                "<th style='text-align:left;padding:4px 8px;vertical-align:top'>overall</th>"
                "<th style='text-align:left;padding:4px 8px;vertical-align:top'>experiment</th>"
                "<th style='text-align:left;padding:4px 8px;vertical-align:top'>checks</th>"
                "<th style='text-align:left;padding:4px 8px;vertical-align:top'>lock<br>"
                "<span style='font-weight:400;color:#57606a;font-size:11px'>(meta.final)</span></th>"
                "</tr></thead>"
                f"<tbody>{''.join(html_rows)}</tbody></table>"
            )

        done_n = sum(r["overall"] == "done" for r in rows)
        run_n = sum(r["overall"] == "in_progress" for r in rows)
        miss_n = sum(r["overall"] == "missing" for r in rows)

        html = (
            f"<div><b>Dataset:</b> <code>{self.dataset_root}</code> "
            f"(standard horizons: {self.HORIZONS})</div>"
            f"{''.join(sections)}"
            f"<div style='margin-top:10px'><b>Totals:</b> done={done_n}, in_progress={run_n}, "
            f"missing={miss_n}, all={len(rows)}</div>"
        )
        display(HTML(html))

    def summary_notebook_flat(self):
        rows = self.check()
        palette = {"done": "#1a7f37", "in_progress": "#0a7aca", "missing": "#cf222e"}
        marks = {"done": "DONE", "in_progress": "RUN", "missing": "MISS"}
        dot_palette = {
            "done": "#2da44e",
            "in_progress": "#1f6feb",
            "missing": "#cf222e",
            "failed": "#8b5a2b",
        }

        html_rows = []
        for r in rows:
            checks = []
            mf = r.get("meta_final")
            if mf is True:
                lock_html = (
                    "<span style='color:#1a7f37;font-weight:600' title='meta.yaml final=true'>"
                    "lock:YES</span>"
                )
            elif mf is False:
                lock_html = (
                    "<span style='color:#9a6700;font-weight:600' title='meta.yaml final=false'>"
                    "lock:no</span>"
                )
            else:
                lock_html = (
                    "<span style='color:#57606a' title='no meta.final key'>lock:—</span>"
                )
            for k, v in r["checks"].items():
                state = v["state"]
                label = marks[state]
                if k == "final":
                    checks.append(
                        f"<span style='color:{palette[state]};font-weight:600'>{k}:{label}</span>"
                    )
                else:
                    dots = []
                    for cell in v["items"].values():
                        c_state = self._cell_sweep_dot_state(cell)
                        dots.append(
                            "<span "
                            f"title='{c_state}' "
                            "style='display:inline-block;width:8px;height:8px;border-radius:50%;"
                            f"background:{dot_palette[c_state]};margin:0 2px 0 0;vertical-align:middle'></span>"
                        )
                    dots_html = "".join(dots)
                    checks.append(
                        f"<span style='color:{palette[state]};font-weight:600'>{k}:{label}</span>"
                        f" ({v['done_n']}/{v['total_n']}) "
                        f"<span style='white-space:nowrap'>{dots_html}</span>"
                    )

            overall = r["overall"]
            overall_text = self._overall_mark(overall, r.get("meta_grid"))
            html_rows.append(
                "<tr>"
                f"<td style='padding:4px 8px;text-align:left;color:{palette[overall]};font-weight:700'>{overall_text}</td>"
                f"<td style='padding:4px 8px;text-align:left'><code>{r['exp']}</code></td>"
                f"<td style='padding:4px 8px;text-align:left'>{' ; '.join(checks)}</td>"
                f"<td style='padding:4px 8px;text-align:left;white-space:nowrap'>{lock_html}</td>"
                "</tr>"
            )

        done_n = sum(r["overall"] == "done" for r in rows)
        run_n = sum(r["overall"] == "in_progress" for r in rows)
        miss_n = sum(r["overall"] == "missing" for r in rows)

        html = (
            f"<div><b>Dataset:</b> <code>{self.dataset_root}</code> "
            f"(standard horizons: {self.HORIZONS})</div>"
            "<table style='border-collapse:collapse;margin-top:8px;text-align:left'>"
            "<thead><tr>"
            "<th style='text-align:left;padding:4px 8px;vertical-align:top'>overall</th>"
            "<th style='text-align:left;padding:4px 8px;vertical-align:top'>experiment</th>"
            "<th style='text-align:left;padding:4px 8px;vertical-align:top'>checks</th>"
            "<th style='text-align:left;padding:4px 8px;vertical-align:top'>lock<br>"
            "<span style='font-weight:400;color:#57606a;font-size:11px'>(meta.final)</span></th>"
            "</tr></thead>"
            f"<tbody>{''.join(html_rows)}</tbody></table>"
            f"<div style='margin-top:8px'><b>Totals:</b> done={done_n}, in_progress={run_n}, "
            f"missing={miss_n}, all={len(rows)}</div>"
        )
        display(HTML(html))

    def _experiment_yaml(self, exp_rel):
        return Path("scripts/experiments") / self.dataset / f"{exp_rel}.yaml"

    @staticmethod
    def _horizon_from_exp(exp_rel):
        p0 = exp_rel.split("/")[0]
        if p0.startswith("horizon_"):
            try:
                return int(p0.split("_", 1)[1])
            except ValueError:
                return None
        return None

    @staticmethod
    def _model_from_exp(exp_rel):
        parts = exp_rel.split("/")
        if "baselines" in parts:
            i = parts.index("baselines")
            return "/".join(parts[i : i + 2])
        if parts[0].startswith("horizon_"):
            return "/".join(parts[1:])
        return exp_rel

    @staticmethod
    def _cross_pair_from_exp(exp_rel):
        parts = exp_rel.split("/")
        if len(parts) < 2:
            return None
        eval_h = ExpChecker._horizon_from_exp(exp_rel)
        if eval_h is None:
            return None
        tail = parts[-1]
        if "_h" not in tail:
            return None
        try:
            train_h = int(tail.rsplit("_h", 1)[1])
        except ValueError:
            return None
        return train_h, eval_h

    @staticmethod
    def _cross_model_from_exp(exp_rel):
        pair = ExpChecker._cross_pair_from_exp(exp_rel)
        if pair is None:
            return None
        train_h, _ = pair
        suffix = f"_h{train_h}"
        model = ExpChecker._model_from_exp(exp_rel)
        if not model.endswith(suffix):
            return None
        return model[: -len(suffix)]

    def _match_filters(self, exp_rel, model=None, horizon=None):
        if horizon is not None and self._horizon_from_exp(exp_rel) != int(horizon):
            return False
        if model is None:
            return True
        exp_model = self._model_from_exp(exp_rel)
        if exp_model == model:
            return True
        cross_model = self._cross_model_from_exp(exp_rel)
        return cross_model == model

    def _suggest_commands_dedup(
        self,
        device="cuda:0",
        model=None,
        horizon=None,
        grid=None,
        grid_value=None,
        only_missing=True,
    ):
        rows = self.check()
        commands = []
        need_baselines = False
        cross_needed = {m: set() for m in self.CROSS_MODELS}

        for r in rows:
            exp_rel = r["exp"]
            checks = r["checks"]
            exp_root = self.dataset_root / exp_rel
            if not self._match_filters(exp_rel, model=model, horizon=horizon):
                continue

            if "lr" in checks and (grid in (None, "lr")):
                if (not only_missing) or checks["lr"]["state"] != "done":
                    yml = self._experiment_yaml(exp_rel)
                    if grid_value is None:
                        commands += [f"./scripts/grid_lr.sh {yml} {device}"]
                    else:
                        commands += [f"./scripts/grid_lr.sh {yml} {device} {grid_value}"]

            if "temp" in checks and (grid in (None, "temp")):
                if (not only_missing) or checks["temp"]["state"] != "done":
                    h = self._horizon_from_exp(exp_rel)
                    m = self._model_from_exp(exp_rel).split("/")[-1]
                    if grid_value is None:
                        if h is None:
                            commands += [f"python scripts/grid_temp.py {self.dataset_root} --model {m} --device {device}"]
                        else:
                            commands += [
                                f"python scripts/grid_temp.py {self.dataset_root} --model {m} --device {device} --horizon {h}"
                            ]
                    elif h is not None:
                        commands += [
                            f"python scripts/grid_temp.py {self.dataset_root} --model {m} --device {device} --horizon {h} --temp {grid_value}"
                        ]

            if "cfg" in checks and (grid in (None, "cfg")):
                if (not only_missing) or checks["cfg"]["state"] != "done":
                    if grid_value is None:
                        commands += [f"python scripts/grid_cfg.py {exp_root} {device}"]
                    else:
                        commands += [f"python scripts/grid_cfg.py {exp_root} {device} --cfg {grid_value}"]

            if "/baselines/" in exp_rel and checks.get("final", {}).get("state") != "done":
                need_baselines = True
            cross_model = self._cross_model_from_exp(exp_rel)
            cross_pair = self._cross_pair_from_exp(exp_rel)
            if cross_model and cross_pair and (grid in (None, "final")):
                if (not only_missing) or checks.get("final", {}).get("state") != "done":
                    if cross_model in cross_needed:
                        cross_needed[cross_model].add(cross_pair)

        if need_baselines and grid in (None, "final"):
            commands += [f"./scripts/run_baselines.sh {self.dataset} {device}"]
        for cross_model, pairs in cross_needed.items():
            if not pairs:
                continue
            pairs_arg = ",".join(f"{train}:{eval_}" for train, eval_ in sorted(pairs))
            commands += [
                "python scripts/grid_cross_horizon.py "
                f"{self.dataset_root} --model {cross_model} --device {device} --pairs {pairs_arg}"
            ]

        if not commands:
            return []
        return list(dict.fromkeys(str(c) for c in commands))

    def suggest_commands(
        self,
        device="cuda:0",
        model=None,
        horizon=None,
        grid=None,
        grid_value=None,
        only_missing=True,
        as_script=True,
    ):
        dedup = self._suggest_commands_dedup(
            device, model, horizon, grid, grid_value, only_missing
        )
        if not dedup:
            print("No commands to suggest. Everything looks complete for the standard scope.")
            return []
        if as_script:
            script_lines = [f"{c} ; \\" for c in dedup[:-1]] + [f"{dedup[-1]} ;"]
            text = "\n".join(script_lines)
            hint = (
                "<div style='font-size:12px;color:#57606a;margin:0 0 6px 0'>"
                "Run from project root. Select all (Ctrl+A) and copy (Ctrl+C) if needed.</div>"
            )
        else:
            text = "\n".join(dedup)
            hint = (
                "<div style='font-size:12px;color:#57606a;margin:0 0 6px 0'>"
                "Suggested commands (one per line). Select and copy if needed.</div>"
            )
        n_lines = len(text.splitlines())
        rows = min(24, max(3, n_lines + 1))
        ta = widgets.Textarea(
            value=text,
            rows=rows,
            layout=widgets.Layout(width="100%"),
        )
        display(widgets.VBox([widgets.HTML(hint), ta]))
        return dedup
