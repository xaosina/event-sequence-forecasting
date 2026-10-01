"""
Illustration: a roulette bet on black splits the evening into two branches
("Lost" / "Won") + a prediction of the form mode(category) + mean(amount).

The branch probabilities are not made up: they come directly from European
roulette (single zero) — 18 of 37 pockets are black.
  - Won (bet on black comes up):  18/37 = 48.65%
  - Lost (red or zero comes up):  19/37 = 51.35%

Each branch is a Transport -> Food -> Entertainment sequence after a $100 bet
on black (the bet itself is off-chart; history only shows a $5 entrance fee):
  Branch "Lost" (51.35%): Bus -> Fast food -> Streaming, small amounts.
  Branch "Won"  (48.65%): Taxi -> Restaurant -> Night club, spends the $200 won.

At each step the prediction uses:
  - category = category of the more likely branch (Lost, since 51.35% > 48.65%),
  - amount = mean of the "Lost" and "Won" amounts.
The result names plausible everyday categories (bus, fast food, streaming), but
the amounts are unrealistically large for them (e.g. "Fast food for $63"),
because they are averaged with the much more expensive "Won" branch.

Run:
    python roulette_fork_plot.py
    python roulette_fork_plot.py --show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.transforms import ScaledTranslation

OUTPUT_STEM = 'roulette_fork_plot'
FONT_FAMILY = 'Lora'
FONT_DIR = Path(__file__).resolve().parent / 'fonts'


def setup_fonts() -> None:
    # Prefer the project's own Lora-Regular.ttf/Lora-Bold.ttf (as in the
    # reference evening_fork_named_plot.py). Fall back to whatever Lora
    # build is already registered with the system (e.g. a variable font),
    # so the script still runs in environments without the local fonts/ dir.
    local_fonts = [FONT_DIR / 'Lora-Regular.ttf', FONT_DIR / 'Lora-Bold.ttf']
    if all(p.exists() for p in local_fonts):
        for p in local_fonts:
            font_manager.fontManager.addfont(str(p))
    plt.rcParams.update({
        'font.family': FONT_FAMILY,
        'font.serif': [FONT_FAMILY],
        'mathtext.fontset': 'custom',
        'mathtext.rm': FONT_FAMILY,
        'mathtext.bf': f'{FONT_FAMILY}:bold',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })

POINT_LABEL_FONT_SIZE = 20
POINT_LABEL_PRICE_GAP = 24
FONT_SMALL = 16
FONT_LABEL = 22
FONT_TITLE = 28
FONT_LEGEND = 19
FONT_LEGEND_HEADER = FONT_LEGEND
LEGEND_HEADER_SHIFT_PT = 40
FONT_TICK = 16
COLOR_LOST = '#24796C'
COLOR_WON = '#5D69B1'
COLOR_PREDICTION = '#E58606'

PREDICTION_PRICE_COLOR = '#fa2742'
HISTORY_COLOR = '#888780'
PREDICTION_COLOR = COLOR_PREDICTION

DEFAULT_LABEL_OFFSET = (12, 18)
DEFAULT_PRICE_OFFSET = (12, 18 - POINT_LABEL_PRICE_GAP)
Y_MIN = -35
Y_MAX = 105
FIGSIZE = (15, 6)
X_MIN = -4.5
X_MAX = 3.6
LINEWIDTH = 3.0
POINT_MARKER_SCATTER_SIZE = 120
POINT_MARKER_LEGEND_SIZE = 11  # ~sqrt(POINT_MARKER_SCATTER_SIZE), matches scatter markers

# Per-label layout: category/price offsets in points, optional horizontal alignment.
LABEL_LAYOUT = {
    'Taxi': {
        'label': (-14, 18),
        'price': (-14, 18 - POINT_LABEL_PRICE_GAP),
        'ha': 'right',
    },
    'Bus': {
        'label': (12, -24),
        'price': (12, -24 - POINT_LABEL_PRICE_GAP),
    },
    'Fast food': {
        'label': (-50, -24),
        'price': (-50, -24 - POINT_LABEL_PRICE_GAP),
    },
}

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

# Off-chart: $100 bet on black at the fork; history shows only the $5 entrance fee.
BET_AMOUNT = 100

history_t = [-1, -0.5, 0]
history_amt = [15, 4, 5]  # Groceries, Coffee, casino entrance fee (anchor)

branch_lost = dict(
    t=[1.18, 2.4, 2.8],
    amt=[2, 4, 2],
    labels=['Bus', 'Fast food', 'Streaming'],
    color=COLOR_LOST,
    label='Lost the bet (probability of this chain: 51.35%)',
)

branch_won = dict(
    t=[0.85, 1.92, 2.88],  # Taxi left of Bus at step 1
    amt=[40, 100, 70],
    labels=['Taxi', 'Restaurant', 'Night club'],
    color=COLOR_WON,
    label='Won the bet (probability of this chain: 48.65%)',
)

pred_amt = [round((a + b) / 2, 1) for a, b in zip(branch_lost['amt'], branch_won['amt'])]
prediction = dict(
    t=[(tl + tw) / 2 for tl, tw in zip(branch_lost['t'], branch_won['t'])],
    amt=pred_amt,
    labels=branch_lost['labels'],
    color=PREDICTION_PRICE_COLOR,
    label_color=PREDICTION_PRICE_COLOR,
    price_color=PREDICTION_PRICE_COLOR,
    label='Prediction (mode + mean)',
    use_label_layout=False,
    label_overrides={
        'Bus': {
            'label': (0, 46),
            'price': (0, 46 - POINT_LABEL_PRICE_GAP),
        },
        'Fast food': {
            'label': (-14, 23),
            'price': (-14, 23 - POINT_LABEL_PRICE_GAP),
            'ha': 'right',
        },
    },
)


def plot_branch(ax, branch, now_t, now_amt, linestyle='-', point_label_font_size=POINT_LABEL_FONT_SIZE):
    t_vals, amt_vals, labels = branch['t'], branch['amt'], branch['labels']
    color = branch['color']

    ax.plot([now_t] + t_vals, [now_amt] + amt_vals,
            color=color, linewidth=LINEWIDTH, linestyle=linestyle, zorder=2)

    label_color = branch.get('label_color', branch.get('annot_color', color))
    price_color = branch.get('price_color', label_color)

    for t, a, lab in zip(t_vals, amt_vals, labels):
        amount = f'${a:g}'
        if branch.get('use_label_layout', True):
            layout = LABEL_LAYOUT.get(lab, {})
        else:
            layout = branch.get('label_overrides', {}).get(lab, {})
        label_xy = layout.get('label', DEFAULT_LABEL_OFFSET)
        price_xy = layout.get('price', DEFAULT_PRICE_OFFSET)
        ha = layout.get('ha', 'left')

        ax.scatter(t, a, s=POINT_MARKER_SCATTER_SIZE, color=color, zorder=3, edgecolors='none')
        ax.annotate(lab, (t, a), textcoords='offset points',
                    xytext=label_xy, fontsize=point_label_font_size, color=label_color,
                    linespacing=1.3, zorder=10, ha=ha)
        ax.annotate(amount, (t, a), textcoords='offset points',
                    xytext=price_xy, fontsize=point_label_font_size, color=price_color,
                    linespacing=1.3, zorder=10, ha=ha)


def apply_bold_legend_prefixes(legend, prefixes=('Lost:', 'Won:', 'Prediction:')):
    for text in legend.get_texts():
        label = text.get_text()
        for prefix in prefixes:
            if label.startswith(prefix):
                text.set_text(rf'$\bf{{{prefix}}}$' + label[len(prefix):])
                break


def build_legend(ax):
    handles = [
        Line2D([], [], linestyle='None', marker='None',
               label='Past history'),
        Line2D([], [], color=HISTORY_COLOR, linewidth=LINEWIDTH, marker='o', markersize=POINT_MARKER_LEGEND_SIZE,
               label='Known transactions'),
        Line2D([], [], linestyle='None', marker='None',
               label='Possible outcomes (bet on black)'),
        Line2D([], [], color=branch_lost['color'], linewidth=LINEWIDTH, marker='o', markersize=POINT_MARKER_LEGEND_SIZE,
               label='Lost: red/zero comes up\n(probability of this chain: 51.35%)'),
        Line2D([], [], color=branch_won['color'], linewidth=LINEWIDTH, marker='o', markersize=POINT_MARKER_LEGEND_SIZE,
               label='Won: black comes up\n(probability of this chain: 48.65%)'),
        Line2D([], [], color=PREDICTION_PRICE_COLOR, linewidth=LINEWIDTH, linestyle='--', marker='o', markersize=POINT_MARKER_LEGEND_SIZE,
               label='Prediction: optimal by MSE and Accuracy\nyet completely unrealistic'),
    ]
    legend = ax.legend(
        handles=handles,
        loc='center left',
        # bbox_to_anchor=(0, 0),
        frameon=True,
        facecolor='none',
        edgecolor='none',
        framealpha=0,
        fontsize=FONT_LEGEND,
        handlelength=1.2,
        handletextpad=0.8,
        labelspacing=0.45,
        borderpad=0.8,
    )
    for handle in legend.legend_handles:
        if handle.get_linestyle() != 'None':
            handle.set_linewidth(LINEWIDTH)
    header_indices = {0, 2}
    header_shift = ScaledTranslation(
        -LEGEND_HEADER_SHIFT_PT / 72, 0, ax.figure.dpi_scale_trans,
    )
    for idx, text in enumerate(legend.get_texts()):
        if idx in header_indices:
            text.set_fontweight('bold')
            text.set_fontsize(FONT_LEGEND_HEADER)
            text.set_transform(text.get_transform() + header_shift)
    apply_bold_legend_prefixes(legend)
    return legend


def make_figure(point_label_font_size: int = POINT_LABEL_FONT_SIZE):
    setup_fonts()
    fig, ax = plt.subplots(figsize=FIGSIZE)

    now_t, now_amt = 0, history_amt[-1]

    ax.plot(history_t, history_amt, color=HISTORY_COLOR, linewidth=LINEWIDTH, zorder=2)
    for t, a in zip(history_t, history_amt):
        ax.scatter(t, a, s=POINT_MARKER_SCATTER_SIZE, color=HISTORY_COLOR, zorder=3, edgecolors='none')
    ax.annotate('Casino entry', (now_t, now_amt), textcoords='offset points',
                xytext=(80, -24), fontsize=point_label_font_size, color=HISTORY_COLOR,
                linespacing=1.3, zorder=10, ha='right')
    ax.annotate('$5', (now_t, now_amt), textcoords='offset points',
                xytext=(-14, -24 - POINT_LABEL_PRICE_GAP), fontsize=point_label_font_size,
                color=HISTORY_COLOR, linespacing=1.3, zorder=10, ha='right')

    plot_branch(ax, branch_lost, now_t, now_amt, point_label_font_size=point_label_font_size)
    plot_branch(ax, branch_won, now_t, now_amt, point_label_font_size=point_label_font_size)
    plot_branch(ax, prediction, now_t, now_amt, linestyle='--',
                point_label_font_size=point_label_font_size)

    ax.set_ylabel('Transaction amount, $', fontsize=FONT_LABEL)
    ax.set_xlabel('Time', fontsize=FONT_LABEL)
    ax.set_title('Data example: maximizing predictive metrics\n'
                 'leads to unrealistic forecasts',
                 fontsize=FONT_TITLE, pad=16)
    build_legend(ax)
    ax.tick_params(axis='y', labelsize=FONT_TICK)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xlim(X_MIN, X_MAX)

    fig.tight_layout()
    return fig


def save_figure(fig, out_dir: Path | None = None) -> tuple[Path, Path]:
    out_dir = out_dir or Path(__file__).resolve().parent
    png_path = out_dir / f'{OUTPUT_STEM}.png'
    pdf_path = out_dir / f'{OUTPUT_STEM}.pdf'

    fig.savefig(png_path, dpi=150)
    fig.savefig(pdf_path)

    return png_path, pdf_path


def main(
    show: bool = False,
    out_dir: Path | None = None,
    point_label_font_size: int = POINT_LABEL_FONT_SIZE,
) -> tuple[Path, Path]:
    fig = make_figure(point_label_font_size=point_label_font_size)
    png_path, pdf_path = save_figure(fig, out_dir=out_dir)

    print(f'Saved: {png_path}')
    print(f'Saved: {pdf_path}')

    if show:
        plt.show()
    else:
        plt.close(fig)

    return png_path, pdf_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--show',
        action='store_true',
        help='show the plot in a window after saving',
    )
    parser.add_argument(
        '--point-label-font-size',
        type=int,
        default=POINT_LABEL_FONT_SIZE,
        help='font size for category/price labels on points (default: %(default)s)',
    )
    args = parser.parse_args()
    main(show=args.show, point_label_font_size=args.point_label_font_size)