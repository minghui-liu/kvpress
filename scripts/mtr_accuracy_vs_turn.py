"""MTR-Bench: accuracy vs. turn number, one panel per (model, cache budget).

The layout is one row containing one panel for each configuration present in
``ACC``. With the current data this is a 1x4 figure: one 512-token panel for
each model. Within each panel, x = turn number and y is the cumulative average
through that turn of half-scaled accuracy. Full is plotted as
``0.5 * Full accuracy``. Each method is plotted as
``0.5 * Full accuracy - ACC[method]``.

--------------------------------------------------------------------------------
DATA CONTRACT
--------------------------------------------------------------------------------
ACC[(model, budget)][method] = ([0.5 * (Full accuracy - method accuracy)], n_games)
FULL_ACC[model] = ([Full accuracy], n_games)

The Full baselines have already been subtracted in ACC, so plotting reconstructs
half-scaled method accuracy as ``0.5 * FULL_ACC - ACC``. Series may contain
fewer than 15 values when later turns are absent. After a method stops, each
later turn contributes zero: a three-turn method therefore has
``(acc1 + acc2 + acc3 + 0) / 4`` at turn 4.

Error bands are one binomial standard error of each half-scaled accuracy. The
per-turn errors are propagated in quadrature through the cumulative average.
Full and every measured method are drawn with an error band.

MISSING (= -1) marks any turn that has not been measured. Rules:
  * a single -1 breaks the line at that turn (drawn as a gap, not as y=-1)
  * after the last measured turn, later turns contribute zero through turn 15
  * a curve that is entirely -1 is not drawn at all; the method is instead listed
    in the panel's "not measured" annotation
  * a panel with nothing measured is labelled "no data"
n_games = 0 goes with an all-missing curve.

--------------------------------------------------------------------------------
HOW ACCURACY IS DEFINED
--------------------------------------------------------------------------------
Straight from the per-turn answer JSONLs written by reason/evaluate_mtr.py, NOT
from the terminal `success` flag and NOT from game length:

    accuracy(t) = ok[t] / n[t]

    n[t]   turns that actually reached turn t
    ok[t]  of those, responses that parsed into a legal move (result != "Invalid")
           and were not judged wrong by the monitor (feedback lacking
           incorrect / lose / wrong)

This is survivor-conditional -- the denominator is turns that happened, so unlike
a survival curve it is free to go up as well as down, and turn 1 is a real
measurement rather than 1.0 by construction.

Observations are POOLED (averaged) across all MTR categories, game types, seeds
and dataset blocks for a configuration; runs are not deduplicated, since the same
basename legitimately recurs once per game type. A turn with fewer than 5
observations is reported as -1.

Regenerate with results/mtr_perturn_agg.py (per cluster) + mtr_perturn_emit.py.

Run:  python scripts/mtr_accuracy_vs_turn.py
"""

import argparse
import textwrap

import numpy as np
import matplotlib.pyplot as plt

MISSING = -1
MAXR = 15
TURNS = np.arange(1, MAXR + 1)

AVAILABLE_MODELS = ["ML", "NL", "DQ", "DL"]
DEFAULT_MODELS = AVAILABLE_MODELS
BUDGETS = [128, 256, 384, 512]
TITLE = {
    "ML": "ML",
    "NL": "NL",
    "DQ": "DQ",
    "DL": "DL",
}

METHOD_ORDER = [
    "H2O",
    "KNorm",
    "R-KV",
    "SnapKV-D",
    "StreamingLLM",
    "SCOPE",
    "RPC",
]
PLOT_ORDER = ["Full", *METHOD_ORDER]
COLOR = {
    "Full": "#222222",
    "H2O": "#1fbfa0",
    "KNorm": "#e5442f",
    "R-KV": "#3b8fd4",
    "SnapKV-D": "#f5a623",
    "StreamingLLM": "#8e6ec8",
    "SCOPE": "#e377c2",
    "RPC": "#8c564b",
}
MARK = {
    "Full": "*",
    "H2O": "p",
    "KNorm": "D",
    "R-KV": "s",
    "SnapKV-D": "v",
    "StreamingLLM": "^",
    "SCOPE": "o",
    "RPC": "X",
}

NONE = ([MISSING] * MAXR, 0)

# --- Full accuracy reference: one curve per model ----------------------------
# Full does not evict KV entries, so its accuracy is cache-budget independent.
# fmt: off
FULL_ACC = {
    "ML": ([0.8900, 0.7059, 1.0000, 1.0000, 0.98182, 1.0000, 0.99000, 0.98000, 0.98889, 0.96667, 1.0000, 0.97500, 0.98571, 0.97143, 0.98333], 20),
    "NL": ([0.9343, 0.9627, 0.9279, 0.9429, 0.9589, 0.9670, 0.9477, 0.98844, 0.988444, 0.98800, 0.9403, 0.98333, 0.9879, 0.9826, 0.9614], 274),
    "DQ": ([0.8917, 0.8704, 0.87391, 0.93913, 0.93182, 0.94762, 0.94000, 0.93000, 0.93500, 0.9632, 0.94211, 0.95263, 0.94211, 0.95000, 0.96111], 120),
    "DL": ([0.9640, 0.95225, 0.97336, 0.96578, 0.98314, 0.97025, 0.97655, 0.97174, 0.96212, 0.97459, 0.96810, 0.96887, 0.96768, 0.97303, 0.96988], 356),
}
# fmt: on

# --- hardcoded half-scaled Full-minus-method differences ----------------------
# 7727 games / 52430 turns pooled over categories, game types, seeds and blocks.
# Keep one method per line: this large embedded data table is substantially
# easier to review and update in its compact representation.
# fmt: off
ACC = {
    ("ML", 512): {
        "H2O": ([0.03250, -0.03595, 0.08065, 0.05170, 0.09861, 0.07275, 0.08590, 0.06145, 0.06944, 0.07464, 0.10525, 0.09860, 0.12620, 0.13276, 0.16812], 40),
        "KNorm": ([0.23544, 0.26695, 0.24153, 0.24232, 0.29068, 0.29092, 0.28965, 0.29106, 0.29143], 111),
        "R-KV": ([0.01390, 0.02075, 0.10335, 0.06185, 0.05756, 0.06330, 0.06870, 0.08665, 0.08474, 0.07824, 0.09540, 0.10080, 0.10830, 0.09586, 0.12212], 312),
        "SnapKV-D": ([0.02000, -0.05880, 0.07145, 0.11540, 0.08181], 20),
        "StreamingLLM": ([0.00210, -0.03990, 0.07545, 0.03750, 0.02956, 0.05200, 0.07030, 0.06190, 0.06090, 0.05368, 0.05120, 0.06575, 0.05410, 0.09396, 0.07032], 254),
        "SCOPE": ([0.02155, -0.04535, 0.07145, 0.06485, 0.05431, 0.06500, 0.05685, 0.05215, 0.07264, 0.07104, 0.07414, 0.07084, 0.06900, 0.07312, 0.07136], 320),
        "RPC": ([0.01220, -0.04300, 0.05960, 0.05620, 0.06326, 0.06810, 0.06405, 0.07170, 0.08770, 0.03978, 0.10540, 0.10903, 0.11206, 0.09752, 0.12862], 320),
    },
    ("NL", 512): {
        "H2O": ([0.07515, 0.12115, 0.13150, 0.18685, 0.17175, 0.15756, 0.16254, 0.15657, 0.16142, 0.15849, 0.16280, 0.16345, 0.16060, 0.16413, 0.17932], 111),
        "KNorm": ([0.39672, 0.40990, 0.48060], 9),
        "R-KV": ([0.05595, 0.07590, 0.05930, 0.05430, 0.06175, 0.06015, 0.06825, 0.07057, 0.08292, 0.04805, 0.04960, 0.07582, 0.08675, 0.11895, 0.10290], 304),
        "SnapKV-D": ([0.27230, 0.26530, 0.26920, 0.25215, 0.27061, 0.31124, 0.30954], 16),
        "StreamingLLM": ([0.04572, 0.09245, 0.08895, 0.10480, 0.09485, 0.15015, 0.20110, 0.21642, 0.16087, 0.19400], 50),
        "SCOPE": ([0.07185, 0.06260, 0.02205, 0.01075, 0.01975, 0.03795, 0.04120, 0.08437, 0.07857, 0.07945, 0.04785, 0.06722, 0.08040, 0.07065, 0.06540], 320),
        "RPC": ([0.03120, 0.07470, 0.01805, 0.03225, 0.05275, 0.05000, 0.03960, 0.07267, 0.08797, 0.09185, 0.07230, 0.07266, 0.13640, 0.09625, 0.11765], 320),
    },
    ("DQ", 512): {
        "H2O": ([-0.01210, -0.00980, -0.02790, 0.00386, 0.01806, 0.03421, 0.03345, 0.02335, 0.03415, 0.04075, 0.05440, 0.04386, 0.06666, 0.07985, 0.05950], 226),
        "KNorm": ([0.24585, 0.32410, 0.32447, 0.38622], 13),
        "R-KV": ([0.14404, 0.11775, -0.02600, 0.17412, 0.10481, 0.31866, 0.35000, 0.36241, 0.37044, 0.36066, 0.35865, 0.36263, 0.36377, 0.36000, 0.36949], 92),
        "SnapKV-D": ([0.14118, 0.15220, 0.10020, 0.20362, 0.17701, 0.15986, 0.16050, 0.27450, 0.29250, 0.28425, 0.29536, 0.30966, 0.35996, 0.32220, 0.30910], 220),
        "StreamingLLM": ([0.00446, -0.00230, 0.00840, 0.05292, 0.16591, 0.27381, 0.26700, 0.26650, 0.26675, 0.26816, 0.26711, 0.27632, 0.27106, 0.27500, 0.28056], 22),
        "SCOPE": ([0.09194, 0.09287], 585),
        "RPC": ([0.02085, 0.04270, -0.00684, 0.03582, 0.04961, 0.04726, 0.05010, 0.06400, 0.05350, 0.05810, 0.06880, 0.07402, 0.09906, 0.08575, 0.10476], 320),
    },
    ("DL", 512): {
        "H2O": ([0.24370, 0.43292, 0.45478, 0.43920, 0.42012, 0.42262, 0.46198, 0.44025, 0.42226, 0.45604, 0.48405, 0.48444, 0.48384, 0.48652, 0.46569], 128),
        "KNorm": ([0.21200, 0.28382, 0.30718, 0.25919, 0.24872, 0.24982, 0.39738, 0.42337, 0.46381, 0.48730, 0.49340, 0.49055], 100),
        "R-KV": ([0.21640, 0.37612, 0.44123, 0.42734, 0.42907, 0.41368, 0.40492, 0.40859], 32),
        "SnapKV-D": ([0.24463, 0.27302, 0.26848, 0.23289, 0.25572, 0.27358, 0.28418, 0.25862, 0.27176, 0.28490, 0.23405, 0.26824, 0.26164, 0.28062, 0.24254], 168),
        "StreamingLLM": ([0.41600, 0.40148, 0.34193, 0.45584, 0.45102, 0.47162, 0.44662, 0.44422, 0.42226, 0.42850, 0.43995], 197),
        "SCOPE": ([0.23805, 0.30132, 0.26238, 0.29924, 0.42977, 0.43752, 0.35538, 0.39791, 0.36676], 266),
        "RPC": ([0.07748, 0.07461, 0.08419, 0.09579, 0.24157, 0.27082, 0.45702, 0.44742, 0.43106, 0.41584, 0.42840], 70),
    },
}
# fmt: on


def _binomial_se(probability, n_games):
    """Return the standard error of a binomial proportion."""
    if n_games <= 0 or probability == MISSING:
        return MISSING
    probability = float(np.clip(probability, 0.0, 1.0))
    return float(np.sqrt(probability * (1.0 - probability) / n_games))


def _build_acc_error():
    """Build per-turn half-accuracy SEs without modifying ACC estimates."""
    errors = {}
    for (model, budget), methods in ACC.items():
        full_values, full_n = FULL_ACC[model]
        errors[(model, budget)] = {
            "Full": [0.5 * _binomial_se(value, full_n) for value in full_values]
        }
        for method, (gaps, method_n) in methods.items():
            method_errors = []
            for index, gap in enumerate(gaps):
                if gap == MISSING:
                    method_errors.append(MISSING)
                    continue
                full_accuracy = full_values[index]
                method_accuracy = full_accuracy - 2.0 * gap
                method_se = _binomial_se(method_accuracy, method_n)
                method_errors.append(0.5 * method_se)
            errors[(model, budget)][method] = method_errors
    return errors


# Separate uncertainty table; ACC's central values above remain unchanged.
ACC_ERROR = _build_acc_error()

BASE_FONT_SIZE = 15
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["cmr10", "CMU Serif", "Computer Modern Roman"],
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": BASE_FONT_SIZE,
        "axes.labelsize": BASE_FONT_SIZE,
        "xtick.labelsize": BASE_FONT_SIZE - 3,
        "ytick.labelsize": BASE_FONT_SIZE - 3,
        "legend.fontsize": BASE_FONT_SIZE - 2,
        "lines.linewidth": 1.7,
    }
)


def masked(values):
    """-1 -> nan so matplotlib leaves a gap instead of plotting a fake point."""
    a = np.asarray(values, dtype=float)
    a[a == MISSING] = np.nan
    return a


def any_measured(values):
    return bool(np.isfinite(masked(values)).any())


def cumulative_average(values):
    """Return 15 cumulative means, treating post-stop turns as zero.

    Missing turns within the measured range remain gaps. After the final
    measurement, the measured sum stays fixed while the denominator continues
    to increase with the turn number.
    """
    raw = masked(values)[:MAXR]
    result = np.full(MAXR, np.nan)
    measured = np.flatnonzero(np.isfinite(raw))
    if measured.size == 0:
        return result

    total = 0.0
    last_measured = int(measured[-1])
    for index in range(MAXR):
        if index < raw.size and np.isfinite(raw[index]):
            total += raw[index]
            result[index] = total / (index + 1)
        elif index > last_measured:
            result[index] = total / (index + 1)
    return result


def cumulative_standard_error(values):
    """Propagate independent per-turn SEs through the cumulative average."""
    raw = masked(values)[:MAXR]
    result = np.full(MAXR, np.nan)
    measured = np.flatnonzero(np.isfinite(raw))
    if measured.size == 0:
        return result

    variance = 0.0
    last_measured = int(measured[-1])
    for index in range(MAXR):
        if index < raw.size and np.isfinite(raw[index]):
            variance += raw[index] ** 2
            result[index] = np.sqrt(variance) / (index + 1)
        elif index > last_measured:
            result[index] = np.sqrt(variance) / (index + 1)
    return result


def series(mo, bud, meth):
    """Return one precomputed 0.5 * (Full-minus-method) curve."""
    return ACC.get((mo, bud), {}).get(meth, NONE)


def accuracy_series(mo, bud, meth):
    """Return a half-scaled Full or reconstructed method-accuracy curve."""
    full_values, full_n = FULL_ACC[mo]
    if meth == "Full":
        values = [
            0.5 * value if value != MISSING else MISSING for value in full_values
        ]
        return values, full_n

    gaps, method_n = series(mo, bud, meth)
    values = [
        MISSING if gap == MISSING else 0.5 * full_values[index] - gap
        for index, gap in enumerate(gaps)
    ]
    return values, method_n


def error_series(mo, bud, meth):
    """Return one precomputed per-turn standard-error curve."""
    return ACC_ERROR.get((mo, bud), {}).get(meth, [MISSING] * MAXR)


def draw_panel(ax, mo, bud):
    """Plot one (model, budget) cell. Returns list of methods with no data."""
    absent = []
    plotted = 0

    for meth in PLOT_ORDER:
        y, _ = accuracy_series(mo, bud, meth)
        if not any_measured(y):
            absent.append(meth)
            continue
        ym = cumulative_average(y)
        ye = cumulative_standard_error(error_series(mo, bud, meth))
        ax.fill_between(
            TURNS,
            ym - ye,
            ym + ye,
            color=COLOR[meth],
            alpha=0.16,
            linewidth=0.0,
            zorder=2,
        )
        ax.plot(
            TURNS,
            ym,
            color=COLOR[meth],
            marker=MARK[meth],
            markersize=4.2,
            markeredgewidth=0.0,
            linewidth=1.7,
            zorder=3,
            label=meth,
        )
        plotted += 1

    ax.axhline(0.0, color="0.4", linewidth=0.9, linestyle="--", zorder=1)

    if plotted == 0:
        ax.text(
            0.5,
            0.5,
            "no data",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="0.55",
            fontsize=BASE_FONT_SIZE - 1,
        )
    elif absent:
        note = textwrap.fill("not measured: " + ", ".join(absent), width=34)
        ax.text(
            0.015,
            0.02,
            note,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=BASE_FONT_SIZE - 5,
            color="0.45",
            linespacing=1.25,
        )

    ax.set_xlim(0.7, MAXR + 0.3)
    ax.set_xticks([1, 3, 5, 7, 9, 11, 13, 15])
    ax.grid(True, color="0.88", linewidth=0.7)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_linewidth(0.8)
    return absent


def shared_legend(fig, axes, y=-0.02):
    handles, labels = [], []
    for ax in np.atleast_1d(axes).flat:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    if not labels:
        return
    idx = [labels.index(m) for m in PLOT_ORDER if m in labels]
    fig.legend(
        [handles[k] for k in idx],
        [labels[k] for k in idx],
        loc="lower center",
        ncol=len(idx),
        bbox_to_anchor=(0.5, y),
        frameon=True,
        columnspacing=1.5,
        handletextpad=0.5,
    )


def accuracy_axis_limits(configurations):
    """Return shared limits containing curves and their cumulative error bands."""
    lower_values = []
    upper_values = []
    for mo, bud in configurations:
        for meth in PLOT_ORDER:
            values = cumulative_average(accuracy_series(mo, bud, meth)[0])
            errors = cumulative_standard_error(error_series(mo, bud, meth))
            finite = np.isfinite(values) & np.isfinite(errors)
            lower_values.extend((values[finite] - errors[finite]).tolist())
            upper_values.extend((values[finite] + errors[finite]).tolist())
    if not lower_values:
        return -0.05, 0.05
    lower = min(0.0, min(lower_values))
    upper = max(0.0, max(upper_values))
    padding = max(0.03, 0.05 * (upper - lower))
    return lower - padding, upper + padding


def grid_figure(models):
    """Draw every selected configuration from ACC in a single row."""
    ylab = "Cumulative Accuracy"
    configurations = [(mo, bud) for mo, bud in ACC if mo in models]
    if not configurations:
        raise ValueError("No ACC configurations match --models")
    fig, axes = plt.subplots(
        1,
        len(configurations),
        figsize=(3.5 * len(configurations), 4.0),
        sharex=True,
        sharey=False,
        squeeze=False,
    )
    for index, (mo, bud) in enumerate(configurations):
        ax = axes[0][index]
        draw_panel(ax, mo, bud)
        ax.set_title(f"{TITLE[mo]}", fontsize=BASE_FONT_SIZE)
        ax.set_xlabel("Turn Stopped")
        ax.set_ylim(*accuracy_axis_limits([(mo, bud)]))
        if index == 0:
            ax.set_ylabel(ylab)
    shared_legend(fig, axes, y=0.01)
    fig.subplots_adjust(
        left=0.065, right=0.99, top=0.82, bottom=0.25, hspace=0.16, wspace=0.12
    )
    out = "mtr_accuracy_drop_vs_turn.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def coverage_report(models):
    """Print which cells are still all -1, so gaps are visible without plotting."""
    print("\ncoverage (measured turns / 15, n games):")
    for mo, bud in ((mo, bud) for mo, bud in ACC if mo in models):
        bits = []
        for meth in METHOD_ORDER:
            y, n = series(mo, bud, meth)
            k = int(np.isfinite(masked(y)).sum())
            bits.append(f"{meth}={k}/15(n={n})")
        print(f"  {mo}, budget {bud}: " + "  ".join(bits))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--coverage", action="store_true", help="print which cells are still unmeasured"
    )
    ap.add_argument(
        "--models",
        nargs="+",
        choices=AVAILABLE_MODELS,
        default=DEFAULT_MODELS,
        help="models to include (default: all models represented in ACC)",
    )
    args = ap.parse_args()

    grid_figure(args.models)
    if args.coverage:
        coverage_report(args.models)


if __name__ == "__main__":
    main()
