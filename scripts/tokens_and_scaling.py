"""Figure 2: Generation Number of Tokens and Model Scaling.

Left  : Mean output tokens on MATH-500 (7B), showing generation bloat for
        aggressive eviction methods (KNorm).
Middle: GSM8K accuracy scaling from 7B -> 14B (DeepSeek-R1-Distill-Qwen).
Right : MATH-500 accuracy scaling from 7B -> 14B.

SCOPE and RPC values are budget-averaged from the score JSONs; the eviction
baselines match the KVbench measurements. Full GSM8K (7B) set to 0.80.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

BASE_FONT_SIZE = 14
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": BASE_FONT_SIZE,
        "axes.titlesize": BASE_FONT_SIZE,
        "axes.labelsize": BASE_FONT_SIZE,
        "xtick.labelsize": BASE_FONT_SIZE - 2,
        "ytick.labelsize": BASE_FONT_SIZE - 2,
        "legend.fontsize": BASE_FONT_SIZE - 2,
        "lines.linewidth": 2.0,
        "axes.linewidth": 0.4,
    }
)

# Top-to-bottom bar order: Full, then the two path-aware methods, then eviction.
METHODS = ["Full", "SCOPE", "RPC", "H2O", "SnapKV-D", "StreamingLLM", "KNorm"]

STYLE = {
    "Full":         {"color": "black",   "marker": "o"},
    "SCOPE":        {"color": "#9467bd",  "marker": "P"},
    "RPC":          {"color": "#8c564b",  "marker": "X"},
    "H2O":          {"color": "#1f77b4",  "marker": "s"},
    "SnapKV-D":     {"color": "#ff7f0e",  "marker": "^"},
    "StreamingLLM": {"color": "#2ca02c",  "marker": "D"},
    "KNorm":        {"color": "#d62728",  "marker": "v"},
}

# Left panel: MATH-500 mean output tokens (7B).
MATH_TOKENS = {
    "Full":         2600,
    "SCOPE":        2815,   # extracted, budget-avg
    "RPC":          2977,   # extracted, budget-avg
    "H2O":          4600,
    "SnapKV-D":     4300,
    "StreamingLLM": 2600,
    "KNorm":       10000,
}

# Middle panel: GSM8K accuracy at [7B, 14B].
GSM8K_ACC = {
    "Full":         [0.80, 0.84],   # 7B revised to 0.80
    "SCOPE":        [0.81, 0.85],   # extracted
    "RPC":          [0.79, 0.86],   # extracted
    "H2O":          [0.57, 0.77],
    "SnapKV-D":     [0.51, 0.73],
    "StreamingLLM": [0.33, 0.54],
    "KNorm":        [0.06, 0.07],
}

# Right panel: MATH-500 accuracy at [7B, 14B].
MATH_ACC = {
    "Full":         [0.61, 0.68],
    "SCOPE":        [0.57, 0.65],   # extracted
    "RPC":          [0.59, 0.68],   # extracted
    "H2O":          [0.37, 0.52],
    "SnapKV-D":     [0.21, 0.46],
    "StreamingLLM": [0.17, 0.30],
    "KNorm":        [0.03, 0.005],
}

X = [0, 1]              # 7B, 14B
X_LABELS = ["7B", "14B"]

# Fixed bar thickness / gap so each bar keeps the original width regardless of
# how many methods are drawn.
BAR_HEIGHT = 0.82
BAR_STEP = 1.0


def full_box(ax):
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(0.4)


def plot_bars(ax):
    ypos = np.arange(len(METHODS))[::-1] * BAR_STEP   # first method at top
    for y, m in zip(ypos, METHODS):
        ax.barh(y, MATH_TOKENS[m], color=STYLE[m]["color"],
                edgecolor="black", linewidth=1.0, height=BAR_HEIGHT)
    ax.axvline(MATH_TOKENS["Full"], color="black", linestyle="--", linewidth=1.3)
    ax.set_yticks(ypos)
    ax.set_yticklabels(METHODS)
    for lbl in ax.get_yticklabels():
        lbl.set_style("italic")
        lbl.set_rotation(30)
        lbl.set_va("center")
        lbl.set_ha("right")
    ax.set_ylim(ypos.min() - BAR_STEP * 0.6, ypos.max() + BAR_STEP * 0.6)
    ax.set_xlabel(r"\textbf{MATH-500 Tokens}")
    ax.set_xlim(0, 10800)
    ax.set_xticks([0, 5000, 10000])
    ax.tick_params(axis="both", direction="out")
    full_box(ax)


def plot_lines(ax, acc, ylabel, ylim, yticks):
    for m in METHODS:
        s = STYLE[m]
        lw = 2.4 if m == "Full" else 2.0
        ax.plot(X, acc[m], color=s["color"], marker=s["marker"],
                markersize=8, markeredgecolor="black", markeredgewidth=0.4,
                linewidth=lw, label=m, zorder=3 if m == "Full" else 2)
    ax.set_xticks(X)
    ax.set_xticklabels([r"\textbf{" + t + "}" for t in X_LABELS])
    ax.set_xlim(-0.22, 1.22)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", linestyle="--", alpha=0.45)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", direction="out")
    full_box(ax)


def main():
    fig = plt.figure(figsize=(8, 3))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.12, 1, 1], wspace=0.34)

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])

    plot_bars(ax0)
    plot_lines(ax1, GSM8K_ACC, r"\textbf{GSM8K Acc.}", (0.02, 0.88),
               [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    plot_lines(ax2, MATH_ACC, r"\textbf{MATH-500 Acc.}", (-0.03, 0.73),
               [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", ncol=4,
        bbox_to_anchor=(0.5, -0.20),
        frameon=True, columnspacing=1.6, handletextpad=0.5,
    )

    fig.subplots_adjust(left=0.13, right=0.98, top=0.96, bottom=0.22)

    out_file = "scripts/tokens_and_scaling.pdf"
    fig.savefig(out_file, bbox_inches="tight")
    print(f"Saved figure to {out_file}")


if __name__ == "__main__":
    main()
