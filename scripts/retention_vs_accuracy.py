"""Figure 8: Critical Token Retention vs. Downstream (GSM8K) Accuracy.

One subplot per model. Each eviction method is a trajectory across cache budgets
[128, 256, 384, 512]; marker size grows with budget. X = critical-token retention
rate (%), Y = GSM8K accuracy.

Retention (x) from the Critical-Token-Retention table; accuracy (y) from the
main per-model results tables.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

BASE_FONT_SIZE = 14
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": BASE_FONT_SIZE,
        "axes.labelsize": BASE_FONT_SIZE,
        "xtick.labelsize": BASE_FONT_SIZE - 3,
        "ytick.labelsize": BASE_FONT_SIZE - 3,
        "legend.fontsize": BASE_FONT_SIZE - 3,
    }
)

BUDGETS = [128, 256, 384, 512]
BUDGET_SIZES = [45, 100, 175, 270]

METHODS = ["H2O", "SnapKV-D", "StreamingLLM", "KNorm"]
STYLE = {
    "H2O":          {"color": "#1f77b4", "marker": "o"},
    "SnapKV-D":     {"color": "#ff7f0e", "marker": "s"},
    "StreamingLLM": {"color": "#2ca02c", "marker": "^"},
    "KNorm":        {"color": "#d62728", "marker": "D"},
}

RET = {
    "Llama-3.1": {
        "H2O":          [68.11, 68.11, 69.19, 70.27],
        "SnapKV-D":     [68.11, 68.11, 68.65, 69.73],
        "StreamingLLM": [67.03, 67.03, 67.57, 69.73],
        "KNorm":        [68.11, 68.11, 69.19, 69.73],
    },
    "R1-Distill-Qwen-7B": {
        "H2O":          [72.30, 72.30, 73.24, 74.18],
        "SnapKV-D":     [72.30, 72.30, 73.24, 74.18],
        "StreamingLLM": [71.36, 71.36, 72.30, 73.24],
        "KNorm":        [67.60, 67.60, 68.72, 69.27],
    },
    "Nemotron-8B": {
        "H2O":          [68.11, 68.11, 69.19, 70.27],
        "SnapKV-D":     [68.11, 68.11, 69.19, 70.27],
        "StreamingLLM": [67.03, 67.03, 68.11, 69.19],
        "KNorm":        [65.48, 65.48, 66.67, 67.26],
    },
    "R1-Distill-Llama-8B": {
        "H2O":          [71.61, 71.61, 72.30, 72.30],
        "SnapKV-D":     [68.11, 68.11, 69.19, 70.27],
        "StreamingLLM": [67.03, 67.03, 68.11, 69.19],
        "KNorm":        [68.11, 68.11, 69.19, 69.73],
    },
}

ACC = {
    "Llama-3.1": {
        "H2O":          [0.25, 0.44, 0.51, 0.56],
        "SnapKV-D":     [0.29, 0.51, 0.56, 0.62],
        "StreamingLLM": [0.07, 0.43, 0.52, 0.52],
        "KNorm":        [0.02, 0.21, 0.39, 0.49],
    },
    "R1-Distill-Qwen-7B": {
        "H2O":          [0.17, 0.46, 0.60, 0.68],
        "SnapKV-D":     [0.08, 0.33, 0.47, 0.65],
        "StreamingLLM": [0.02, 0.18, 0.26, 0.33],
        "KNorm":        [0.00, 0.01, 0.04, 0.06],
    },
    "Nemotron-8B": {
        "H2O":          [0.22, 0.43, 0.53, 0.54],
        "SnapKV-D":     [0.06, 0.38, 0.49, 0.59],
        "StreamingLLM": [0.02, 0.16, 0.39, 0.49],
        "KNorm":        [0.01, 0.01, 0.04, 0.13],
    },
    "R1-Distill-Llama-8B": {
        "H2O":          [0.34, 0.56, 0.66, 0.73],
        "SnapKV-D":     [0.10, 0.26, 0.40, 0.42],
        "StreamingLLM": [0.02, 0.16, 0.26, 0.28],
        "KNorm":        [0.01, 0.07, 0.14, 0.21],
    },
}

MODEL_ORDER = ["Llama-3.1", "R1-Distill-Qwen-7B", "Nemotron-8B", "R1-Distill-Llama-8B"]


def plot_model(ax, model):
    for method in METHODS:
        x = RET[model][method]
        y = ACC[model][method]
        s = STYLE[method]
        ax.plot(x, y, color=s["color"], linewidth=1.5, alpha=0.75, zorder=1)
        ax.scatter(x, y, s=BUDGET_SIZES, marker=s["marker"], facecolor=s["color"],
                   edgecolor="black", linewidths=0.6, alpha=0.9, zorder=2)
    ax.set_title(r"\textbf{" + model + "}", loc="left",
                 fontsize=BASE_FONT_SIZE - 1, y=0.88, x=0.03)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_ylim(-0.05, 0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    fig = plt.figure(figsize=(11, 7))
    gs = gridspec.GridSpec(2, 2, hspace=0.28, wspace=0.18)
    axes = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(4)]

    for ax, model in zip(axes, MODEL_ORDER):
        plot_model(ax, model)

    for i, ax in enumerate(axes):
        if i % 2 == 0:
            ax.set_ylabel(r"\textbf{GSM8K Accuracy}")
        if i // 2 == 1:
            ax.set_xlabel(r"\textbf{Critical Token Retention (\%)}")

    # ---- Budget-size legend ----
    size_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="0.5",
               markeredgecolor="black", markersize=np.sqrt(sz), label=f"Budget {b}")
        for b, sz in zip(BUDGETS, BUDGET_SIZES)
    ]
    leg1 = fig.legend(handles=size_handles, loc="lower center", ncol=4,
                      bbox_to_anchor=(0.5, -0.02), frameon=True, columnspacing=1.8,
                      handletextpad=0.4)
    fig.add_artist(leg1)

    # ---- Method legend ----
    method_handles = [
        Line2D([0], [0], marker=STYLE[m]["marker"], color=STYLE[m]["color"],
               markerfacecolor=STYLE[m]["color"], markeredgecolor="black",
               markersize=9, linewidth=2, label=m)
        for m in METHODS
    ]
    fig.legend(handles=method_handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.08), frameon=True, columnspacing=1.5,
               handletextpad=0.4)

    fig.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.14)
    out_file = "scripts/retention_vs_accuracy.pdf"
    fig.savefig(out_file, bbox_inches="tight")
    print(f"Saved figure to {out_file}")


if __name__ == "__main__":
    main()
