"""Radar chart of category-level accuracy (Math / Logic / Reading / Commonsense)
for three models under different KV-cache compression methods.

Category aggregation (accuracy averaged over cache budgets 128-512 for the
compressed methods; Full is budget-independent):
    Math        = mean(GSM8K, Math500)
    Logic       = mean(ReClor, FOLIO)
    Reading     = DROP
    Commonsense = mean(CSQA, OBQA, StrategyQA)
"""

import numpy as np
import matplotlib.pyplot as plt

BASE_FONT_SIZE = 13
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": BASE_FONT_SIZE,
        "legend.fontsize": BASE_FONT_SIZE,
    }
)

# Axis order: Math (top), Logic (right), Reading (bottom), Commonsense (left)
CATEGORIES = ["Math", "Logic", "Reading", "Commonsense"]

# values: [Math, Logic, Reading, Commonsense]
DATA = {
    "R1-Distill-Qwen-7B": {
        "Full":     [0.755, 0.535, 0.160, 0.707],
        "H2O":      [0.319, 0.128, 0.085, 0.595],
        "SnapKV-D": [0.201, 0.031, 0.077, 0.647],
        "SCOPE":    [0.726, 0.486, 0.133, 0.636],
        "RPC":      [0.718, 0.509, 0.145, 0.636],
    },
    "Nemotron-Nano-8B": {
        "Full":     [0.665, 0.580, 0.130, 0.680],
        "H2O":      [0.386, 0.341, 0.092, 0.638],
        "SnapKV-D": [0.312, 0.128, 0.070, 0.668],
        "SCOPE":    [0.674, 0.551, 0.120, 0.655],
        "RPC":      [0.667, 0.555, 0.122, 0.662],
    },
    "Llama-3.1-8B-Inst.": {
        "Full":     [0.528, 0.505, 0.150, 0.813],
        "H2O":      [0.286, 0.379, 0.120, 0.825],
        "SnapKV-D": [0.280, 0.365, 0.120, 0.740],
        "SCOPE":    [0.416, 0.511, 0.138, 0.761],
        "RPC":      [0.430, 0.519, 0.135, 0.753],
    },
}

STYLES = {
    "Full":     {"color": "#1f77b4", "linestyle": "-",  "fill": True},
    "H2O":      {"color": "#ff7f0e", "linestyle": "--", "fill": False},
    "SnapKV-D": {"color": "#2ca02c", "linestyle": "--", "fill": True},
    "SCOPE":    {"color": "#9467bd", "linestyle": "--", "fill": False},
    "RPC":      {"color": "#8c564b", "linestyle": "--", "fill": False},
}

# Angles: Math at 90deg (top), Logic at 0deg (right), Reading at 270deg, Commonsense at 180deg
ANGLES = np.deg2rad([90, 0, 270, 180])


def plot_radar(ax, model_data):
    # close the polygon: order = Math, Logic, Reading, Commonsense
    for method, values in model_data.items():
        style = STYLES[method]
        vals = np.array(values + values[:1])
        angs = np.concatenate([ANGLES, ANGLES[:1]])
        ax.plot(
            angs,
            vals,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.8,
            label=method,
        )
        if style["fill"]:
            ax.fill(angs, vals, color=style["color"], alpha=0.12)

    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.5, 0.8])
    ax.set_yticklabels([r"20\%", r"50\%", r"80\%"], color="gray", fontsize=BASE_FONT_SIZE - 4)
    ax.set_rlabel_position(62)

    # Keep the angular grid spokes but hide the default (always-horizontal)
    # tick labels; we place our own so the side labels can be vertical.
    ax.set_xticks(ANGLES)
    ax.set_xticklabels([])

    # Manual axis labels: Math (top) and Reading (bottom) horizontal;
    # Logic (right) and Commonsense (left) vertical, matching the original.
    label_r = 1.28
    for angle, cat in zip(ANGLES, CATEGORIES):
        rot = 90 if cat in ("Logic", "Commonsense") else 0
        ax.text(
            angle,
            label_r,
            r"\textbf{" + cat + "}",
            fontsize=BASE_FONT_SIZE,
            ha="center",
            va="center",
            rotation=rot,
            rotation_mode="anchor",
        )

    ax.grid(True, linestyle="--", alpha=0.6)
    ax.spines["polar"].set_visible(False)


def main():
    fig, axes = plt.subplots(
        1, 3, figsize=(13.5, 4.6), subplot_kw={"projection": "polar"}
    )

    for ax, (model, model_data) in zip(axes, DATA.items()):
        plot_radar(ax, model_data)
        ax.set_title(
            r"\textbf{" + model + "}",
            y=-0.32,
            fontsize=BASE_FONT_SIZE,
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        bbox_to_anchor=(0.5, -0.06),
        frameon=True,
        columnspacing=1.6,
    )

    fig.subplots_adjust(left=0.06, right=0.96, top=0.92, bottom=0.18, wspace=0.65)

    out_file = "scripts/radar_categories.pdf"
    fig.savefig(out_file, bbox_inches="tight")
    print(f"Saved figure to {out_file}")


if __name__ == "__main__":
    main()
