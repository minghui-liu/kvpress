"""Radar chart of category-level accuracy (Math / Logic / Reading / Commonsense)
for three models under different KV-cache compression methods.

Category aggregation follows the dataset grouping declared in the paper's
experimental setup (accuracy averaged over cache budgets 128-512 for the
compressed methods; Full is budget-independent):
    Math        = mean(GSM8K, MATH-500)
    Logic       = mean(StrategyQA, FOLIO)
    Reading     = mean(DROP, ReClor)
    Commonsense = mean(OBQA, CSQA)
"""

import numpy as np
import matplotlib.pyplot as plt

BASE_FONT_SIZE = 22
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": BASE_FONT_SIZE,
        "legend.fontsize": BASE_FONT_SIZE,
        "axes.titlesize": BASE_FONT_SIZE + 2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

# Axis order: Math (top), Logic (right), Reading (bottom), Commonsense (left)
CATEGORIES = ["Math", "Logic", "Reading", "Commonsense"]

# values: [Math, Logic, Reading, Commonsense]
DATA = {
    "R1-Distill-Qwen-7B": {
        "Full":     [0.705, 0.595, 0.355, 0.725],
        "H2O":      [0.398, 0.395, 0.079, 0.589],
        "SnapKV-D": [0.251, 0.322, 0.046, 0.671],
        "SCOPE":    [0.726, 0.518, 0.315, 0.674],
        "RPC":      [0.718, 0.530, 0.338, 0.667],
    },
    "Nemotron-Nano-8B": {
        "Full":     [0.665, 0.750, 0.340, 0.575],
        "H2O":      [0.386, 0.610, 0.188, 0.546],
        "SnapKV-D": [0.312, 0.505, 0.078, 0.583],
        "SCOPE":    [0.674, 0.715, 0.320, 0.559],
        "RPC":      [0.667, 0.711, 0.328, 0.570],
    },
    "Llama-3.1-8B-Instruct": {
        "Full":     [0.530, 0.640, 0.355, 0.805],
        "H2O":      [0.297, 0.605, 0.269, 0.806],
        "SnapKV-D": [0.295, 0.560, 0.269, 0.719],
        "SCOPE":    [0.436, 0.624, 0.355, 0.761],
        "RPC":      [0.453, 0.626, 0.366, 0.746],
    },
}

# Same aggregation for DeepSeek-R1-Distill-Llama-8B (not plotted; move into DATA
# and widen the figure if a fourth panel is wanted):
#   "R1-Distill-Llama-8B": {
#       "Full":     [0.875, 0.685, 0.375, 0.795],
#       "H2O":      [0.451, 0.500, 0.129, 0.699],
#       "SnapKV-D": [0.320, 0.417, 0.041, 0.779],
#       "SCOPE":    [0.713, 0.626, 0.361, 0.791],
#       "RPC":      [0.705, 0.630, 0.376, 0.794],
#   },

STYLES = {
    "Full":     {"color": "#1f77b4", "linestyle": "-",  "marker": "o"},
    "H2O":      {"color": "#ff7f0e", "linestyle": "--", "marker": "s"},
    "SnapKV-D": {"color": "#2ca02c", "linestyle": "--", "marker": "^"},
    "SCOPE":    {"color": "#9467bd", "linestyle": "--", "marker": "D"},
    "RPC":      {"color": "#8c564b", "linestyle": "--", "marker": "X"},
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
            linewidth=3.0,
            marker=style["marker"],
            markersize=9.0,
            markeredgecolor="white",
            markeredgewidth=0.7,
            label=method,
            zorder=3,
        )
        ax.fill(angs, vals, color=style["color"], alpha=0.075, zorder=2)

    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.5, 0.8])
    ax.set_yticklabels(
        [r"\textbf{20\%}", r"\textbf{50\%}", r"\textbf{80\%}"],
        color="#555555",
        fontsize=BASE_FONT_SIZE - 2,
    )
    ax.set_rlabel_position(58)

    # Keep the angular grid spokes but hide the default (always-horizontal)
    # tick labels; we place our own so the side labels can be vertical.
    ax.set_xticks(ANGLES)
    ax.set_xticklabels([])

    # Manual axis labels: Math (top) and Reading (bottom) horizontal;
    # Logic (right) and Commonsense (left) vertical, matching the original.
    label_r = 1.08
    for angle, cat in zip(ANGLES, CATEGORIES):
        alignment = {
            "Math": ("center", "bottom", 0),
            "Logic": ("left", "center", -90),
            "Reading": ("center", "top", 0),
            "Commonsense": ("right", "center", 90),
        }
        ha, va, rot = alignment[cat]
        ax.text(
            angle,
            label_r,
            r"\textbf{" + cat + "}",
            fontsize=BASE_FONT_SIZE,
            ha=ha,
            va=va,
            rotation=rot,
            rotation_mode="anchor",
        )

    ax.grid(True, color="#8a8a8a", linestyle="--", linewidth=1.15, alpha=0.68)
    ax.spines["polar"].set_visible(False)


def main():
    fig, axes = plt.subplots(
        1, 3, figsize=(19.5, 6.2), subplot_kw={"projection": "polar"}
    )

    for ax, (model, model_data) in zip(axes, DATA.items()):
        plot_radar(ax, model_data)
        ax.set_title(
            r"\textbf{" + model + "}",
            y=-0.22,
            fontsize=BASE_FONT_SIZE + 2,
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        bbox_to_anchor=(0.5, -0.10),
        frameon=True,
        columnspacing=1.8,
        handlelength=2.6,
        handletextpad=0.7,
    )

    fig.subplots_adjust(left=0.045, right=0.955, top=0.96, bottom=0.20, wspace=0.34)

    out_file = "scripts/radar_categories.pdf"
    fig.savefig(out_file, bbox_inches="tight")
    png_file = "scripts/radar_categories.png"
    fig.savefig(png_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figures to {out_file} and {png_file}")


if __name__ == "__main__":
    main()
