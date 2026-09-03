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
        "Full":     [0.785, 0.585, 0.555, 0.685],
        "H2O":      [0.518, 0.312, 0.316, 0.593],
        "SnapKV-D": [0.359, 0.236, 0.198, 0.517],
        "SCOPE":    [0.790, 0.559, 0.520, 0.664],
        "RPC":      [0.794, 0.546, 0.570, 0.654],
    },
    "Nemotron-Nano-8B": {
        "Full":     [0.705, 0.710, 0.605, 0.570],
        "H2O":      [0.520, 0.590, 0.419, 0.532],
        "SnapKV-D": [0.406, 0.415, 0.266, 0.511],
        "SCOPE":    [0.721, 0.706, 0.537, 0.554],
        "RPC":      [0.716, 0.696, 0.575, 0.569],
    },
    "Llama-3.1-8B-Instruct": {
        "Full":     [0.465, 0.625, 0.550, 0.785],
        "H2O":      [0.416, 0.585, 0.470, 0.730],
        "SnapKV-D": [0.374, 0.560, 0.448, 0.681],
        "SCOPE":    [0.479, 0.629, 0.521, 0.732],
        "RPC":      [0.475, 0.628, 0.491, 0.735],
    },
}

# Same aggregation for DeepSeek-R1-Distill-Llama-8B (not plotted; move into DATA
# and widen the figure if a fourth panel is wanted):
#   "R1-Distill-Llama-8B": {
#   "Full":     [0.700, 0.635, 0.460, 0.635],
#   "H2O":      [0.451, 0.348, 0.163, 0.535],
#   "SnapKV-D": [0.320, 0.254, 0.155, 0.526],
#   "SCOPE":    [0.713, 0.593, 0.297, 0.637],
#   "RPC":      [0.705, 0.601, 0.326, 0.638],
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
