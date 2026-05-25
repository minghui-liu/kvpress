import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import seaborn as sns

BASE_FONT_SIZE = 15
plt.rcParams.update({
    "text.usetex": True,
    "font.size": BASE_FONT_SIZE,
    "axes.titlesize": BASE_FONT_SIZE - 2,
    "axes.labelsize": BASE_FONT_SIZE,
    "xtick.labelsize": BASE_FONT_SIZE - 2,
    "ytick.labelsize": BASE_FONT_SIZE - 2,
    "legend.fontsize": BASE_FONT_SIZE - 2,
    "lines.linewidth": 2.0,
})

budgets = [128, 256, 384, 512]

# GSM8K mean output token lengths.
# Primary source: max_new_tokens=5096, seeds {24, 42, 130}.
# Fallback to max_new_tokens=16384 only when 5096 has no runs:
#   - Nvidia Nemotron StreamingLLM (all 4 budgets)
#   - DeepSeek R1 Distill Llama 8B StreamingLLM (all 4 budgets)
data = {
    "Nvidia Llama 3.1 Nemotron Nano 8B": {
        "Full": [1429.04, 1429.04, 1429.04, 1429.04],
        "H2O": [3038.87, 2657.84, 1962.29, 2171.64],
        "Knorm": [4713.96, 4599.87, 4531.52, 4190.94],
        "RKV": [817.35, 1469.20, 2642.46, 2634.30],
        "SnapKV-D": [1937.04, 1906.43, 1800.48, 1588.00],
        "StreamingLLM": [1516.62, 1555.52, 1514.99, 1427.08],  # max=16384 fallback
    },
    "DeepSeek R1 Distill Llama 8B": {
        "Full": [1188.30, 1188.30, 1188.30, 1188.30],
        "H2O": [2128.69, 1583.80, 1454.26, 1421.01],
        "Knorm": [4241.47, 4420.30, 3944.60, 3615.37],
        "RKV": [2262.73, 3756.56, 3370.00, 1572.47],
        "SnapKV-D": [1206.26, 1217.73, 1226.48, 1329.38],
        "StreamingLLM": [2247.70, 1400.47, 1430.38, 1379.00],  # max=16384 fallback
    },
    "DeepSeek R1 Distill Qwen 7B": {
        "Full": [888.69, 888.69, 888.69, 888.69],
        "H2O": [1629.62, 1920.71, 1864.74, 1761.50],
        "Knorm": [4927.38, 4729.75, 4446.63, 4414.22],
        "RKV": [2666.97, 2092.44, 2221.32, 2454.65],
        "SnapKV-D": [1591.89, 1572.69, 1786.56, 1609.07],
        "StreamingLLM": [2022.07, 1311.57, 1008.07, 1034.27],
    },
    "Meta Llama 3.1 8B Instruct": {
        "Full": [1460.26, 1460.26, 1460.26, 1460.26],
        "H2O": [1582.11, 1470.93, 1562.48, 1472.81],
        "Knorm": [5046.60, 3845.33, 2563.82, 2105.97],
        "RKV": [4821.16, 3799.74, 2941.62, 2303.92],
        "SnapKV-D": [1256.65, 1724.68, 1585.21, 1478.36],
        "StreamingLLM": [1259.19, 1180.28, 1261.63, 1448.29],
    }
}

colors = sns.color_palette("tab10").as_hex()
colors = ['black'] + colors
methods_config = {
    "Full": {"color": colors[0], "marker": None, "linestyle": "--", "label": "Full"},
    "H2O": {"color": colors[1], "marker": "s", "linestyle": "-", "label": "H2O", "markerfacecolor": "none"},
    "Knorm": {"color": colors[2], "marker": "^", "linestyle": "-", "label": "KNorm", "markerfacecolor": "none"},
    "RKV": {"color": colors[3], "marker": "o", "linestyle": "-", "label": "RKV", "markerfacecolor": "none"},
    "SnapKV-D": {"color": colors[4], "marker": "d", "linestyle": "-", "label": "SnapKV-D", "markerfacecolor": "none"},
    "StreamingLLM": {"color": colors[5], "marker": "x", "linestyle": "-", "label": "StreamingLLM"},
}


def plot_single_ax(ax, model_name, data_dict):
    for method_name, config in methods_config.items():
        y_values = data_dict[method_name]

        if method_name == "Full":
            ax.plot(budgets, y_values,
                    color=config["color"],
                    linestyle=config["linestyle"],
                    linewidth=1.7,
                    label=config["label"],
                    alpha=0.7,
                    zorder=2)
        else:
            ax.plot(budgets, y_values,
                    color=config["color"],
                    marker='.',
                    linestyle=config["linestyle"],
                    linewidth=2.2,

                    markersize=10,

                    label=config["label"],
                    alpha=0.85,
                    zorder=1)

    ax.set_title(r"\textbf{" + model_name.replace(" ", r"\ ") + "}", fontweight='bold', pad=10)

    ax.grid(True, linestyle="--", alpha=0.4)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')

    ax.set_xticks(budgets)
    ax.set_xlim(110, 530)


def main():
    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])

    model_names = list(data.keys())
    axes = []

    for i in range(4):
        row = i // 2
        col = i % 2
        ax = fig.add_subplot(gs[row, col])
        model = model_names[i]

        plot_single_ax(ax, model, data[model])

        if col == 0:
            ax.set_ylabel(r"Mean Output Tokens")
        if row == 1:
            ax.set_xlabel(r"Cache Budget")

        axes.append(ax)

    handles, labels = axes[-1].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.04),
        frameon=True,
        columnspacing=1.5
    )

    fig.subplots_adjust(
        left=0.12,
        right=0.98,
        top=0.94,
        bottom=0.16,
        hspace=0.38,
        wspace=0.22,
    )

    out_file = "budget_vs_output_reproduced.pdf"
    plt.savefig(out_file, bbox_inches='tight')
    print(f"Saved figure to {out_file}")


if __name__ == "__main__":
    main()
