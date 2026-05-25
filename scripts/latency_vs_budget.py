#!/usr/bin/env python3
import matplotlib.pyplot as plt


BASE_FONT_SIZE = 12
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": BASE_FONT_SIZE,
        "axes.labelsize": BASE_FONT_SIZE + 1,
        "xtick.labelsize": BASE_FONT_SIZE,
        "ytick.labelsize": BASE_FONT_SIZE,
        "legend.fontsize": BASE_FONT_SIZE,
        "lines.linewidth": 2.0,
    }
)


budgets = [128, 256, 384, 512]

data = {
    "H2O": [0.190, 0.163, 0.168, 0.152],
    "SnapKV-D": [0.493, 0.171, 0.183, 0.196],
    "KNorm": [0.097, 0.100, 0.089, 0.053],
    "StreamingLLM": [0.107, 0.102, 0.084, 0.073],
}

styles = {
    "H2O": {"color": "#1f77b4"},
    "SnapKV-D": {"color": "#ff7f0e"},
    "KNorm": {"color": "#2ca02c"},
    "StreamingLLM": {"color": "#d62728"},
}


def main():
    fig, ax = plt.subplots(figsize=(6, 3))

    for method in ["H2O", "SnapKV-D", "KNorm", "StreamingLLM"]:
        ax.plot(
            budgets,
            data[method],
            marker="o",
            markersize=4.8,
            color=styles[method]["color"],
            label=method,
            alpha=0.82,
        )

    ax.set_xlabel("Cache Budget")
    ax.set_ylabel("Avg. Time per Token (ms)")
    ax.set_xticks(budgets)
    ax.set_xlim(112, 528)
    ax.set_ylim(0, 0.55)
    ax.grid(True, linestyle="--", alpha=0.65)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.58, 1.12),
        ncol=2,
        frameon=True,
        columnspacing=1.6,
        handlelength=2.2,
    )

    fig.tight_layout(pad=0.3)
    out_file = "latency_vs_budget.pdf"
    fig.savefig(out_file, bbox_inches="tight")
    print(f"Saved figure to {out_file}")


if __name__ == "__main__":
    main()
