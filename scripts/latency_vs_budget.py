#!/usr/bin/env python3
import matplotlib.pyplot as plt


BASE_FONT_SIZE = 12
plt.rcParams.update(
    {
        # ACL/ARR font compliance: Computer Modern (matplotlib's bundled cmr10),
        # embedded as Type 42 rather than DejaVu / Type 3.
        "font.family": "serif",
        "font.serif": ["cmr10", "CMU Serif", "Computer Modern Roman"],
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
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
    "SCOPE": [0.857, 0.853, 0.856, 0.855],
    "RPC": [1.561, 1.560, 1.558, 1.562],
}

styles = {
    "H2O": {"color": "#1f77b4"},
    "SnapKV-D": {"color": "#ff7f0e"},
    "KNorm": {"color": "#2ca02c"},
    "StreamingLLM": {"color": "#d62728"},
    "SCOPE": {"color": "#9467bd"},
    "RPC": {"color": "#8c564b"},
}


def main():
    fig, ax = plt.subplots(figsize=(6, 3))

    for method in ["H2O", "SnapKV-D", "KNorm", "StreamingLLM", "SCOPE", "RPC"]:
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
    ax.set_ylabel("Compression Overhead\nper Token (ms)", linespacing=1.3)
    ax.set_xticks(budgets)
    ax.set_xlim(112, 528)
    ax.set_ylim(0, 1.9)
    ax.grid(True, linestyle="--", alpha=0.65)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        ncol=3,
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
