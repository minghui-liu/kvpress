import matplotlib.pyplot as plt


MAX_NEW_TOKENS = [2048, 5096, 10192, 16384]

ACCURACY = {
    "H2O": [0.41, 0.51, 0.51, 0.56],
    "KNorm": [0.31, 0.30, 0.30, 0.30],
    "SnapKV": [0.31, 0.59, 0.62, 0.63],
    "StreamingLLM": [0.48, 0.54, 0.54, 0.54],
    "SCOPE": [0.35, 0.58, 0.63, 0.66],
    "RPC": [0.40, 0.60, 0.66, 0.65],
}

COLORS = {
    "H2O": "#1f77b4",
    "KNorm": "#ff7f0e",
    "SnapKV": "#2ca02c",
    "StreamingLLM": "#d62728",
    "SCOPE": "#9467bd",
    "RPC": "#8c564b",
}


def main():
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 12,
            "lines.linewidth": 1.8,
        }
    )

    fig, ax = plt.subplots(figsize=(5.0, 3.6))

    for method, values in ACCURACY.items():
        ax.plot(
            MAX_NEW_TOKENS,
            values,
            marker="o",
            markersize=5,
            color=COLORS[method],
            label=method,
        )

    ax.set_xlabel("Max New Tokens")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(MAX_NEW_TOKENS)
    ax.set_ylim(0.28, 0.70)
    ax.grid(True, alpha=0.55)
    ax.legend(loc="lower right", bbox_to_anchor=(0.98, 0.16), frameon=True,ncol=2)

    fig.tight_layout()
    out_file = "scripts/max_token_ablation.pdf"
    fig.savefig(out_file, bbox_inches="tight")
    print(f"Saved figure to {out_file}")


if __name__ == "__main__":
    main()
