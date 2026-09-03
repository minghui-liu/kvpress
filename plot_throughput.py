#!/usr/bin/env python3
"""Plot throughput vs cache budget using metrics_output_5.csv and RKV-LSH.xlsx."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

METRICS_PATH = Path("metrics_output_5.csv")
EXCEL_PATH = Path("RKV-LSH.xlsx")

BUDGETS = [128, 256, 512, 1024]
MODELS = ["deepseek-ai--DeepSeek-R1-Distill-Llama-8B", "deepseek-ai--DeepSeek-R1-Distill-Qwen-14B"]
MODEL_LABELS = {
    "deepseek-ai--DeepSeek-R1-Distill-Llama-8B": "DeepSeek-R1-Dstill-Llama-8B",
    "deepseek-ai--DeepSeek-R1-Distill-Qwen-14B": "DeepSeek-R1-Dstill-Qwen-14B",
}
DISPLAY_DATASET = {"math500": "Math500", "aime24": "AIME24"}

# Row indices for throughput in Efficiency sheet
# Small tokens (2048)
THROUGHPUT_SMALL_ROWS = {
    "math500": {
        "deepseek-ai--DeepSeek-R1-Distill-Llama-8B": 12,
        "deepseek-ai--DeepSeek-R1-Distill-Qwen-14B": 11,
    },
    "aime24": {
        "deepseek-ai--DeepSeek-R1-Distill-Llama-8B": 16,
        "deepseek-ai--DeepSeek-R1-Distill-Qwen-14B": 15,
    },
}
# Large tokens section (16384 for math500, 32768 for aime24)
THROUGHPUT_LARGE_ROWS = {
    "math500": {
        "deepseek-ai--DeepSeek-R1-Distill-Llama-8B": 36,
        "deepseek-ai--DeepSeek-R1-Distill-Qwen-14B": 37,
    },
    "aime24": {
        "deepseek-ai--DeepSeek-R1-Distill-Llama-8B": 40,
        "deepseek-ai--DeepSeek-R1-Distill-Qwen-14B": 41,
    },
}

# Column groups: rkv-lsh, rkv, full
METHOD_COLS = {
    "rkvlsh": [1, 2, 3, 4],
    "rkv": [5, 6, 7, 8],
    "full": [9, 10, 11, 12],
}


def load_excel_throughput(eff_df: pd.DataFrame, dataset: str, model: str, large: bool) -> dict[str, dict[int, float]]:
    rows = THROUGHPUT_LARGE_ROWS if large else THROUGHPUT_SMALL_ROWS
    row_idx = rows[dataset][model]
    out = {}
    for method, cols in METHOD_COLS.items():
        values = eff_df.loc[row_idx, cols].astype(float).tolist()
        out[method] = dict(zip(BUDGETS, values))
    return out


def load_metrics_rkvlsh_throughput(metrics_df: pd.DataFrame, dataset: str, model: str, max_tokens: int) -> dict[int, float]:
    subset = metrics_df[
        (metrics_df["Dataset"] == dataset)
        & (metrics_df["Model"] == model)
        & (metrics_df["Max_Tokens"] == max_tokens)
    ]
    return dict(zip(subset["Budget"].astype(int), subset["Throughput"].astype(float)))


def combine_throughput(metrics_df: pd.DataFrame, eff_df: pd.DataFrame, dataset: str, model: str, max_tokens: int, large: bool) -> dict[str, dict[int, float]]:
    excel_vals = load_excel_throughput(eff_df, dataset, model, large)
    rkvlsh_vals = load_metrics_rkvlsh_throughput(metrics_df, dataset, model, max_tokens)
    combined = {
        "full": excel_vals["full"],
        "rkv": excel_vals["rkv"],
        "rkvlsh": rkvlsh_vals,
    }
    return combined


def plot_case(metrics_df: pd.DataFrame, eff_df: pd.DataFrame, max_tokens: int, output_path: Path) -> None:
    # Four subplots in 2x2 grid: Math500 on top, AIME24 on bottom.
    fig, axes = plt.subplots(2, 2, figsize=(22, 8.5))
    combos = [
        ("math500", "deepseek-ai--DeepSeek-R1-Distill-Llama-8B", max_tokens),
        ("math500", "deepseek-ai--DeepSeek-R1-Distill-Qwen-14B", max_tokens),
        (
            "aime24",
            "deepseek-ai--DeepSeek-R1-Distill-Llama-8B",
            32768 if max_tokens != 2048 else 2048,
        ),
        (
            "aime24",
            "deepseek-ai--DeepSeek-R1-Distill-Qwen-14B",
            32768 if max_tokens != 2048 else 2048,
        ),
    ]

    for ax, (dataset, model, current_max_tokens) in zip(axes.flat, combos):
        combined = combine_throughput(
            metrics_df,
            eff_df,
            dataset,
            model,
            max_tokens=current_max_tokens,
            large=(current_max_tokens != 2048),
        )
        label_map = {"rkv": "RKV", "rkvlsh": "RKV-LSH", "full": "Full"}
        style_map = {"rkv": ("tab:orange", "--"), "rkvlsh": ("tab:blue", "-"), "full": ("tab:green", ":")}
        for method in ["full", "rkv", "rkvlsh"]:
            budgets = sorted(combined[method].keys())
            values = [combined[method][b] for b in budgets]
            color, linestyle = style_map[method]
            ax.plot(budgets, values, marker="o", label=label_map[method], color=color, linestyle=linestyle,
                   linewidth=3, markersize=13)
        ax.set_title(f"{DISPLAY_DATASET[dataset]} • {MODEL_LABELS[model]}", fontsize=28, fontweight="bold", pad=10)
        ax.set_xlabel("Cache budget", fontsize=28, fontweight="bold")
        ax.tick_params(axis="both", which="major", labelsize=26)
        ax.grid(True, linestyle=":", alpha=0.5, linewidth=1.3)
    axes[0, 0].set_ylabel("Throughput\n(tokens/s)", fontsize=28, fontweight="bold")
    axes[1, 1].legend(loc="best", fontsize=26, framealpha=0.95, edgecolor="black", fancybox=True)
    fig.tight_layout()
    plt.subplots_adjust(
        left=0.12, bottom=0.11, right=0.995, top=0.90, hspace=0.70, wspace=0.28
    )
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path)
    print(f"Saved plot -> {pdf_path}")


def main() -> None:
    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"Missing {METRICS_PATH}")
    if not EXCEL_PATH.exists():
        raise FileNotFoundError(f"Missing {EXCEL_PATH}")

    metrics_df = pd.read_csv(METRICS_PATH)
    eff_df = pd.read_excel(EXCEL_PATH, sheet_name="Efficiency", header=None)

    plot_case(
        metrics_df,
        eff_df,
        max_tokens=2048,
        output_path=Path("throughput_2048.png"),
    )
    plot_case(
        metrics_df,
        eff_df,
        max_tokens=16384,  # math500; aime24 handled as 32768
        output_path=Path("throughput_large.png"),
    )


if __name__ == "__main__":
    main()
