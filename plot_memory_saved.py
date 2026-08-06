#!/usr/bin/env python3
"""Plot memory saved vs cache budget using metrics_output_5.csv and RKV-LSH.xlsx."""

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

# Row indices inside Efficiency sheet for the three methods (rkv-lsh, rkv, full)
SMALL_TOKEN_ROWS = {
    "math500": {
        "deepseek-ai--DeepSeek-R1-Distill-Llama-8B": 3,
        "deepseek-ai--DeepSeek-R1-Distill-Qwen-14B": 4,
    },
    "aime24": {
        "deepseek-ai--DeepSeek-R1-Distill-Llama-8B": 7,
        "deepseek-ai--DeepSeek-R1-Distill-Qwen-14B": 8,
    },
}
LARGE_TOKEN_ROWS = {
    "math500": {
        "deepseek-ai--DeepSeek-R1-Distill-Llama-8B": 28,
        "deepseek-ai--DeepSeek-R1-Distill-Qwen-14B": 29,
    },
    "aime24": {
        "deepseek-ai--DeepSeek-R1-Distill-Llama-8B": 32,
        "deepseek-ai--DeepSeek-R1-Distill-Qwen-14B": 33,
    },
}

# Column groups for methods inside the Efficiency sheet
METHOD_COLS = {
    "rkvlsh": [1, 2, 3, 4],
    "rkv": [5, 6, 7, 8],
    "full": [9, 10, 11, 12],
}


def load_excel_section(eff_df: pd.DataFrame, dataset: str, model: str, large: bool) -> dict[str, dict[int, float]]:
    rows = LARGE_TOKEN_ROWS if large else SMALL_TOKEN_ROWS
    row_idx = rows[dataset][model]
    out = {}
    for method, cols in METHOD_COLS.items():
        values = eff_df.loc[row_idx, cols].astype(float).tolist()
        out[method] = dict(zip(BUDGETS, values))
    return out


def load_metrics_rkvlsh(metrics_df: pd.DataFrame, dataset: str, model: str, max_tokens: int) -> dict[int, float]:
    subset = metrics_df[
        (metrics_df["Dataset"] == dataset)
        & (metrics_df["Model"] == model)
        & (metrics_df["Max_Tokens"] == max_tokens)
    ]
    memory_column = "Peak_Cache_Memory_MB" if "Peak_Cache_Memory_MB" in subset else "Memory_MB"
    return dict(zip(subset["Budget"].astype(int), subset[memory_column].astype(float)))


def compute_memory_saved(metrics_df: pd.DataFrame, eff_df: pd.DataFrame, dataset: str, model: str, max_tokens: int, large: bool) -> dict[str, dict[int, float]]:
    excel_vals = load_excel_section(eff_df, dataset, model, large)
    rkvlsh_metrics = load_metrics_rkvlsh(metrics_df, dataset, model, max_tokens)

    memory = {
        "full": excel_vals["full"],
        "rkv": excel_vals["rkv"],
        "rkvlsh": rkvlsh_metrics,
    }

    saved = {}
    for method in ["rkv", "rkvlsh"]:
        saved[method] = {}
        for budget in BUDGETS:
            full_val = memory["full"].get(budget)
            method_val = memory[method].get(budget)
            if full_val is None or method_val is None:
                continue
            saved[method][budget] = full_val - method_val
    return saved


def plot_case(metrics_df: pd.DataFrame, eff_df: pd.DataFrame, max_tokens: int, output_path: Path) -> None:
    # Two subplots in 1x2 grid: math500-llama, math500-qwen
    fig, axes = plt.subplots(1, 2, figsize=(20, 4))
    combos = [
        ("math500", "deepseek-ai--DeepSeek-R1-Distill-Llama-8B"),
        ("math500", "deepseek-ai--DeepSeek-R1-Distill-Qwen-14B"),
    ]

    for ax, (dataset, model) in zip(axes, combos):
        saved = compute_memory_saved(
            metrics_df,
            eff_df,
            dataset,
            model,
            max_tokens=max_tokens,
            large=(max_tokens != 2048),
        )
        label_map = {"rkv": "RKV", "rkvlsh": "RKV-LSH"}
        for method, color in [("rkv", "tab:orange"), ("rkvlsh", "tab:blue")]:
            budgets = sorted(saved[method].keys())
            values = [saved[method][b] for b in budgets]
            label = label_map.get(method, method)
            ax.plot(budgets, values, marker="o", label=label, color=color,
                   linestyle="--" if method == "rkv" else "-",
                   linewidth=3, markersize=13)
        ax.set_title(f"{DISPLAY_DATASET[dataset]} • {MODEL_LABELS[model]}", fontsize=28, fontweight="bold", pad=10)
        ax.set_xlabel("Cache budget", fontsize=28, fontweight="bold")
        ax.tick_params(axis="both", which="major", labelsize=26)
        ax.grid(True, linestyle=":", alpha=0.5, linewidth=1.3)
    axes[0].set_ylabel("Peak KV Cache Memory Saved\n(MB)", fontsize=28, fontweight="bold")
    axes[1].legend(loc="best", fontsize=26, framealpha=0.95, edgecolor="black", fancybox=True)
    fig.tight_layout()
    plt.subplots_adjust(
        left=0.10, bottom=0.12, right=0.99, top=0.85, wspace=0.25
    )
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path)
    print(f"Saved plot -> {pdf_path}")


def plot_case_2048(metrics_df: pd.DataFrame, eff_df: pd.DataFrame, output_path: Path) -> None:
    # Four subplots in 2x2 grid, preserving the 10x4 subplot footprint used by plot_case.
    fig, axes = plt.subplots(2, 2, figsize=(22, 8.5))
    combos = [
        ("math500", "deepseek-ai--DeepSeek-R1-Distill-Llama-8B", 2048),
        ("math500", "deepseek-ai--DeepSeek-R1-Distill-Qwen-14B", 2048),
        ("aime24", "deepseek-ai--DeepSeek-R1-Distill-Llama-8B", 32768),
        ("aime24", "deepseek-ai--DeepSeek-R1-Distill-Qwen-14B", 32768),
    ]

    label_map = {"rkv": "RKV", "rkvlsh": "RKV-LSH"}
    for ax, (dataset, model, max_tokens) in zip(axes.flat, combos):
        saved = compute_memory_saved(
            metrics_df,
            eff_df,
            dataset,
            model,
            max_tokens=max_tokens,
            large=False,
        )
        for method, color in [("rkv", "tab:orange"), ("rkvlsh", "tab:blue")]:
            budgets = sorted(saved[method].keys())
            values = [saved[method][b] for b in budgets]
            ax.plot(
                budgets,
                values,
                marker="o",
                label=label_map[method],
                color=color,
                linestyle="--" if method == "rkv" else "-",
                linewidth=3,
                markersize=13,
            )
        ax.set_title(
            f"{DISPLAY_DATASET[dataset]} • {MODEL_LABELS[model]}",
            fontsize=28,
            fontweight="bold",
            pad=10,
        )
        ax.set_xlabel("Cache budget", fontsize=28, fontweight="bold")
        ax.tick_params(axis="both", which="major", labelsize=26)
        ax.grid(True, linestyle=":", alpha=0.5, linewidth=1.3)

    axes[0, 0].set_ylabel("Peak KV Cache Memory Saved\n(MB)", fontsize=28, fontweight="bold")
    axes[1, 0].legend(loc="best", fontsize=26, framealpha=0.95, edgecolor="black", fancybox=True)

    fig.tight_layout()
    plt.subplots_adjust(
        left=0.08, bottom=0.11, right=0.995, top=0.90, hspace=0.70, wspace=0.28
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
        max_tokens=16384,  # for math500; aime24 handled as 32768 in helper
        output_path=Path("memory_saved_large.png"),
    )
    plot_case_2048(
        metrics_df,
        eff_df,
        output_path=Path("memory_saved_2048.png"),
    )


if __name__ == "__main__":
    main()
