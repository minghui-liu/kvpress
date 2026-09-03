#!/usr/bin/env python3
import os
import csv
import argparse
from tqdm import tqdm


def read_single_csv(csv_path: str):
    step_layer_to_losses = {}
    # Count rows (excluding header) for progress bar
    try:
        with open(csv_path, "r", newline="") as f:
            total = sum(1 for _ in f) - 1
            if total < 0:
                total = 0
    except Exception:
        total = None

    print(f"[INFO] Opening CSV: {csv_path}", flush=True)
    if total is not None:
        print(f"[INFO] Total data rows (excluding header): {total}", flush=True)

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        # We will both print and (if available) show a tqdm bar.
        iterator = reader if total is None else tqdm(reader, total=total, desc=f"Reading {os.path.basename(csv_path)}", leave=False)
        for i, row in enumerate(iterator, start=1):
            if "attn_loss" not in row:
                continue
            try:
                step = int(row.get("prune_step", 0))
            except Exception:
                continue
            try:
                layer = int(row.get("layer_idx", -1))
            except Exception:
                layer = -1
            try:
                loss_val = float(row.get("attn_loss", 0.0))
            except Exception:
                loss_val = 0.0
            step_layer_to_losses.setdefault((step, layer), []).append(loss_val)
            # Verbose per-row print for debugging stuck reads
            print(f"[ROW {i}] step={step} layer={layer} loss={loss_val}", flush=True)
    return step_layer_to_losses


def aggregate(step_layer_to_losses: dict, agg: str):
    import numpy as np

    steps = sorted({s for (s, _l) in step_layer_to_losses.keys()})
    layers = sorted({l for (_s, l) in step_layer_to_losses.keys()})
    if not steps or not layers:
        raise RuntimeError("CSV missing required columns or no rows: need prune_step, layer_idx, attn_loss")

    step_index = {s: i for i, s in enumerate(steps)}
    layer_index = {l: i for i, l in enumerate(layers)}

    print(f"[INFO] Aggregating to matrix with {len(layers)} layers x {len(steps)} steps", flush=True)

    H = np.zeros((len(layers), len(steps)), dtype=float)
    for (s, l), vals in step_layer_to_losses.items():
        if not vals:
            continue
        if agg == "mean":
            v = float(sum(vals) / max(len(vals), 1))
        elif agg == "max":
            v = float(max(vals))
        else:
            v = float(sum(vals))
        H[layer_index[l], step_index[s]] += v

    return steps, layers, H


def plot_heatmap(steps, layers, H, out_path: str, title: str, dpi: int):
    import matplotlib
    if os.environ.get("DISPLAY", "") == "":
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_w = min(max(6, len(steps) * 0.3), 24)
    fig_h = min(max(4, len(layers) * 0.3), 18)
    print(f"[INFO] Plot figure size: {fig_w} x {fig_h}", flush=True)
    plt.figure(figsize=(fig_w, fig_h))
    im = plt.imshow(H, aspect="auto", origin="lower", cmap="viridis")
    plt.colorbar(im, label="attention loss")
    plt.xlabel("prune_step")
    plt.ylabel("layer_idx")
    plt.yticks(range(len(layers)), layers)
    if len(steps) <= 40:
        plt.xticks(range(len(steps)), steps, rotation=45, ha="right")
    plt.title(title)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.tight_layout()
    print(f"[INFO] Saving heatmap to {out_path}", flush=True)
    plt.savefig(out_path, dpi=dpi)
    print(f"[INFO] Saved heatmap to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot per-layer attention loss heatmap from a single CSV")
    parser.add_argument("--csv", required=True, help="Path to attn_loss CSV file (single file)")
    parser.add_argument("--out", default=None, help="Output heatmap image path (default: <csv_dir>/attn_loss_heatmap.png)")
    parser.add_argument("--agg", choices=["sum", "mean", "max"], default="sum", help="Aggregate across rows per (step, layer) (default: sum)")
    parser.add_argument("--dpi", type=int, default=150, help="Figure DPI (default: 150)")
    parser.add_argument("--title", default="Per-layer Attention Loss", help="Plot title")
    args = parser.parse_args()

    csv_path = args.csv
    if not os.path.exists(csv_path) or not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    step_layer_to_losses = read_single_csv(csv_path)
    steps, layers, H = aggregate(step_layer_to_losses, agg=args.agg)

    out_path = args.out
    if out_path is None:
        base_dir = os.path.dirname(os.path.abspath(csv_path))
        out_path = os.path.join(base_dir, "attn_loss_heatmap.png")

    plot_heatmap(steps, layers, H, out_path=out_path, title=args.title, dpi=args.dpi)


if __name__ == "__main__":
    main()


