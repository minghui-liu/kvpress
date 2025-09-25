#!/usr/bin/env python3
import os
import sys
import csv
import signal
import argparse
from collections import defaultdict


STOP = False


def _handle_sigint(signum, frame):
    global STOP
    STOP = True
    print("\n[INFO] Caught Ctrl-C (SIGINT). Finishing current iteration and exiting...", flush=True)


def read_attn_csv(csv_path: str, filename_pattern: str | None = None, max_files: int | None = None, max_rows_per_file: int | None = None, verbose: bool = True):
    steps_to_losses = defaultdict(list)
    step_layer_to_losses = defaultdict(list)  # key: (step, layer_idx)

    def _consume_file(path: str):
        global STOP
        if STOP:
            return
        try:
            with open(path, "r", newline="") as f:
                reader = csv.DictReader(f)
                # Expected columns: prune_step, layer_idx, kv_len_pre, attn_len, diff_indices, attn_loss
                for i, row in enumerate(reader):
                    if STOP:
                        break
                    if max_rows_per_file is not None and i >= max_rows_per_file:
                        break
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
                    steps_to_losses[step].append(loss_val)
                    step_layer_to_losses[(step, layer)].append(loss_val)
        except KeyboardInterrupt:
            STOP = True
        except Exception as e:
            if verbose:
                print(f"[WARN] Failed reading {path}: {e}")

    # Accept a file or a directory of CSV files
    if os.path.isdir(csv_path):
        import fnmatch
        names = sorted(os.listdir(csv_path))
        if filename_pattern:
            names = [n for n in names if fnmatch.fnmatch(n, filename_pattern)]
        names = [n for n in names if n.lower().endswith(".csv")]
        if max_files is not None:
            names = names[:max_files]
        total = len(names)
        for idx, name in enumerate(names):
            if STOP:
                break
            p = os.path.join(csv_path, name)
            if verbose:
                print(f"[INFO] Reading ({idx+1}/{total}): {p}", flush=True)
            _consume_file(p)
    else:
        _consume_file(csv_path)

    return steps_to_losses, step_layer_to_losses


def aggregate_steps(steps_to_losses, agg: str = "sum"):
    steps = sorted(steps_to_losses.keys())
    agg_vals = []
    for s in steps:
        vals = steps_to_losses[s]
        if agg == "mean":
            agg_vals.append(sum(vals) / max(len(vals), 1))
        elif agg == "max":
            agg_vals.append(max(vals) if vals else 0.0)
        else:
            agg_vals.append(sum(vals))
    return steps, agg_vals


def cumulative(vals):
    out = []
    running = 0.0
    for v in vals:
        running += float(v)
        out.append(running)
    return out


def plot(steps, cum_vals, out_path: str, title: str, dpi: int):
    # Use a non-interactive backend for headless environments
    import matplotlib
    if os.environ.get("DISPLAY", "") == "":
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 4.5))
    plt.plot(steps, cum_vals, linewidth=2)
    plt.xlabel("prune_step")
    plt.ylabel("cumulative attention loss")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    print(f"[INFO] Saved plot to {out_path}")


def plot_heatmap(step_layer_to_losses: dict, out_path: str, title: str, agg: str, dpi: int, max_steps: int | None = None, max_layers: int | None = None):
    import numpy as np
    import matplotlib
    if os.environ.get("DISPLAY", "") == "":
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Collect axes
    steps = sorted({s for (s, _l) in step_layer_to_losses.keys()})
    layers = sorted({l for (_s, l) in step_layer_to_losses.keys()})
    if max_steps is not None:
        steps = steps[:max_steps]
    if max_layers is not None:
        layers = layers[:max_layers]
    if not steps or not layers:
        raise RuntimeError("No data for heatmap")

    step_index = {s: i for i, s in enumerate(steps)}
    layer_index = {l: i for i, l in enumerate(layers)}

    H = np.zeros((len(layers), len(steps)), dtype=float)
    for (s, l), vals in step_layer_to_losses.items():
        if agg == "mean":
            v = float(sum(vals) / max(len(vals), 1))
        elif agg == "max":
            v = float(max(vals)) if vals else 0.0
        else:
            v = float(sum(vals))
        H[layer_index[l], step_index[s]] += v

    # Cap figure size to avoid freezing with huge matrices
    fig_w = min(max(6, len(steps) * 0.3), 24)
    fig_h = min(max(4, len(layers) * 0.3), 18)
    plt.figure(figsize=(fig_w, fig_h))
    im = plt.imshow(H, aspect="auto", origin="lower", cmap="viridis")
    plt.colorbar(im, label="attention loss")
    plt.xlabel("prune_step")
    plt.ylabel("layer_idx")
    plt.yticks(range(len(layers)), layers)
    # For large steps, reduce tick clutter
    if len(steps) <= 20:
        plt.xticks(range(len(steps)), steps, rotation=45, ha="right")
    plt.title(title)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    print(f"[INFO] Saved heatmap to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot cumulative attention loss vs prune_step from CSV")
    parser.add_argument("--csv", dest="csv_path", type=str, default="attn_loss.csv", help="Path to attn loss CSV file or directory (default: attn_loss.csv)")
    parser.add_argument("--out", dest="out_path", type=str, default=None, help="Output image/CSV path (default: <csv_dir>/attn_loss_cumulative.png)")
    parser.add_argument("--agg", dest="agg", choices=["sum", "mean", "max"], default="sum", help="Aggregate across layers per step (default: sum)")
    parser.add_argument("--dpi", dest="dpi", type=int, default=150, help="Figure DPI (default: 150)")
    parser.add_argument("--title", dest="title", type=str, default="Cumulative Attention Loss", help="Plot title")
    parser.add_argument("--heatmap_out", dest="heatmap_out", type=str, default=None, help="Output heatmap image path (default: <csv_dir>/attn_loss_heatmap.png)")
    parser.add_argument("--heatmap_agg", dest="heatmap_agg", choices=["sum", "mean", "max"], default="sum", help="Aggregate across rows per (step, layer) in heatmap (default: sum)")
    parser.add_argument("--csv_pattern", dest="csv_pattern", type=str, default="attn_loss*.csv", help="When --csv is a directory, only read files matching this glob (default: attn_loss*.csv)")
    parser.add_argument("--max_csv", dest="max_csv", type=int, default=None, help="Limit number of CSV files read from directory")
    parser.add_argument("--max_steps", dest="max_steps", type=int, default=None, help="Limit number of steps in heatmap")
    parser.add_argument("--max_layers", dest="max_layers", type=int, default=None, help="Limit number of layers in heatmap")
    parser.add_argument("--max_rows", dest="max_rows", type=int, default=None, help="Limit number of rows read per CSV (for debugging)")
    parser.add_argument("--quiet", dest="quiet", action="store_true", help="Reduce logging while reading")
    args = parser.parse_args()

    csv_path = args.csv_path
    # Resolve CSV path if a directory is passed
    # If a directory is provided, we'll read all CSVs inside for heatmap/line plot.
    if os.path.isdir(csv_path):
        if not any(name.lower().endswith('.csv') for name in os.listdir(csv_path)):
            raise FileNotFoundError(f"No CSV files found in directory: {csv_path}")
        resolved_csv = csv_path
    else:
        resolved_csv = csv_path
        if not os.path.exists(resolved_csv):
            raise FileNotFoundError(f"CSV not found: {resolved_csv}")

    # Install SIGINT handler for graceful stop
    signal.signal(signal.SIGINT, _handle_sigint)

    steps_to_losses, step_layer_to_losses = read_attn_csv(
        resolved_csv,
        filename_pattern=(args.csv_pattern if os.path.isdir(resolved_csv) else None),
        max_files=args.max_csv,
        max_rows_per_file=args.max_rows,
        verbose=not args.quiet,
    )
    if not steps_to_losses:
        raise RuntimeError("No data rows found in CSV (or incorrect headers)")

    steps, agg_vals = aggregate_steps(steps_to_losses, agg=args.agg)
    cum_vals = cumulative(agg_vals)

    base_dir = os.path.dirname(os.path.abspath(resolved_csv)) if not os.path.isdir(resolved_csv) else os.path.abspath(resolved_csv)
    if args.out_path is None:
        out_path = os.path.join(base_dir, "attn_loss_cumulative.png")
    else:
        out_path = args.out_path
    heatmap_out = args.heatmap_out or os.path.join(base_dir, "attn_loss_heatmap.png")

    try:
        # If user asked for CSV output, write the cumulative series instead of plotting
        if out_path.lower().endswith(".csv"):
            out_dir = os.path.dirname(out_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(out_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["prune_step", "cumulative_attn_loss"])
                for s, v in zip(steps, cum_vals):
                    w.writerow([s, v])
            print(f"[INFO] Wrote cumulative data to {out_path}")
        else:
            plot(steps, cum_vals, out_path, title=args.title, dpi=args.dpi)
        # Always produce heatmap as well
        plot_heatmap(step_layer_to_losses, heatmap_out, title=f"Per-layer Attention Loss", agg=args.heatmap_agg, dpi=args.dpi, max_steps=args.max_steps, max_layers=args.max_layers)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
        sys.exit(130)
    except Exception as e:
        # Fallback: just print the series if plotting fails
        print(f"[WARN] Plotting failed: {e}")
        print("prune_step,cumulative_attn_loss")
        for s, v in zip(steps, cum_vals):
            print(f"{s},{v}")


if __name__ == "__main__":
    main()


