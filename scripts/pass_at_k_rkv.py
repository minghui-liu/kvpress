#!/usr/bin/env python3
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


FILENAME_RE = re.compile(
    r"^(?P<dataset>.*?)____(?P<model>.*?)__(?P<method>rkv|rkvlsh)__"
    r"budget(?P<budget>\d+)__.*?__block(?P<block>\d+)_size(?P<block_size>\d+)__"
    r"seed(?P<seed>\d+)__.*?\.jsonl$"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute pass@1..pass@K accuracy for RKV/RKV-LSH repeated runs."
    )
    parser.add_argument("--results_dir", default="results_rkv")
    parser.add_argument("--dataset", default=None, help="Optional dataset filter, e.g. aime24.")
    parser.add_argument("--model_name", default=None, help="Optional HF model filter.")
    parser.add_argument("--budget", type=int, default=None, help="Optional budget filter.")
    parser.add_argument("--methods", nargs="+", default=["rkv", "rkvlsh"])
    parser.add_argument("--max_k", type=int, default=64)
    parser.add_argument("--output_csv", default="results_rkv/pass_at_k.csv")
    parser.add_argument("--output_json", default="results_rkv/pass_at_k_summary.json")
    parser.add_argument("--plot", default="results_rkv/pass_at_k.pdf")
    parser.add_argument(
        "--strict_exact",
        action="store_true",
        help="Use raw extracted_answer == gt_answer instead of numeric/boxed normalization.",
    )
    return parser.parse_args()


def model_from_filename(value):
    return value.replace("--", "/")


def normalize_answer(value):
    if value is None:
        return ""

    text = str(value).strip()
    boxed = re.search(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        text = boxed.group(1)

    # AIME answers are integers. Prefer the final integer-like token so that
    # "answer is 025" and "\\boxed{25}" compare as the same value.
    numbers = re.findall(r"-?\d+", text)
    if numbers:
        try:
            return str(int(numbers[-1]))
        except ValueError:
            return numbers[-1].lstrip("0") or "0"

    return re.sub(r"\s+", "", text)


def is_correct(row, strict_exact=False):
    pred = row.get("extracted_answer")
    gt = row.get("gt_answer", row.get("solution", row.get("answer")))
    if strict_exact:
        return pred == gt
    return normalize_answer(pred) == normalize_answer(gt)


def problem_key(row, fallback_index):
    for key in ("id", "unique_id", "problem_id"):
        if row.get(key) is not None:
            return str(row[key])
    for key in ("input_text", "problem", "question"):
        if row.get(key):
            return str(row[key])
    return f"line:{fallback_index}"


def iter_matching_files(args):
    root = Path(args.results_dir).expanduser()
    for path in root.glob("*.jsonl"):
        match = FILENAME_RE.match(path.name)
        if not match:
            continue

        meta = match.groupdict()
        meta["model_name"] = model_from_filename(meta["model"])
        meta["budget"] = int(meta["budget"])
        meta["block"] = int(meta["block"])
        meta["block_size"] = int(meta["block_size"])
        meta["seed"] = int(meta["seed"])

        if meta["method"] not in args.methods:
            continue
        if args.dataset and meta["dataset"] != args.dataset:
            continue
        if args.model_name and meta["model_name"] != args.model_name:
            continue
        if args.budget is not None and meta["budget"] != args.budget:
            continue

        yield path, meta


def load_runs(args):
    runs = defaultdict(lambda: defaultdict(dict))
    files_by_method = defaultdict(list)
    blocks_by_method_seed = defaultdict(lambda: defaultdict(set))
    row_counts = defaultdict(lambda: defaultdict(int))

    for path, meta in iter_matching_files(args):
        method = meta["method"]
        seed = meta["seed"]
        files_by_method[method].append(str(path))
        blocks_by_method_seed[method][seed].add(meta["block"])

        with path.open() as handle:
            for line_index, line in enumerate(handle):
                if not line.strip():
                    continue
                row = json.loads(line)
                key = problem_key(row, line_index)
                runs[method][seed][key] = runs[method][seed].get(key, False) or is_correct(
                    row, args.strict_exact
                )
                row_counts[method][seed] += 1

    return runs, files_by_method, blocks_by_method_seed, row_counts


def compute_pass_at_k(runs, max_k):
    rows = []
    summary = {}

    for method in sorted(runs):
        seeds = sorted(runs[method])
        all_problems = sorted({key for seed in seeds for key in runs[method][seed]})
        method_summary = {
            "num_seeds": len(seeds),
            "seeds": seeds,
            "num_problems": len(all_problems),
            "pass_at_k": {},
        }

        upper_k = min(max_k, len(seeds))
        for k in range(1, upper_k + 1):
            selected = seeds[:k]
            solved = 0
            for key in all_problems:
                if any(runs[method][seed].get(key, False) for seed in selected):
                    solved += 1
            accuracy = solved / len(all_problems) if all_problems else 0.0
            rows.append(
                {
                    "method": method,
                    "k": k,
                    "accuracy": accuracy,
                    "solved": solved,
                    "num_problems": len(all_problems),
                    "seeds_used": " ".join(map(str, selected)),
                }
            )
            method_summary["pass_at_k"][str(k)] = accuracy

        summary[method] = method_summary

    return rows, summary


def write_csv(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["method", "k", "accuracy", "solved", "num_problems", "seeds_used"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_plot(rows, output_path):
    if not output_path:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping plot")
        return

    by_method = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)

    labels = {
        "rkv": "R-KV",
        "rkvlsh": "R-KV Hash",
    }

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for method in sorted(by_method):
        points = sorted(by_method[method], key=lambda row: row["k"])
        ax.plot(
            [row["k"] for row in points],
            [row["accuracy"] for row in points],
            marker="o",
            linewidth=2,
            markersize=3,
            label=labels.get(method, method),
        )

    ax.set_xlabel("k")
    ax.set_ylabel("Pass@k Accuracy")
    ax.set_xlim(1, max(row["k"] for row in rows))
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)


def main():
    args = parse_args()
    runs, files_by_method, blocks_by_method_seed, row_counts = load_runs(args)
    if not runs:
        raise FileNotFoundError(f"No matching JSONL files found in {args.results_dir}")

    rows, summary = compute_pass_at_k(runs, args.max_k)
    for method, data in summary.items():
        missing_blocks = []
        all_blocks = sorted({block for blocks in blocks_by_method_seed[method].values() for block in blocks})
        if all_blocks:
            expected_blocks = set(range(min(all_blocks), max(all_blocks) + 1))
            for seed in data["seeds"]:
                missing = sorted(expected_blocks - blocks_by_method_seed[method][seed])
                if missing:
                    missing_blocks.append({"seed": seed, "missing_blocks": missing})

        data["num_files"] = len(files_by_method[method])
        data["rows_per_seed"] = {str(seed): row_counts[method][seed] for seed in data["seeds"]}
        data["missing_blocks"] = missing_blocks

    write_csv(rows, args.output_csv)
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(summary, indent=2))
    write_plot(rows, args.plot)

    for method in sorted(summary):
        pass_at_k = summary[method]["pass_at_k"]
        last_k = str(max(map(int, pass_at_k)))
        print(
            f"{method}: seeds={summary[method]['num_seeds']}, "
            f"problems={summary[method]['num_problems']}, "
            f"pass@1={pass_at_k.get('1', 0.0):.4f}, pass@{last_k}={pass_at_k[last_k]:.4f}"
        )
        if summary[method]["missing_blocks"]:
            print(f"  missing blocks: {summary[method]['missing_blocks'][:10]}")

    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_json}")
    if args.plot:
        print(f"Wrote {args.plot}")


if __name__ == "__main__":
    main()
