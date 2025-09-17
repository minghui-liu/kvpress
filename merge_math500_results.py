#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path
from glob import glob


MODELS = [
    "deepseek-ai--DeepSeek-R1-Distill-Llama-8B",
    "deepseek-ai--DeepSeek-R1-Distill-Qwen-14B",
]


def collect_scores(files):
    metrics = []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                obj = json.load(f)
                p = Path(fp)
                obj.setdefault("_source_file", p.name)
                obj.setdefault("_score_path", str(p))
                metrics.append(obj)
        except FileNotFoundError:
            continue
        except json.JSONDecodeError:
            continue
    return metrics


def summarize(metrics_list):
    summary = {}
    if not metrics_list:
        return {"total_shards": 0}

    total_shards = len(metrics_list)
    total_samples = sum(int(m.get("num_samples", 0) or 0) for m in metrics_list)

    # Accuracy (weighted by num_samples if available, else simple mean)
    acc_vals = [(float(m.get("accuracy", 0.0) or 0.0), int(m.get("num_samples", 0) or 0)) for m in metrics_list]
    w_num = sum(a * n for a, n in acc_vals)
    if total_samples > 0:
        acc_overall = w_num / total_samples
    else:
        # fallback: mean of accuracies
        acc_overall = sum(a for a, _ in acc_vals) / total_shards

    # Average compression (simple mean)
    comp_vals = [float(m.get("avg_compression", 0.0) or 0.0) for m in metrics_list]
    comp_mean = sum(comp_vals) / total_shards

    # Timing aggregates if present
    dec_tokens = sum(int(m.get("decoding_tokens_total", 0) or 0) for m in metrics_list)
    dec_time = sum(float(m.get("decoding_time_total_s", 0.0) or 0.0) for m in metrics_list)
    prefill_time = sum(float(m.get("prefill_time_total_s", 0.0) or 0.0) for m in metrics_list)
    total_runtime = sum(float(m.get("total_runtime_s", 0.0) or 0.0) for m in metrics_list)

    summary.update(
        {
            "total_shards": total_shards,
            "total_samples": total_samples,
            "accuracy_overall": acc_overall,
            "avg_compression_mean": comp_mean,
            "decoding_tokens_total": dec_tokens,
            "decoding_time_total_s": dec_time,
            "prefill_time_total_s": prefill_time,
            "total_runtime_s_sum": total_runtime,
        }
    )
    if dec_time > 0:
        summary["decoding_toks_per_s_overall"] = dec_tokens / dec_time
    return summary


def main():
    parser = argparse.ArgumentParser(description="Merge math500 score JSON shards per model into a single JSON with per-shard metrics and a summary.")
    parser.add_argument("--out_dir", type=str, default="reason/results", help="Directory to write merged JSONs")
    parser.add_argument("--base_dir", type=str, default=".", help="Repository root (where reason/ and results_old/ live)")
    args = parser.parse_args()

    base = Path(args.base_dir)
    out_dir = base / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # We specifically aggregate past runs from results_old
    search_dirs = [base / "results_old"]

    for model_slug in MODELS:
        # Find exactly the 25 math500 split score.json shards for this model across results_old
        expected_splits = [f"splittest[{s}:{s+20}]" for s in range(0, 500, 20)]
        files = []
        for split_tag in expected_splits:
            found_for_split = []
            for sd in search_dirs:
                pattern = str(sd / f"math500*__{model_slug}__*{split_tag}*score.json")
                found_for_split.extend(glob(pattern))
            # If duplicates, keep the lexicographically last
            if found_for_split:
                files.append(sorted(found_for_split)[-1])

        files = sorted(set(files))
        metrics_list = collect_scores(files)

        # Write a combined score.json-style file with averaged metrics only
        summary = summarize(metrics_list)
        # For some fields, pick the most frequent value across shards
        from collections import Counter
        def most_common_or_none(key):
            vals = [m.get(key) for m in metrics_list if key in m and m.get(key) is not None]
            return Counter(vals).most_common(1)[0][0] if vals else None

        # Derive total samples by reading companion JSONL when num_samples is missing
        def effective_num_samples(m):
            n = int(m.get("num_samples", 0) or 0)
            if n > 0:
                return n
            score_path = m.get("_score_path")
            if not score_path:
                return 0
            # Companion JSONL lives next to the score.json (same directory)
            try:
                sp = Path(score_path)
                jsonl_path = sp.with_name(sp.name.replace("_score.json", ".jsonl"))
                if jsonl_path.exists():
                    with open(jsonl_path, "r", encoding="utf-8") as f:
                        return sum(1 for line in f if line.strip())
            except Exception:
                return 0
            return 0

        weights = [effective_num_samples(m) for m in metrics_list]
        total_samples_eff = sum(weights)

        # Accuracy weighted by effective samples (fallback to mean)
        acc_vals = [float(m.get("accuracy", 0.0) or 0.0) for m in metrics_list]
        if total_samples_eff > 0:
            acc_weighted = sum(a * w for a, w in zip(acc_vals, weights)) / total_samples_eff
        else:
            acc_weighted = sum(acc_vals) / len(acc_vals) if acc_vals else 0.0

        # Compression weighted by effective samples (fallback to mean)
        comp_vals = [float(m.get("avg_compression", 0.0) or 0.0) for m in metrics_list]
        if total_samples_eff > 0:
            comp_weighted = sum(c * w for c, w in zip(comp_vals, weights)) / total_samples_eff
        else:
            comp_weighted = sum(comp_vals) / len(comp_vals) if comp_vals else 0.0

        combined = {
            "accuracy": acc_weighted,
            "avg_compression": comp_weighted,
            "num_samples": 500 if metrics_list else 0,
            "dataset": "math500",
            "data_split": "test",
            "data_dir": most_common_or_none("data_dir"),
            "model_name": model_slug.replace("--", "/"),
            "press_name": most_common_or_none("press_name") or "various",
            "cache_budget": most_common_or_none("cache_budget"),
            "fraction": most_common_or_none("fraction"),
            "max_new_tokens": most_common_or_none("max_new_tokens"),
            "max_context_length": most_common_or_none("max_context_length"),
            "random_seed": most_common_or_none("random_seed"),
        }
        # Timing fields if present
        if "total_runtime_s_sum" in summary:
            combined["total_runtime_s"] = summary["total_runtime_s_sum"]
        if "decoding_time_total_s" in summary and summary["decoding_time_total_s"] > 0:
            combined["decoding_tokens_total"] = summary.get("decoding_tokens_total", 0)
            combined["decoding_time_total_s"] = summary.get("decoding_time_total_s", 0.0)
            combined["prefill_time_total_s"] = summary.get("prefill_time_total_s", 0.0)
            combined["decoding_toks_per_s"] = summary.get("decoding_toks_per_s_overall", 0.0)

        combined_path = out_dir / f"math500__{model_slug}__combined_score.json"
        with open(combined_path, "w", encoding="utf-8") as f:
            json.dump(combined, f)

        print(f"Wrote combined metrics to {combined_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())


