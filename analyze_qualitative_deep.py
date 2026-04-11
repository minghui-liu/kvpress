"""
Deep qualitative analysis: decode full sequences, show context around evicted tokens,
categorize meaningful vs filler, and provide publication-ready examples.
"""

import json
from collections import Counter, defaultdict
from transformers import AutoTokenizer

REPETITIVE_KEYWORDS = {"wait", "so", "but"}
RANKING_DIR = "ranking_analysis"

FILES = {
    "7b_rkv": f"{RANKING_DIR}/token_decisions_rkv_budget1024.jsonl",
    "14b_rkv": f"{RANKING_DIR}/token_decisions_rkv_deepseek-ai--DeepSeek-R1-Distill-Qwen-14B_budget1024.jsonl",
    "14b_rkvlsh": f"{RANKING_DIR}/token_decisions_rkvlsh_deepseek-ai--DeepSeek-R1-Distill-Qwen-14B_budget1024_buckets8.jsonl",
}


def load_jsonl(path):
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def decode_token(tokenizer, token_id):
    return tokenizer.decode([token_id], skip_special_tokens=False)


def is_repetitive(text):
    return text.strip().lower() in REPETITIVE_KEYWORDS


def decode_context_window(tokenizer, all_tokens, center_pos, window=10):
    """Decode a window of tokens around a position for context."""
    tokens_by_pos = {t["position"]: t for t in all_tokens}
    positions = sorted(tokens_by_pos.keys())

    start = max(0, center_pos - window)
    end = min(max(positions) + 1, center_pos + window + 1)

    parts = []
    for p in range(start, end):
        if p in tokens_by_pos:
            tok = tokens_by_pos[p]
            text = decode_token(tokenizer, tok["token_id"])
            if p == center_pos:
                parts.append(f">>>{text}<<<")
            else:
                parts.append(text)
    return "".join(parts)


def categorize_token(text):
    """Categorize a token into content types."""
    stripped = text.strip().lower()

    if stripped in REPETITIVE_KEYWORDS:
        return "repetitive"

    # Punctuation/symbols
    if all(c in ".,;:!?()[]{}\"'`~@#$%^&*-_=+/<>|\\  \n\t" for c in stripped):
        return "punctuation"

    # Numbers
    if stripped.replace(".", "").replace("-", "").isdigit():
        return "number"

    # Math/formula tokens
    math_tokens = {"x", "y", "z", "n", "k", "m", "i", "j", "a", "b", "c", "d", "f",
                   "sin", "cos", "tan", "log", "ln", "sqrt", "sum", "prod", "lim",
                   "frac", "cdot", "times", "div", "mod", "eq", "ne", "le", "ge",
                   "alpha", "beta", "gamma", "delta", "theta", "pi", "sigma"}
    if stripped in math_tokens:
        return "math"

    # Short subword fragments (1-2 chars, not meaningful words)
    if len(stripped) <= 2 and stripped not in {"is", "in", "on", "at", "to", "of", "or", "an", "if", "do", "no", "we", "he", "me", "my", "up"}:
        return "subword"

    # Common thinking/filler words
    filler_words = {"the", "and", "is", "in", "to", "of", "a", "for", "that", "this",
                    "with", "it", "on", "as", "at", "by", "an", "be", "or", "from",
                    "we", "can", "have", "has", "had", "will", "would", "could", "should",
                    "let", "need", "want", "think", "know", "see", "get", "make", "take",
                    "then", "now", "here", "there", "also", "just"}
    if stripped in filler_words:
        return "common_word"

    # Content words (likely meaningful)
    if len(stripped) >= 3:
        return "content"

    return "other"


def detailed_comparison(rkv_samples, lsh_samples, tokenizer, model_label):
    """Detailed comparison with full context."""
    print(f"\n{'='*80}")
    print(f"  DETAILED COMPARISON: RKV vs RKV-LSH ({model_label})")
    print(f"{'='*80}")

    rkv_by_id = {s["sample_id"]: s for s in rkv_samples}
    lsh_by_id = {s["sample_id"]: s for s in lsh_samples}
    common_ids = sorted(set(rkv_by_id.keys()) & set(lsh_by_id.keys()))

    if not common_ids:
        print("  No matching samples!")
        return

    # Track categories of differentially evicted tokens
    lsh_evicts_rkv_keeps_categories = Counter()
    rkv_evicts_lsh_keeps_categories = Counter()

    # Track full examples
    lsh_drops_meaningful_examples = []
    lsh_drops_repetitive_examples = []
    rkv_drops_meaningful_examples = []

    for sid in common_ids:
        rkv_data = rkv_by_id[sid]
        lsh_data = lsh_by_id[sid]

        rkv_steps_by_num = {s["eviction_step"]: s for s in rkv_data["eviction_steps"]}
        lsh_steps_by_num = {s["eviction_step"]: s for s in lsh_data["eviction_steps"]}
        common_steps = sorted(set(rkv_steps_by_num.keys()) & set(lsh_steps_by_num.keys()))

        for step_num in common_steps:
            rkv_step = rkv_steps_by_num[step_num]
            lsh_step = lsh_steps_by_num[step_num]

            rkv_tokens = {t["position"]: t for t in rkv_step["all_tokens"] if not t.get("in_window", False)}
            lsh_tokens = {t["position"]: t for t in lsh_step["all_tokens"] if not t.get("in_window", False)}
            common_positions = set(rkv_tokens.keys()) & set(lsh_tokens.keys())

            for pos in common_positions:
                rkv_tok = rkv_tokens[pos]
                lsh_tok = lsh_tokens[pos]
                text = decode_token(tokenizer, rkv_tok["token_id"])
                cat = categorize_token(text)

                # RKV-LSH evicts, RKV keeps
                if not lsh_tok["retained"] and rkv_tok["retained"]:
                    lsh_evicts_rkv_keeps_categories[cat] += 1

                    context = decode_context_window(tokenizer, lsh_step["all_tokens"], pos)
                    entry = {
                        "sample_id": sid, "step": step_num, "position": pos,
                        "text": text.strip(), "category": cat,
                        "rkv_score": rkv_tok.get("final_score"),
                        "lsh_score": lsh_tok.get("final_score"),
                        "context": context,
                    }

                    if cat == "repetitive":
                        lsh_drops_repetitive_examples.append(entry)
                    elif cat in ("content", "math", "number"):
                        lsh_drops_meaningful_examples.append(entry)

                # RKV evicts, RKV-LSH keeps
                if not rkv_tok["retained"] and lsh_tok["retained"]:
                    rkv_evicts_lsh_keeps_categories[cat] += 1

                    if cat in ("content", "math", "number"):
                        context = decode_context_window(tokenizer, rkv_step["all_tokens"], pos)
                        rkv_drops_meaningful_examples.append({
                            "sample_id": sid, "step": step_num, "position": pos,
                            "text": text.strip(), "category": cat,
                            "rkv_score": rkv_tok.get("final_score"),
                            "lsh_score": lsh_tok.get("final_score"),
                            "context": context,
                        })

    # Print category breakdown
    print(f"\n  ======= Tokens RKV-LSH EVICTS but RKV KEEPS =======")
    print(f"  (These are tokens LSH considers less important)")
    total = sum(lsh_evicts_rkv_keeps_categories.values())
    print(f"  Total: {total}")
    for cat, cnt in lsh_evicts_rkv_keeps_categories.most_common():
        print(f"    {cat:20s}: {cnt:5d} ({cnt/total*100:5.1f}%)")

    print(f"\n  ======= Tokens RKV EVICTS but RKV-LSH KEEPS =======")
    print(f"  (These are tokens RKV considers less important)")
    total2 = sum(rkv_evicts_lsh_keeps_categories.values())
    print(f"  Total: {total2}")
    for cat, cnt in rkv_evicts_lsh_keeps_categories.most_common():
        print(f"    {cat:20s}: {cnt:5d} ({cnt/total2*100:5.1f}%)")

    # Print meaningful examples
    print(f"\n  ======= CASE (a): RKV-LSH drops MEANINGFUL tokens, RKV keeps =======")
    print(f"  Count: {len(lsh_drops_meaningful_examples)}")
    for ex in sorted(lsh_drops_meaningful_examples, key=lambda x: x["lsh_score"] or 0)[:15]:
        print(f"\n    Pos {ex['position']:4d} | '{ex['text']}' [{ex['category']}]")
        print(f"    RKV_score={ex['rkv_score']:.6f}, LSH_score={ex['lsh_score']:.6f}")
        print(f"    Context: ...{ex['context'][:120]}...")

    print(f"\n  ======= CASE (b): RKV-LSH drops REPETITIVE tokens, RKV keeps =======")
    print(f"  Count: {len(lsh_drops_repetitive_examples)}")
    for ex in lsh_drops_repetitive_examples:
        print(f"\n    Pos {ex['position']:4d} | '{ex['text']}' [{ex['category']}]")
        print(f"    RKV_score={ex['rkv_score']:.6f}, LSH_score={ex['lsh_score']:.6f}")
        print(f"    Context: ...{ex['context'][:120]}...")

    print(f"\n  ======= RKV drops MEANINGFUL tokens, RKV-LSH keeps =======")
    print(f"  (LSH advantage: retains meaningful tokens that RKV discards)")
    print(f"  Count: {len(rkv_drops_meaningful_examples)}")
    for ex in sorted(rkv_drops_meaningful_examples, key=lambda x: x["rkv_score"] or 0)[:15]:
        print(f"\n    Pos {ex['position']:4d} | '{ex['text']}' [{ex['category']}]")
        print(f"    RKV_score={ex['rkv_score']:.6f}, LSH_score={ex['lsh_score']:.6f}")
        print(f"    Context: ...{ex['context'][:120]}...")


def repetitive_density_over_steps(samples, tokenizer, label):
    """Track how repetitive token density changes across eviction steps."""
    print(f"\n{'='*80}")
    print(f"  REPETITIVE TOKEN DENSITY ACROSS STEPS: {label}")
    print(f"{'='*80}")

    step_data = defaultdict(lambda: {"evicted_rep": 0, "evicted_total": 0,
                                      "retained_rep": 0, "retained_total": 0})

    for sample in samples:
        for step in sample["eviction_steps"]:
            snum = step["eviction_step"]
            for tok in step["all_tokens"]:
                if tok.get("in_window", False):
                    continue
                text = decode_token(tokenizer, tok["token_id"])
                is_rep = is_repetitive(text)

                if tok["retained"]:
                    step_data[snum]["retained_total"] += 1
                    if is_rep:
                        step_data[snum]["retained_rep"] += 1
                else:
                    step_data[snum]["evicted_total"] += 1
                    if is_rep:
                        step_data[snum]["evicted_rep"] += 1

    print(f"\n  {'Step':>6s} | {'Evicted_Rep':>12s} | {'Evicted_Tot':>12s} | {'Evict_Rep%':>10s} | {'Retained_Rep':>13s} | {'Retained_Tot':>13s} | {'Retain_Rep%':>11s}")
    print(f"  {'-'*6}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}-+-{'-'*13}-+-{'-'*13}-+-{'-'*11}")

    for snum in sorted(step_data.keys()):
        d = step_data[snum]
        evict_pct = d["evicted_rep"] / d["evicted_total"] * 100 if d["evicted_total"] > 0 else 0
        retain_pct = d["retained_rep"] / d["retained_total"] * 100 if d["retained_total"] > 0 else 0
        print(f"  {snum:6d} | {d['evicted_rep']:12d} | {d['evicted_total']:12d} | {evict_pct:9.2f}% | {d['retained_rep']:13d} | {d['retained_total']:13d} | {retain_pct:10.2f}%")


def full_sequence_decode_example(samples, tokenizer, label, sample_idx=0, step_idx=0):
    """Decode the full token sequence for one example, highlighting evictions."""
    print(f"\n{'='*80}")
    print(f"  FULL SEQUENCE DECODE: {label} (sample={sample_idx}, step={step_idx})")
    print(f"{'='*80}")

    if sample_idx >= len(samples):
        print("  Sample not available")
        return

    sample = samples[sample_idx]
    if step_idx >= len(sample["eviction_steps"]):
        print("  Step not available")
        return

    step = sample["eviction_steps"][step_idx]
    print(f"  Method: {step['method']}, KV_len: {step['kv_len']}")

    # Decode all tokens, marking evicted ones
    all_tokens = sorted(step["all_tokens"], key=lambda t: t["position"])

    # Show first 200 tokens with markers
    print(f"\n  First 100 tokens (evicted shown with [X], repetitive with [R]):")
    text_parts = []
    for tok in all_tokens[:100]:
        text = decode_token(tokenizer, tok["token_id"])
        is_rep = is_repetitive(text)
        marker = ""
        if not tok["retained"] and not tok.get("in_window", False):
            marker = "[X]"
        if is_rep:
            marker += "[R]"
        if marker:
            text_parts.append(f"{marker}{text}")
        else:
            text_parts.append(text)
    print("  " + "".join(text_parts))

    # Show tokens around positions 700-800 (where repetitive tokens tend to be)
    print(f"\n  Tokens at positions 690-790 (where repetitive keywords appear):")
    text_parts = []
    for tok in all_tokens:
        if 690 <= tok["position"] <= 790:
            text = decode_token(tokenizer, tok["token_id"])
            is_rep = is_repetitive(text)
            marker = ""
            if not tok["retained"] and not tok.get("in_window", False):
                marker = "[X]"
            if is_rep:
                marker += "[R]"
            if marker:
                text_parts.append(f"{marker}{text}")
            else:
                text_parts.append(text)
    print("  " + "".join(text_parts))


def aggregate_repetitive_stats(all_data, tokenizer):
    """Print a clean summary table of repetitive token stats across all methods."""
    print(f"\n{'='*80}")
    print(f"  SUMMARY TABLE: Repetitive Token Statistics")
    print(f"{'='*80}")

    print(f"\n  {'Method':<30s} | {'Rep Evicted':>12s} | {'Rep Retained':>13s} | {'Rep Total':>10s} | {'Evict Rate':>10s} | {'Rep in Evicted':>14s} | {'Rep in Retained':>15s}")
    print(f"  {'-'*30}-+-{'-'*12}-+-{'-'*13}-+-{'-'*10}-+-{'-'*10}-+-{'-'*14}-+-{'-'*15}")

    for label, (samples, tok) in all_data.items():
        rep_evicted = 0
        rep_retained = 0
        total_evicted = 0
        total_retained = 0

        for sample in samples:
            for step in sample["eviction_steps"]:
                for t in step["all_tokens"]:
                    if t.get("in_window", False):
                        continue
                    text = decode_token(tok, t["token_id"])
                    is_rep = is_repetitive(text)
                    if t["retained"]:
                        total_retained += 1
                        if is_rep:
                            rep_retained += 1
                    else:
                        total_evicted += 1
                        if is_rep:
                            rep_evicted += 1

        rep_total = rep_evicted + rep_retained
        evict_rate = rep_evicted / rep_total * 100 if rep_total > 0 else 0
        rep_in_evicted = rep_evicted / total_evicted * 100 if total_evicted > 0 else 0
        rep_in_retained = rep_retained / total_retained * 100 if total_retained > 0 else 0

        print(f"  {label:<30s} | {rep_evicted:12d} | {rep_retained:13d} | {rep_total:10d} | {evict_rate:9.1f}% | {rep_in_evicted:13.2f}% | {rep_in_retained:14.2f}%")


def per_keyword_breakdown(samples, tokenizer, label):
    """Break down by individual repetitive keyword."""
    print(f"\n  --- Per-keyword breakdown: {label} ---")

    keyword_stats = defaultdict(lambda: {"evicted": 0, "retained": 0})

    for sample in samples:
        for step in sample["eviction_steps"]:
            for t in step["all_tokens"]:
                if t.get("in_window", False):
                    continue
                text = decode_token(tokenizer, t["token_id"]).strip().lower()
                if text in REPETITIVE_KEYWORDS:
                    if t["retained"]:
                        keyword_stats[text]["retained"] += 1
                    else:
                        keyword_stats[text]["evicted"] += 1

    for kw in sorted(keyword_stats.keys()):
        s = keyword_stats[kw]
        total = s["evicted"] + s["retained"]
        rate = s["evicted"] / total * 100 if total > 0 else 0
        print(f"    '{kw}': evicted={s['evicted']}, retained={s['retained']}, total={total}, evict_rate={rate:.1f}%")


def main():
    print("Loading tokenizers...")
    tok_7b = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    tok_14b = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")

    print("Loading data files...")
    data_7b_rkv = load_jsonl(FILES["7b_rkv"])
    data_14b_rkv = load_jsonl(FILES["14b_rkv"])
    data_14b_lsh = load_jsonl(FILES["14b_rkvlsh"])
    print(f"  7B RKV: {len(data_7b_rkv)} samples, 14B RKV: {len(data_14b_rkv)} samples, 14B RKV-LSH: {len(data_14b_lsh)} samples")

    # 1. Summary table
    all_data = {
        "Qwen-7B RKV": (data_7b_rkv, tok_7b),
        "Qwen-14B RKV": (data_14b_rkv, tok_14b),
        "Qwen-14B RKV-LSH": (data_14b_lsh, tok_14b),
    }
    aggregate_repetitive_stats(all_data, None)

    # 2. Per-keyword breakdown
    per_keyword_breakdown(data_7b_rkv, tok_7b, "Qwen-7B RKV")
    per_keyword_breakdown(data_14b_rkv, tok_14b, "Qwen-14B RKV")
    per_keyword_breakdown(data_14b_lsh, tok_14b, "Qwen-14B RKV-LSH")

    # 3. Detailed comparison RKV vs RKV-LSH (14B)
    detailed_comparison(data_14b_rkv, data_14b_lsh, tok_14b, "Qwen-14B")

    # 4. Repetitive density across steps
    repetitive_density_over_steps(data_7b_rkv, tok_7b, "Qwen-7B RKV")
    repetitive_density_over_steps(data_14b_rkv, tok_14b, "Qwen-14B RKV")
    repetitive_density_over_steps(data_14b_lsh, tok_14b, "Qwen-14B RKV-LSH")

    # 5. Full sequence decode examples
    full_sequence_decode_example(data_14b_rkv, tok_14b, "Qwen-14B RKV", sample_idx=0, step_idx=0)
    full_sequence_decode_example(data_14b_lsh, tok_14b, "Qwen-14B RKV-LSH", sample_idx=0, step_idx=0)
    full_sequence_decode_example(data_14b_lsh, tok_14b, "Qwen-14B RKV-LSH", sample_idx=0, step_idx=1)

    # 6. 7B examples
    full_sequence_decode_example(data_7b_rkv, tok_7b, "Qwen-7B RKV", sample_idx=0, step_idx=0)

    print(f"\n{'='*80}")
    print("  ANALYSIS COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
