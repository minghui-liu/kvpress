"""
Qualitative analysis of RKV vs RKV-LSH token eviction decisions.

Analyzes:
a) Where RKV-LSH drops meaningful tokens that RKV keeps
b) Where RKV-LSH drops repetitive tokens that RKV fails to drop
c) Density of repetitive tokens {wait, so, but} (case-insensitive)
"""

import json
import sys
from collections import defaultdict, Counter
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
    stripped = text.strip().lower()
    return stripped in REPETITIVE_KEYWORDS


def analyze_single_method(samples, tokenizer, label):
    """Analyze repetitive token density for a single method."""
    print(f"\n{'='*80}")
    print(f"  REPETITIVE TOKEN DENSITY ANALYSIS: {label}")
    print(f"{'='*80}")

    total_evicted = 0
    total_retained = 0
    rep_evicted = 0
    rep_retained = 0
    rep_evicted_tokens = Counter()
    rep_retained_tokens = Counter()

    for sample in samples:
        for step in sample["eviction_steps"]:
            for tok in step["all_tokens"]:
                text = decode_token(tokenizer, tok["token_id"])
                is_rep = is_repetitive(text)

                if tok.get("in_window", False):
                    continue

                if tok["retained"]:
                    total_retained += 1
                    if is_rep:
                        rep_retained += 1
                        rep_retained_tokens[text.strip()] += 1
                else:
                    total_evicted += 1
                    if is_rep:
                        rep_evicted += 1
                        rep_evicted_tokens[text.strip()] += 1

    print(f"\n  Total tokens analyzed (excluding window): {total_evicted + total_retained}")
    print(f"  Total evicted: {total_evicted}, Total retained: {total_retained}")

    if total_evicted > 0:
        print(f"\n  Repetitive tokens EVICTED: {rep_evicted} / {total_evicted} = {rep_evicted/total_evicted*100:.2f}%")
        for tok, cnt in rep_evicted_tokens.most_common():
            print(f"    '{tok}': {cnt}")
    if total_retained > 0:
        print(f"\n  Repetitive tokens RETAINED: {rep_retained} / {total_retained} = {rep_retained/total_retained*100:.2f}%")
        for tok, cnt in rep_retained_tokens.most_common():
            print(f"    '{tok}': {cnt}")

    if rep_evicted + rep_retained > 0:
        evict_rate = rep_evicted / (rep_evicted + rep_retained) * 100
        print(f"\n  Repetitive token eviction rate: {rep_evicted}/{rep_evicted+rep_retained} = {evict_rate:.1f}%")


def compare_methods(rkv_samples, lsh_samples, tokenizer, model_label):
    """Compare RKV vs RKV-LSH eviction decisions on matching samples."""
    print(f"\n{'='*80}")
    print(f"  COMPARISON: RKV vs RKV-LSH ({model_label})")
    print(f"{'='*80}")

    rkv_by_id = {s["sample_id"]: s for s in rkv_samples}
    lsh_by_id = {s["sample_id"]: s for s in lsh_samples}

    common_ids = sorted(set(rkv_by_id.keys()) & set(lsh_by_id.keys()))
    print(f"\n  Matching samples: {len(common_ids)} (RKV has {len(rkv_samples)}, RKV-LSH has {len(lsh_samples)})")

    if not common_ids:
        print("  No matching samples found!")
        return

    # Aggregate stats
    total_lsh_drops_meaningful = 0
    total_lsh_drops_repetitive_rkv_keeps = 0
    total_rkv_drops_repetitive_lsh_keeps = 0  # should be rare for well-designed RKV-LSH
    examples_lsh_drops_meaningful = []
    examples_lsh_drops_repetitive = []

    for sid in common_ids:
        rkv_data = rkv_by_id[sid]
        lsh_data = lsh_by_id[sid]

        rkv_steps = rkv_data["eviction_steps"]
        lsh_steps = lsh_data["eviction_steps"]

        # Match steps by eviction_step number
        rkv_steps_by_num = {s["eviction_step"]: s for s in rkv_steps}
        lsh_steps_by_num = {s["eviction_step"]: s for s in lsh_steps}

        common_steps = sorted(set(rkv_steps_by_num.keys()) & set(lsh_steps_by_num.keys()))

        for step_num in common_steps:
            rkv_step = rkv_steps_by_num[step_num]
            lsh_step = lsh_steps_by_num[step_num]

            # Build position -> info maps
            rkv_tokens = {t["position"]: t for t in rkv_step["all_tokens"] if not t.get("in_window", False)}
            lsh_tokens = {t["position"]: t for t in lsh_step["all_tokens"] if not t.get("in_window", False)}

            common_positions = set(rkv_tokens.keys()) & set(lsh_tokens.keys())

            for pos in common_positions:
                rkv_tok = rkv_tokens[pos]
                lsh_tok = lsh_tokens[pos]

                text = decode_token(tokenizer, rkv_tok["token_id"])
                is_rep = is_repetitive(text)

                # Case A: RKV-LSH evicts, RKV retains
                if not lsh_tok["retained"] and rkv_tok["retained"]:
                    if not is_rep:
                        total_lsh_drops_meaningful += 1
                        if len(examples_lsh_drops_meaningful) < 30:
                            examples_lsh_drops_meaningful.append({
                                "sample_id": sid,
                                "step": step_num,
                                "position": pos,
                                "token_id": rkv_tok["token_id"],
                                "text": text,
                                "rkv_score": rkv_tok.get("final_score"),
                                "lsh_score": lsh_tok.get("final_score"),
                            })
                    else:
                        # RKV-LSH evicts a repetitive token that RKV keeps - this is GOOD for LSH
                        total_lsh_drops_repetitive_rkv_keeps += 1
                        if len(examples_lsh_drops_repetitive) < 30:
                            examples_lsh_drops_repetitive.append({
                                "sample_id": sid,
                                "step": step_num,
                                "position": pos,
                                "token_id": rkv_tok["token_id"],
                                "text": text,
                                "rkv_score": rkv_tok.get("final_score"),
                                "lsh_score": lsh_tok.get("final_score"),
                            })

                # Case B: RKV evicts, RKV-LSH retains (RKV drops, LSH doesn't)
                if not rkv_tok["retained"] and lsh_tok["retained"]:
                    if is_rep:
                        total_rkv_drops_repetitive_lsh_keeps += 1

    # Print results
    print(f"\n  --- Case (a): RKV-LSH drops MEANINGFUL tokens that RKV keeps ---")
    print(f"  Count: {total_lsh_drops_meaningful}")
    if examples_lsh_drops_meaningful:
        print(f"\n  Top examples:")
        for ex in examples_lsh_drops_meaningful[:20]:
            print(f"    Sample {ex['sample_id']}, Step {ex['step']}, Pos {ex['position']}: "
                  f"'{ex['text'].strip()}' (token_id={ex['token_id']}) "
                  f"RKV_score={ex['rkv_score']:.6f}, LSH_score={ex['lsh_score']:.6f}")

    print(f"\n  --- Case (b): RKV-LSH drops REPETITIVE tokens that RKV keeps ---")
    print(f"  Count: {total_lsh_drops_repetitive_rkv_keeps}")
    if examples_lsh_drops_repetitive:
        print(f"\n  Top examples:")
        for ex in examples_lsh_drops_repetitive[:20]:
            print(f"    Sample {ex['sample_id']}, Step {ex['step']}, Pos {ex['position']}: "
                  f"'{ex['text'].strip()}' (token_id={ex['token_id']}) "
                  f"RKV_score={ex['rkv_score']:.6f}, LSH_score={ex['lsh_score']:.6f}")

    print(f"\n  --- RKV drops repetitive tokens that RKV-LSH keeps ---")
    print(f"  Count: {total_rkv_drops_repetitive_lsh_keeps}")


def show_eviction_context(samples, tokenizer, label, max_samples=3, max_steps=2):
    """Show full eviction context with decoded text for a few samples."""
    print(f"\n{'='*80}")
    print(f"  EVICTION CONTEXT EXAMPLES: {label}")
    print(f"{'='*80}")

    for sample in samples[:max_samples]:
        sid = sample["sample_id"]
        print(f"\n  --- Sample {sid} ({sample['num_eviction_steps']} eviction steps) ---")

        for step in sample["eviction_steps"][:max_steps]:
            step_num = step["eviction_step"]
            kv_len = step["kv_len"]
            method = step["method"]

            evicted = []
            retained_rep = []
            evicted_rep = []

            for tok in step["all_tokens"]:
                if tok.get("in_window", False):
                    continue
                text = decode_token(tokenizer, tok["token_id"])
                is_rep = is_repetitive(text)

                if not tok["retained"]:
                    evicted.append((tok["position"], text, tok.get("final_score", 0)))
                    if is_rep:
                        evicted_rep.append((tok["position"], text))
                else:
                    if is_rep:
                        retained_rep.append((tok["position"], text))

            print(f"\n    Step {step_num} | method={method} | kv_len={kv_len} | evicted={len(evicted)} tokens")

            # Show some evicted tokens sorted by score (lowest first)
            evicted_sorted = sorted(evicted, key=lambda x: x[2])
            print(f"    Evicted tokens (lowest score first, showing top 15):")
            for pos, text, score in evicted_sorted[:15]:
                rep_marker = " [REPETITIVE]" if is_repetitive(text) else ""
                print(f"      pos={pos:4d}: '{text.strip():<20s}' score={score:.6f}{rep_marker}")

            if evicted_rep:
                print(f"    Evicted repetitive tokens: {len(evicted_rep)}")
                for pos, text in evicted_rep[:10]:
                    print(f"      pos={pos}: '{text.strip()}'")

            if retained_rep:
                print(f"    Retained repetitive tokens: {len(retained_rep)}")
                for pos, text in retained_rep[:10]:
                    print(f"      pos={pos}: '{text.strip()}'")


def main():
    print("Loading tokenizers...")
    tok_7b = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    tok_14b = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")

    # Load data
    print("Loading data files...")
    data_7b_rkv = load_jsonl(FILES["7b_rkv"])
    print(f"  7B RKV: {len(data_7b_rkv)} samples loaded")

    data_14b_rkv = load_jsonl(FILES["14b_rkv"])
    print(f"  14B RKV: {len(data_14b_rkv)} samples loaded")

    data_14b_lsh = load_jsonl(FILES["14b_rkvlsh"])
    print(f"  14B RKV-LSH: {len(data_14b_lsh)} samples loaded")

    # ===== 1. Repetitive token density for each method =====
    analyze_single_method(data_7b_rkv, tok_7b, "Qwen-7B RKV")
    analyze_single_method(data_14b_rkv, tok_14b, "Qwen-14B RKV")
    analyze_single_method(data_14b_lsh, tok_14b, "Qwen-14B RKV-LSH")

    # ===== 2. Comparison: RKV vs RKV-LSH (14B) =====
    compare_methods(data_14b_rkv, data_14b_lsh, tok_14b, "Qwen-14B")

    # ===== 3. Eviction context examples =====
    show_eviction_context(data_7b_rkv, tok_7b, "Qwen-7B RKV", max_samples=2, max_steps=2)
    show_eviction_context(data_14b_rkv, tok_14b, "Qwen-14B RKV", max_samples=2, max_steps=2)
    show_eviction_context(data_14b_lsh, tok_14b, "Qwen-14B RKV-LSH", max_samples=2, max_steps=2)

    print(f"\n{'='*80}")
    print("  ANALYSIS COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
