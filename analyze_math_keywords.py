"""
Analyze density of math-critical keywords in eviction decisions.
Focus on tokens that directly affect math problem solving:
- Numbers (digits, numeric values)
- Variable names (n, x, y, k, etc.)
- Math operators (+, -, *, /, =, etc.)
- Math function words (sum, product, total, etc.)
- Problem-specific keywords (answer, value, equal, find, etc.)
- Named entities from questions (proper nouns, names)
"""

import json
from collections import Counter, defaultdict
from transformers import AutoTokenizer

RANKING_DIR = "ranking_analysis"

FILES = {
    "7b_rkv": f"{RANKING_DIR}/token_decisions_rkv_budget1024.jsonl",
    "14b_rkv": f"{RANKING_DIR}/token_decisions_rkv_deepseek-ai--DeepSeek-R1-Distill-Qwen-14B_budget1024.jsonl",
    "14b_rkvlsh": f"{RANKING_DIR}/token_decisions_rkvlsh_deepseek-ai--DeepSeek-R1-Distill-Qwen-14B_budget1024_buckets8.jsonl",
}

# Math-critical token categories
MATH_VARIABLES = {"n", "x", "y", "z", "k", "m", "a", "b", "c", "d", "f", "p", "q", "r", "t", "i", "j"}
MATH_OPERATORS = {"+", "-", "*", "/", "=", "<", ">", "^", "!", "%", "±"}
MATH_FUNCTIONS = {"sin", "cos", "tan", "log", "ln", "sqrt", "sum", "prod", "lim",
                  "max", "min", "mod", "gcd", "lcm", "abs", "exp", "int"}
MATH_SYMBOLS = {"frac", "cdot", "times", "div", "pi", "theta", "alpha", "beta",
                "gamma", "delta", "sigma", "lambda", "omega", "infty", "neq",
                "leq", "geq", "approx", "equiv", "subset", "cup", "cap"}
MATH_KEYWORDS = {"answer", "value", "equal", "equals", "find", "calculate", "compute",
                 "determine", "solve", "prove", "show", "sum", "product", "total",
                 "remainder", "quotient", "ratio", "percent", "percentage",
                 "area", "volume", "perimeter", "radius", "diameter", "angle",
                 "triangle", "circle", "square", "rectangle", "polygon",
                 "equation", "expression", "function", "formula", "theorem",
                 "probability", "combination", "permutation", "factorial",
                 "maximum", "minimum", "average", "mean", "median", "mode",
                 "integer", "prime", "even", "odd", "positive", "negative",
                 "diagonal", "column", "row", "matrix", "sequence", "series"}


def load_jsonl(path):
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def classify_math_critical(text):
    """Classify if a token is math-critical and return its category."""
    stripped = text.strip()
    lower = stripped.lower()

    # Numbers (including decimals, negatives)
    clean = stripped.replace(",", "").replace(" ", "")
    if clean and (clean.replace(".", "").replace("-", "").isdigit() or
                  clean.startswith("$") and clean[1:].replace(".", "").isdigit()):
        return "number"

    # LaTeX dollar signs (often wrap math expressions)
    if stripped == "$" or stripped == "$$":
        return "math_delimiter"

    # Math operators
    if stripped in MATH_OPERATORS:
        return "operator"

    # Variable names (single letters used as math variables)
    if lower in MATH_VARIABLES and len(stripped) <= 2:
        return "variable"

    # Math functions
    if lower in MATH_FUNCTIONS:
        return "math_function"

    # Math symbols (LaTeX-style)
    if lower in MATH_SYMBOLS:
        return "math_symbol"

    # Math keywords (problem-relevant words)
    if lower in MATH_KEYWORDS:
        return "math_keyword"

    # Names / proper nouns (capitalized, not at sentence start typically)
    # These appear in word problems (e.g., "Steve", "Rick")
    if stripped and stripped[0].isupper() and len(stripped) >= 2 and stripped.isalpha():
        # Check if it's a common word that happens to be capitalized
        common_caps = {"the", "and", "for", "that", "this", "with", "from", "what",
                       "how", "when", "where", "which", "each", "all", "but", "not",
                       "are", "was", "were", "been", "being", "have", "has", "had",
                       "will", "would", "could", "should", "may", "might", "can",
                       "let", "sol", "below", "step", "think", "wait", "okay",
                       "now", "then", "first", "next", "since", "because", "therefore"}
        if lower not in common_caps:
            return "name_entity"

    return None


def analyze_math_keywords(samples, tokenizer, label):
    """Analyze math-critical token eviction density."""
    print(f"\n{'='*80}")
    print(f"  MATH-CRITICAL TOKEN ANALYSIS: {label}")
    print(f"{'='*80}")

    # Per-category stats
    cat_evicted = Counter()
    cat_retained = Counter()
    cat_evicted_examples = defaultdict(list)
    cat_retained_examples = defaultdict(list)

    total_evicted = 0
    total_retained = 0
    math_critical_evicted = 0
    math_critical_retained = 0

    for sample in samples:
        for step in sample["eviction_steps"]:
            for tok in step["all_tokens"]:
                if tok.get("in_window", False):
                    continue

                text = tokenizer.decode([tok["token_id"]], skip_special_tokens=False)
                cat = classify_math_critical(text)

                if tok["retained"]:
                    total_retained += 1
                    if cat:
                        math_critical_retained += 1
                        cat_retained[cat] += 1
                        if len(cat_retained_examples[cat]) < 5:
                            cat_retained_examples[cat].append((tok["position"], text.strip(), tok.get("final_score")))
                else:
                    total_evicted += 1
                    if cat:
                        math_critical_evicted += 1
                        cat_evicted[cat] += 1
                        if len(cat_evicted_examples[cat]) < 5:
                            cat_evicted_examples[cat].append((tok["position"], text.strip(), tok.get("final_score")))

    print(f"\n  Total tokens: evicted={total_evicted}, retained={total_retained}")
    print(f"  Math-critical: evicted={math_critical_evicted}, retained={math_critical_retained}")

    mc_total = math_critical_evicted + math_critical_retained
    if mc_total > 0:
        evict_rate = math_critical_evicted / mc_total * 100
        print(f"  Math-critical eviction rate: {math_critical_evicted}/{mc_total} = {evict_rate:.2f}%")

    # Density: math-critical evicted / total retained
    if total_retained > 0:
        density = math_critical_evicted / total_retained * 100
        print(f"  Math-critical evicted / total retained: {math_critical_evicted}/{total_retained} = {density:.4f}%")

    print(f"\n  {'Category':<20s} | {'Evicted':>8s} | {'Retained':>8s} | {'Total':>8s} | {'Evict Rate':>10s}")
    print(f"  {'-'*20}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}")

    all_cats = sorted(set(list(cat_evicted.keys()) + list(cat_retained.keys())))
    for cat in all_cats:
        ev = cat_evicted.get(cat, 0)
        ret = cat_retained.get(cat, 0)
        tot = ev + ret
        rate = ev / tot * 100 if tot > 0 else 0
        print(f"  {cat:<20s} | {ev:8d} | {ret:8d} | {tot:8d} | {rate:9.1f}%")

    # Show examples of evicted math-critical tokens
    print(f"\n  Examples of EVICTED math-critical tokens:")
    for cat in all_cats:
        if cat_evicted_examples[cat]:
            print(f"    [{cat}]:")
            for pos, text, score in cat_evicted_examples[cat]:
                score_str = f"{score:.6f}" if score is not None else "N/A"
                print(f"      pos={pos:4d}: '{text}' (score={score_str})")


def compare_math_keywords(rkv_samples, lsh_samples, tokenizer, label):
    """Compare math-critical token handling between RKV and RKV-LSH."""
    print(f"\n{'='*80}")
    print(f"  MATH-CRITICAL COMPARISON: RKV vs RKV-LSH ({label})")
    print(f"{'='*80}")

    rkv_by_id = {s["sample_id"]: s for s in rkv_samples}
    lsh_by_id = {s["sample_id"]: s for s in lsh_samples}
    common_ids = sorted(set(rkv_by_id.keys()) & set(lsh_by_id.keys()))

    if not common_ids:
        print("  No matching samples!")
        return

    # Track math-critical tokens differently handled
    lsh_evicts_math = []  # LSH evicts math-critical, RKV keeps
    rkv_evicts_math = []  # RKV evicts math-critical, LSH keeps

    cat_lsh_evicts = Counter()
    cat_rkv_evicts = Counter()

    for sid in common_ids:
        rkv_steps = {s["eviction_step"]: s for s in rkv_by_id[sid]["eviction_steps"]}
        lsh_steps = {s["eviction_step"]: s for s in lsh_by_id[sid]["eviction_steps"]}

        for step_num in sorted(set(rkv_steps.keys()) & set(lsh_steps.keys())):
            rkv_tokens = {t["position"]: t for t in rkv_steps[step_num]["all_tokens"]
                          if not t.get("in_window", False)}
            lsh_tokens = {t["position"]: t for t in lsh_steps[step_num]["all_tokens"]
                          if not t.get("in_window", False)}

            for pos in set(rkv_tokens.keys()) & set(lsh_tokens.keys()):
                rkv_tok = rkv_tokens[pos]
                lsh_tok = lsh_tokens[pos]
                text = tokenizer.decode([rkv_tok["token_id"]], skip_special_tokens=False)
                cat = classify_math_critical(text)

                if not cat:
                    continue

                # LSH evicts math-critical, RKV keeps
                if not lsh_tok["retained"] and rkv_tok["retained"]:
                    cat_lsh_evicts[cat] += 1
                    if len(lsh_evicts_math) < 20:
                        lsh_evicts_math.append({
                            "sid": sid, "step": step_num, "pos": pos,
                            "text": text.strip(), "cat": cat,
                            "rkv_score": rkv_tok.get("final_score"),
                            "lsh_score": lsh_tok.get("final_score"),
                        })

                # RKV evicts math-critical, LSH keeps
                if not rkv_tok["retained"] and lsh_tok["retained"]:
                    cat_rkv_evicts[cat] += 1
                    if len(rkv_evicts_math) < 20:
                        rkv_evicts_math.append({
                            "sid": sid, "step": step_num, "pos": pos,
                            "text": text.strip(), "cat": cat,
                            "rkv_score": rkv_tok.get("final_score"),
                            "lsh_score": lsh_tok.get("final_score"),
                        })

    total_lsh = sum(cat_lsh_evicts.values())
    total_rkv = sum(cat_rkv_evicts.values())

    print(f"\n  Math-critical tokens LSH evicts but RKV keeps: {total_lsh}")
    for cat, cnt in cat_lsh_evicts.most_common():
        print(f"    {cat:<20s}: {cnt}")

    print(f"\n  Math-critical tokens RKV evicts but LSH keeps: {total_rkv}")
    for cat, cnt in cat_rkv_evicts.most_common():
        print(f"    {cat:<20s}: {cnt}")

    if lsh_evicts_math:
        print(f"\n  Examples: LSH evicts math-critical, RKV keeps:")
        for ex in lsh_evicts_math:
            print(f"    Pos {ex['pos']:4d} | '{ex['text']}' [{ex['cat']}] | "
                  f"RKV={ex['rkv_score']:.6f}, LSH={ex['lsh_score']:.6f}")

    if rkv_evicts_math:
        print(f"\n  Examples: RKV evicts math-critical, LSH keeps:")
        for ex in rkv_evicts_math:
            print(f"    Pos {ex['pos']:4d} | '{ex['text']}' [{ex['cat']}] | "
                  f"RKV={ex['rkv_score']:.6f}, LSH={ex['lsh_score']:.6f}")


def main():
    print("Loading tokenizers...")
    tok_7b = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    tok_14b = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")

    print("Loading data...")
    data_7b_rkv = load_jsonl(FILES["7b_rkv"])
    data_14b_rkv = load_jsonl(FILES["14b_rkv"])
    data_14b_lsh = load_jsonl(FILES["14b_rkvlsh"])

    # Per-method analysis
    analyze_math_keywords(data_7b_rkv, tok_7b, "Qwen-7B RKV")
    analyze_math_keywords(data_14b_rkv, tok_14b, "Qwen-14B RKV")
    analyze_math_keywords(data_14b_lsh, tok_14b, "Qwen-14B RKV-LSH")

    # Direct comparison
    compare_math_keywords(data_14b_rkv, data_14b_lsh, tok_14b, "Qwen-14B")

    print(f"\n{'='*80}")
    print("  DONE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
