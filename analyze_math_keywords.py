#!/usr/bin/env python3
"""Calculate math-critical token retention from an evaluation result JSONL."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

MATH_VARIABLES = {
    "n", "x", "y", "z", "k", "m", "a", "b", "c", "d", "f", "p", "q", "r", "t", "i", "j"
}
MATH_OPERATORS = {"+", "-", "*", "/", "=", "<", ">", "^", "!", "%", "±"}
MATH_FUNCTIONS = {
    "sin", "cos", "tan", "log", "ln", "sqrt", "sum", "prod", "lim",
    "max", "min", "mod", "gcd", "lcm", "abs", "exp", "int",
}
MATH_SYMBOLS = {
    "frac", "cdot", "times", "div", "pi", "theta", "alpha", "beta",
    "gamma", "delta", "sigma", "lambda", "omega", "infty", "neq",
    "leq", "geq", "approx", "equiv", "subset", "cup", "cap",
}
MATH_KEYWORDS = {
    "answer", "value", "equal", "equals", "find", "calculate", "compute",
    "determine", "solve", "prove", "show", "sum", "product", "total",
    "remainder", "quotient", "ratio", "percent", "percentage",
    "area", "volume", "perimeter", "radius", "diameter", "angle",
    "triangle", "circle", "square", "rectangle", "polygon",
    "equation", "expression", "function", "formula", "theorem",
    "probability", "combination", "permutation", "factorial",
    "maximum", "minimum", "average", "mean", "median", "mode",
    "integer", "prime", "even", "odd", "positive", "negative",
    "diagonal", "column", "row", "matrix", "sequence", "series",
}
COMMON_CAPITALIZED_WORDS = {
    "the", "and", "for", "that", "this", "with", "from", "what",
    "how", "when", "where", "which", "each", "all", "but", "not",
    "are", "was", "were", "been", "being", "have", "has", "had",
    "will", "would", "could", "should", "may", "might", "can",
    "let", "sol", "below", "step", "think", "wait", "okay",
    "now", "then", "first", "next", "since", "because", "therefore",
}


def classify_math_critical(text: str) -> str | None:
    """Classify a decoded token into one math-critical category."""
    stripped = text.strip()
    lower = stripped.lower().lstrip("\\")
    clean = stripped.replace(",", "").replace(" ", "")

    if clean and (
        clean.replace(".", "").replace("-", "").isdigit()
        or (clean.startswith("$") and clean[1:].replace(".", "").isdigit())
    ):
        return "number"
    if stripped in {"$", "$$"}:
        return "math_delimiter"
    if stripped in MATH_OPERATORS:
        return "operator"
    if lower in MATH_VARIABLES and len(stripped) <= 2:
        return "variable"
    if lower in MATH_FUNCTIONS:
        return "math_function"
    if lower in MATH_SYMBOLS:
        return "math_symbol"
    if lower in MATH_KEYWORDS:
        return "math_keyword"
    if (
        stripped
        and stripped[0].isupper()
        and len(stripped) >= 2
        and stripped.isalpha()
        and lower not in COMMON_CAPITALIZED_WORDS
    ):
        return "name_entity"
    return None


def load_results(path: Path) -> list[dict[str, Any]]:
    results = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
    return results


def analyze_result_file(
    result_file: Path,
    model_name: str,
    include_name_entities: bool = False,
    tokenizer=None,
) -> dict[str, Any]:
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    results = load_results(result_file)

    category_total: Counter[str] = Counter()
    category_retained: Counter[str] = Counter()
    retained_examples: dict[str, list[str]] = defaultdict(list)
    evicted_examples: dict[str, list[str]] = defaultdict(list)
    per_sample = []
    skipped_reasons: Counter[str] = Counter()

    for sample_index, sample in enumerate(results):
        status = sample.get("retention_tracking_status")
        if status not in {"tracked", "no_compression"}:
            skipped_reasons[status or "missing_tracking_status"] += 1
            continue

        token_ids = sample.get("sequence_token_ids", sample.get("input_token_ids"))
        retained_positions = sample.get("final_retained_indices")
        tracked_length = sample.get("tracked_sequence_length")
        if token_ids is None:
            skipped_reasons["missing_token_ids"] += 1
            continue
        if retained_positions is None:
            skipped_reasons["missing_final_retained_indices"] += 1
            continue
        if tracked_length is None:
            # Compatibility for prompt-only records produced during development.
            tracked_length = len(token_ids)

        tracked_length = min(int(tracked_length), len(token_ids))
        retained_set = {
            int(position)
            for position in retained_positions
            if 0 <= int(position) < tracked_length
        }
        sample_total = 0
        sample_retained = 0
        sample_categories: Counter[str] = Counter()
        sample_retained_categories: Counter[str] = Counter()

        special_ids = set(getattr(tokenizer, "all_special_ids", []))
        for position, token_id in enumerate(token_ids[:tracked_length]):
            if token_id in special_ids:
                continue
            text = tokenizer.decode([token_id], skip_special_tokens=False)
            category = classify_math_critical(text)
            if category is None or (category == "name_entity" and not include_name_entities):
                continue

            category_total[category] += 1
            sample_categories[category] += 1
            sample_total += 1
            if position in retained_set:
                category_retained[category] += 1
                sample_retained_categories[category] += 1
                sample_retained += 1
                if len(retained_examples[category]) < 10:
                    retained_examples[category].append(text)
            elif len(evicted_examples[category]) < 10:
                evicted_examples[category].append(text)

        per_sample.append(
            {
                "sample_index": sample_index,
                "critical_total": sample_total,
                "critical_retained": sample_retained,
                "retention_rate": sample_retained / sample_total if sample_total else None,
                "category_total": dict(sample_categories),
                "category_retained": dict(sample_retained_categories),
            }
        )

    categories = {}
    for category in sorted(category_total):
        total = category_total[category]
        retained = category_retained[category]
        categories[category] = {
            "total": total,
            "retained": retained,
            "evicted": total - retained,
            "retention_rate": retained / total if total else None,
            "retained_examples": retained_examples[category],
            "evicted_examples": evicted_examples[category],
        }

    total = sum(category_total.values())
    retained = sum(category_retained.values())
    if not per_sample:
        reasons = ", ".join(
            f"{reason}={count}" for reason, count in sorted(skipped_reasons.items())
        )
        raise ValueError(
            "No analyzable samples found. Run evaluate.py with --track_tokens=true "
            f"using a supported press. Skipped: {reasons or 'no records'}"
        )

    return {
        "result_file": str(result_file),
        "model_name": model_name,
        "retention_tracking_scope": "layer_0_kv_head_0",
        "include_name_entities": include_name_entities,
        "num_result_samples": len(results),
        "num_analyzed_samples": len(per_sample),
        "num_skipped_samples": len(results) - len(per_sample),
        "skipped_reasons": dict(skipped_reasons),
        "critical_total": total,
        "critical_retained": retained,
        "critical_evicted": total - retained,
        "critical_token_retention_rate": retained / total if total else None,
        # Explicit aliases make the JSON self-describing.
        "math_critical_total": total,
        "math_critical_retained": retained,
        "math_critical_evicted": total - retained,
        "math_critical_retention_rate": retained / total if total else None,
        "categories": categories,
        "per_sample": per_sample,
    }


def print_report(report: dict[str, Any]) -> None:
    print("=" * 80)
    print("MATH-CRITICAL TOKEN RETENTION")
    print("=" * 80)
    print(f"Result file: {report['result_file']}")
    print(f"Tracking scope: {report['retention_tracking_scope']}")
    print(f"Analyzed samples: {report['num_analyzed_samples']}")
    print(f"Skipped samples: {report['num_skipped_samples']}")
    print()

    rate = report["critical_token_retention_rate"]
    rate_text = f"{rate:.2%}" if rate is not None else "N/A"
    print(
        "Overall: "
        f"{report['critical_retained']}/{report['critical_total']} retained "
        f"({rate_text})"
    )
    print()
    print(f"{'Category':<20} {'Retained':>10} {'Evicted':>10} {'Total':>10} {'Rate':>10}")
    print("-" * 64)
    for category, values in report["categories"].items():
        category_rate = values["retention_rate"]
        category_rate_text = f"{category_rate:.2%}" if category_rate is not None else "N/A"
        print(
            f"{category:<20} {values['retained']:>10} {values['evicted']:>10} "
            f"{values['total']:>10} {category_rate_text:>10}"
        )

    if report["skipped_reasons"]:
        print()
        print(f"Skipped reasons: {report['skipped_reasons']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate final KV-cache retention of math-critical tokens."
    )
    parser.add_argument("--result_file", type=Path, required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument(
        "--include_name_entities",
        action="store_true",
        help="Include capitalized names as critical tokens.",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        help="Optional JSON file for the complete aggregate and per-sample report.",
    )
    args = parser.parse_args()

    report = analyze_result_file(
        args.result_file,
        args.model_name,
        include_name_entities=args.include_name_entities,
    )
    print_report(report)
    if args.output_file:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        with args.output_file.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"\nSaved JSON report to {args.output_file}")


if __name__ == "__main__":
    main()
