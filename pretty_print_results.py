#!/usr/bin/env python3

import argparse
import json
import re
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=
        "Pretty-print KVPress results. Works with reason/*.jsonl outputs (response, extracted_answer, gt_answer)."
    )
    parser.add_argument("file", type=str, help="Path to results file (JSONL or JSON)")
    parser.add_argument("--start", type=int, default=0, help="Start index (inclusive)")
    parser.add_argument("--end", type=int, default=None, help="End index (exclusive)")
    parser.add_argument(
        "--limit", type=int, default=None, help="Print at most this many entries (applied after start/end)")
    parser.add_argument(
        "--show-input",
        action="store_true",
        help="Also print input_text if available",
    )
    parser.add_argument(
        "--show-stats",
        action="store_true",
        help="Also print token counts and compression stats if available",
    )
    return parser.parse_args()


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # Skip malformed lines rather than crashing
                continue


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # If the JSON contains a list of entries, iterate; otherwise, wrap single dict
    if isinstance(data, list):
        for obj in data:
            yield obj
    else:
        yield data


def pretty_print(obj: dict, idx: int, show_input: bool, show_stats: bool) -> None:
    def _oneline(txt: str) -> str:
        # Remove all newlines and collapse whitespace for compact readability
        return " ".join(str(txt).split())

    # Clean response: remove runs of non-alnum symbols (even if spaced), then collapse whitespace
    raw_resp = str(obj.get("response", ""))
    # Remove sequences of 2+ non-alnum, non-space symbols, allowing spaces between them
    raw_resp = re.sub(r'(?:[^A-Za-z0-9\s]\s*){2,}', ' ', raw_resp)
    # Also remove immediate runs with no spaces as a fallback
    raw_resp = re.sub(r'[^A-Za-z0-9\s]{2,}', ' ', raw_resp)
    response = _oneline(raw_resp)
    extracted = _oneline(obj.get("extracted_answer", ""))
    gt = _oneline(obj.get("gt_answer", obj.get("gt", "")))

    print("=" * 80)
    print(f"Sample #{idx}")
    if show_input and "input_text" in obj:
        print("- Input:")
        print(_oneline(obj["input_text"]))
        print()

    print("- Response:")
    print(response)
    print()
    print(f"- Extracted Answer: {extracted}")
    print(f"- Ground Truth    : {gt}")

    if show_stats:
        itc = obj.get("input_token_count")
        otc = obj.get("output_token_count")
        ttc = obj.get("total_token_count")
        budget = obj.get("cache_budget")
        cr = obj.get("compression_ratio")
        stats = []
        if itc is not None:
            stats.append(f"input_tokens={itc}")
        if otc is not None:
            stats.append(f"output_tokens={otc}")
        if ttc is not None:
            stats.append(f"total_tokens={ttc}")
        if budget is not None:
            stats.append(f"cache_budget={budget}")
        if cr is not None:
            stats.append(f"compression_ratio={cr}")
        if stats:
            print("- Stats: " + ", ".join(stats))


def main() -> int:
    args = parse_args()
    path = Path(args.file)
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        return 1

    if path.suffix.lower() == ".jsonl":
        entries = list(iter_jsonl(path))
    elif path.suffix.lower() == ".json":
        entries = list(load_json(path))
    else:
        print("Unsupported file extension. Use .jsonl or .json", file=sys.stderr)
        return 1

    start = max(0, int(args.start))
    end = int(args.end) if args.end is not None else len(entries)
    end = min(end, len(entries))
    subset = entries[start:end]
    if args.limit is not None:
        subset = subset[: max(0, int(args.limit))]

    for i, obj in enumerate(subset, start=start):
        pretty_print(obj, i, args.show_input, args.show_stats)

    print("=" * 80)
    print(f"Printed {len(subset)} entries (from {start} to {start + len(subset) - 1})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


