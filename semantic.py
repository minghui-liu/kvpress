#!/usr/bin/env python3
"""Compute semantic similarity between compressed-method and full responses."""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


METHODS = {"rkv", "h2o", "knorm", "snapkv", "streaming_llm"}
DEFAULT_RESULTS_DIR = Path("results")


def model_to_file_name(model_name: str) -> str:
    return model_name.replace("/", "--")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare responses from a KV compression method against full responses "
            "using the selected model's input embedding layer."
        )
    )
    parser.add_argument("--model_name", required=True, help="Hugging Face model name used by the experiment.")
    parser.add_argument("--method_name", required=True, choices=sorted(METHODS), help="Compressed method to compare.")
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. gsm8k, math500, folio, reclor, drop.")
    parser.add_argument("--budget", type=int, default=None, help="Optional cache budget to filter method files.")
    parser.add_argument(
        "--results_dir",
        default=str(DEFAULT_RESULTS_DIR),
        help="Directory containing experiment JSONL files. Use ~/Downloads/results locally if needed.",
    )
    parser.add_argument("--output", default=None, help="Optional path to write the summary JSON.")
    parser.add_argument("--batch_size", type=int, default=8, help="Embedding batch size.")
    parser.add_argument("--max_length", type=int, default=2048, help="Max tokens per response for embedding.")
    parser.add_argument("--device", default=None, help="Torch device. Defaults to cuda if available, else cpu.")
    parser.add_argument("--trust_remote_code", action="store_true", help="Pass trust_remote_code=True to HF loaders.")
    return parser.parse_args()


def metadata_from_path(path: Path) -> dict[str, Any]:
    name = path.name
    metadata: dict[str, Any] = {"path": str(path)}
    patterns = {
        "budget": r"__budget(\d+)(?:__|$)",
        "seed": r"__seed(\d+)(?:__|$)",
        "block": r"__block(\d+)_size(\d+)(?:__|$)",
        "max_new_tokens": r"__max_new_tokens(\d+)(?:__|$)",
        "window": r"__window(\d+)(?:__|$)",
        "hash_bucket": r"__hash_bucket(\d+)(?:__|$)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, name)
        if not match:
            continue
        if key == "block":
            metadata["block_index"] = int(match.group(1))
            metadata["block_size"] = int(match.group(2))
        else:
            metadata[key] = int(match.group(1))

    return metadata


def find_files(results_dir: Path, dataset: str, model_name: str, method_name: str, budget: int | None) -> list[Path]:
    model_file = model_to_file_name(model_name)
    pattern = str(results_dir / f"{dataset}*__{model_file}__{method_name}__*.jsonl")
    files = [Path(path) for path in glob.glob(pattern)]

    if budget is not None and method_name != "full":
        files = [path for path in files if f"__budget{budget}__" in path.name]

    return sorted(files)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            obj["_line_index"] = line_no
            records.append(obj)
    return records


def sample_key(record: dict[str, Any]) -> str:
    for key in ("unique_id", "id", "input_text", "question", "problem"):
        value = record.get(key)
        if value is not None:
            return f"{key}:{value}"
    return f"line:{record['_line_index']}"


def file_pair_key(path: Path) -> tuple[int | None, int | None, int | None]:
    metadata = metadata_from_path(path)
    return (
        metadata.get("seed"),
        metadata.get("block_index"),
        metadata.get("block_size"),
    )


def seed_key(path: Path) -> int | None:
    return metadata_from_path(path).get("seed")


def load_full_indexes(full_files: list[Path]) -> dict[int | None, dict[str, str]]:
    by_seed: dict[int | None, dict[str, str]] = defaultdict(dict)

    for path in full_files:
        current_seed = seed_key(path)
        for record in load_jsonl(path):
            key = sample_key(record)
            response = str(record.get("response", ""))
            by_seed[current_seed].setdefault(key, response)

    return by_seed


class ResponseEmbedder:
    def __init__(
        self,
        model_name: str,
        device: str,
        batch_size: int,
        max_length: int,
        trust_remote_code: bool,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
            trust_remote_code=trust_remote_code,
        ).to(device)
        self.model.eval()
        self.embedding = self.model.get_input_embeddings()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def encode(self, texts: list[str]):
        import torch.nn.functional as F

        vectors = []
        with self.torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                ).to(self.device)
                token_embeddings = self.embedding(inputs["input_ids"])
                mask = inputs["attention_mask"].unsqueeze(-1).to(token_embeddings.dtype)
                pooled = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
                pooled = F.normalize(pooled.float(), p=2, dim=-1)
                vectors.append(pooled.cpu())
        return self.torch.cat(vectors, dim=0)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def main() -> None:
    args = parse_args()

    import torch

    results_dir = Path(args.results_dir).expanduser()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    method_files = find_files(results_dir, args.dataset, args.model_name, args.method_name, args.budget)
    full_files = find_files(results_dir, args.dataset, args.model_name, "full", None)

    if not method_files:
        raise FileNotFoundError(
            f"No {args.method_name} JSONL files found for dataset={args.dataset}, "
            f"model={args.model_name}, budget={args.budget}, results_dir={results_dir}"
        )
    if not full_files:
        raise FileNotFoundError(
            f"No full JSONL files found for dataset={args.dataset}, model={args.model_name}, results_dir={results_dir}"
        )

    full_by_seed = load_full_indexes(full_files)

    pairs: list[dict[str, Any]] = []
    missing_files = 0
    missing_records = 0
    for method_file in method_files:
        method_metadata = metadata_from_path(method_file)
        current_seed = method_metadata.get("seed")
        seed_full = full_by_seed.get(current_seed)
        if seed_full is None:
            missing_files += 1
            continue
        for record in load_jsonl(method_file):
            key = sample_key(record)
            full_response = seed_full.get(key)
            if full_response is None:
                missing_records += 1
                continue
            pairs.append(
                {
                    "key": key,
                    "method_response": str(record.get("response", "")),
                    "full_response": full_response,
                    "seed": method_metadata.get("seed"),
                    "block_index": method_metadata.get("block_index"),
                    "block_size": method_metadata.get("block_size"),
                    "method_file": str(method_file),
                }
            )

    if not pairs:
        raise RuntimeError("No matching response pairs found between method and full files.")

    embedder = ResponseEmbedder(
        args.model_name,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        trust_remote_code=args.trust_remote_code,
    )
    method_vectors = embedder.encode([pair["method_response"] for pair in pairs])
    full_vectors = embedder.encode([pair["full_response"] for pair in pairs])
    similarities = (method_vectors * full_vectors).sum(dim=1).tolist()

    by_seed: dict[int, list[float]] = defaultdict(list)
    by_block: dict[str, list[float]] = defaultdict(list)
    by_file: dict[str, list[float]] = defaultdict(list)
    for pair, sim in zip(pairs, similarities):
        if pair["seed"] is not None:
            by_seed[int(pair["seed"])].append(sim)
        if pair["block_index"] is not None:
            by_block[f"{pair['block_index']}_size{pair['block_size']}"].append(sim)
        by_file[pair["method_file"]].append(sim)

    summary = {
        "dataset": args.dataset,
        "model_name": args.model_name,
        "method_name": args.method_name,
        "budget": args.budget,
        "results_dir": str(results_dir),
        "device": device,
        "num_method_files": len(method_files),
        "num_full_files": len(full_files),
        "num_matched_method_files": len({pair["method_file"] for pair in pairs}),
        "num_pairs": len(pairs),
        "missing_method_files_without_matching_full_seed": missing_files,
        "missing_records_within_matched_files": missing_records,
        "avg_semantic_similarity": mean(similarities),
        "min_semantic_similarity": min(similarities),
        "max_semantic_similarity": max(similarities),
        "by_seed": {str(seed): mean(vals) for seed, vals in sorted(by_seed.items())},
        "by_block": {block: mean(vals) for block, vals in sorted(by_block.items())},
        "by_file": {path: {"avg": mean(vals), "num_pairs": len(vals)} for path, vals in sorted(by_file.items())},
    }

    print(json.dumps(summary, indent=2))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
