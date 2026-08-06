#!/usr/bin/env python3
"""Run MTR-Bench's multi-turn monitor/evaluator loop with a KVPress model.

MTR-Bench is intentionally loaded from an external checkout so its official
``game_handlers.py`` and ``answer_evaluator.py`` remain the source of truth:
https://github.com/LittleCirc1e/mtr_bench
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import logging
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

CATEGORIES = (
    "information_query",
    "dynamic_adaptation",
    "state_operation",
    "strategic_gaming",
)

PRESS_NAMES = (
    "full",
    "h2o",
    "knorm",
    "none",
    "pyramidkv",
    "random",
    "rkv",
    "rkvlsh",
    "rpc",
    "scope",
    "snapkv",
    "snapkv_press",
    "streaming_llm",
    "turboquant",
)


def _stable_seed(base_seed: int, *parts: object) -> int:
    payload = "\0".join([str(base_seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "--", value).strip("-") or "unnamed"


def _load_module(module_name: str, source: Path) -> ModuleType:
    if not source.is_file():
        raise FileNotFoundError(f"Required MTR-Bench file not found: {source}")
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Python module from {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class MTRModules:
    game_handlers: ModuleType
    answer_evaluator: ModuleType


@dataclass(frozen=True)
class BenchmarkCase:
    category: str
    game_type: str
    difficulty: str
    max_round: int
    question_file: Path


@dataclass(frozen=True)
class BenchmarkTask:
    case: BenchmarkCase
    question: dict[str, Any]


def load_mtr_modules(mtr_root: Path) -> MTRModules:
    """Load the official MTR monitor and evaluator from a repository checkout."""
    root = mtr_root.expanduser().resolve()
    handlers = _load_module("kvpress_mtr_game_handlers", root / "game_handlers.py")
    evaluator = _load_module(
        "kvpress_mtr_answer_evaluator", root / "answer_evaluator.py"
    )
    if not hasattr(handlers, "get_game_handler"):
        raise AttributeError(f"{root / 'game_handlers.py'} has no get_game_handler()")
    if not hasattr(evaluator, "evaluate_answers"):
        raise AttributeError(
            f"{root / 'answer_evaluator.py'} has no evaluate_answers()"
        )
    return MTRModules(game_handlers=handlers, answer_evaluator=evaluator)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}: {exc}"
                ) from exc
    return rows


def load_latest_answers(path: Path) -> dict[str, dict[str, Any]]:
    """Return the latest checkpoint for each question from an append-only file."""
    if not path.exists():
        return {}
    answers: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path):
        answers[str(row["question_id"])] = row
    return answers


def rewrite_latest_answers(path: Path, answers: dict[str, dict[str, Any]]) -> None:
    """De-duplicate checkpoints using the same final JSONL shape as MTR-Bench."""
    path.parent.mkdir(parents=True, exist_ok=True)

    def sort_key(question_id: str) -> tuple[int, object]:
        return (0, int(question_id)) if question_id.isdigit() else (1, question_id)

    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for question_id in sorted(answers, key=sort_key):
            handle.write(json.dumps(answers[question_id], ensure_ascii=False) + "\n")
    temporary.replace(path)


def append_answer_checkpoint(path: Path, answer: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(answer, ensure_ascii=False) + "\n")
        handle.flush()


def add_round_limit(prompt: str, max_round: int) -> str:
    """Mirror MTR-Bench's instruction that the final answer has a turn limit."""
    marker = "\n\nReady to start"
    if marker not in prompt:
        return prompt
    instruction = (
        f"\n- You have {max_round} attempts to find the answer, which means "
        f"you need to output your answer in the {max_round}-th round or before this round."
    )
    return prompt.replace(marker, instruction + marker, 1)


def build_messages(
    question: dict[str, Any],
    turns: Iterable[dict[str, Any]],
    *,
    category: str,
    max_round: int,
) -> list[dict[str, str]]:
    prompt = str(question["prompt"])
    if category != "strategic_gaming":
        prompt = add_round_limit(prompt, max_round)
    messages = [{"role": "user", "content": prompt}]
    for turn in turns:
        messages.append({"role": "assistant", "content": str(turn["output"])})
        messages.append({"role": "user", "content": str(turn["feedback"])})
    return messages


def question_game_type(
    category: str, requested_game_type: str, question: dict[str, Any]
) -> str:
    """Use the same title parsing convention as MTR-Bench's official runner."""
    title = str(question.get("title", "")).strip()
    if not title:
        return requested_game_type
    if category == "strategic_gaming":
        return title
    return title.split(".", 1)[0].strip()


def turn_limit(category: str, question: dict[str, Any], max_round: int) -> int:
    """Strategic games define their own turn count; other categories use max_round."""
    if category == "strategic_gaming":
        configured = int(question.get("turns", 0))
        if configured <= 0:
            raise ValueError(
                f"Strategic question {question.get('question_id')} has no positive 'turns' value"
            )
        return configured
    return max_round


def should_stop(
    category: str,
    handler: Any,
    result: Any,
    feedback: Any,
    completed_turns: int,
    limit: int,
) -> bool:
    feedback_lower = str(feedback).lower()
    if completed_turns >= limit:
        return True
    if any(marker in feedback_lower for marker in ("invalid", "win", "lose")):
        return True
    if category != "strategic_gaming":
        return bool(handler.is_complete(result))
    return False


def _press_needs_attentions(press: object) -> bool:
    from kvpress import H2OPress, KnormPress, StreamingLLMPress

    return isinstance(press, (H2OPress, KnormPress, StreamingLLMPress))


def make_press(
    press_name: str,
    *,
    cache_budget: int,
    n_hash_buckets: int,
    lam: float,
    n_bits: int,
    snapkv_window_size: int,
    scope_decoding_cache_budget: int,
    scope_compress_interval: int,
    scope_decoding_window_size: int,
    rpc_window_size: int,
    rpc_compress_interval: int,
    rpc_kernel_size: int,
    device: str,
) -> object:
    from kvpress import (
        FullPress,
        H2OPress,
        KnormPress,
        NonePress,
        PyramidKVPress,
        RandomPress,
        RKVLSHPress,
        RKVPress,
        RPCPress,
        SCOPEPress,
        SnapKVPress,
        StreamingLLMPress,
        TurboQuantPress,
    )

    press_factories = {
        "full": FullPress,
        "h2o": H2OPress,
        "knorm": KnormPress,
        "none": NonePress,
        "pyramidkv": PyramidKVPress,
        "random": RandomPress,
        "rkv": RKVPress,
        "rkvlsh": RKVLSHPress,
        "rpc": RPCPress,
        "scope": SCOPEPress,
        "snapkv": SnapKVPress,
        "snapkv_press": SnapKVPress,
        "streaming_llm": StreamingLLMPress,
        "turboquant": TurboQuantPress,
    }
    if press_name not in press_factories:
        choices = ", ".join(PRESS_NAMES)
        raise ValueError(f"Unknown press '{press_name}'. Available presses: {choices}")
    press = press_factories[press_name]()
    press.cache_budget = cache_budget
    if press_name == "turboquant":
        press.n_bits = n_bits
        press.cache_budget = 0
    if press_name in ("snapkv", "snapkv_press", "pyramidkv"):
        press.window_size = snapkv_window_size
    if press_name == "rkvlsh":
        press.n_hash_buckets = n_hash_buckets
        press.lam = lam
        press.initialize_buckets(device=device)
    if press_name == "scope":
        press.window_size = snapkv_window_size
        press.decoding_cache_budget = scope_decoding_cache_budget
        press.compress_interval = scope_compress_interval
        press.decoding_window_size = scope_decoding_window_size
    if press_name == "rpc":
        press.window_size = rpc_window_size
        press.compress_interval = rpc_compress_interval
        press.kernel_size = rpc_kernel_size
    return press


class KVPressGenerator:
    """Transformers generation adapter used by the MTR multi-turn loop."""

    def __init__(
        self,
        *,
        model_name: str,
        device: str,
        press_name: str,
        cache_budget: int,
        max_new_tokens: int,
        max_context_length: Optional[int],
        do_sampling: bool,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        trust_remote_code: bool,
        think_mode: bool,
        n_hash_buckets: int,
        lam: float,
        n_bits: int,
        snapkv_window_size: int,
        scope_decoding_cache_budget: int,
        scope_compress_interval: int,
        scope_decoding_window_size: int,
        rpc_window_size: int,
        rpc_compress_interval: int,
        rpc_kernel_size: int,
        cache_mode: str,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self._none_press_type = __import__("kvpress", fromlist=["NonePress"]).NonePress
        self.model_name = model_name
        self.device = device
        self.press_name = press_name
        self.cache_budget = cache_budget
        self.max_new_tokens = max_new_tokens
        self.max_context_length = max_context_length
        self.do_sampling = do_sampling
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.think_mode = think_mode
        self.cache_mode = cache_mode
        self.uses_persistent_cache = cache_mode == "persistent"
        self.press_kwargs = {
            "cache_budget": cache_budget,
            "n_hash_buckets": n_hash_buckets,
            "lam": lam,
            "n_bits": n_bits,
            "snapkv_window_size": snapkv_window_size,
            "scope_decoding_cache_budget": scope_decoding_cache_budget,
            "scope_compress_interval": scope_compress_interval,
            "scope_decoding_window_size": scope_decoding_window_size,
            "rpc_window_size": rpc_window_size,
            "rpc_compress_interval": rpc_compress_interval,
            "rpc_kernel_size": rpc_kernel_size,
        }

        probe_press = make_press(
            press_name, device=self._initial_tensor_device(), **self.press_kwargs
        )
        attention_config: dict[str, Any] = {}
        if _press_needs_attentions(probe_press):
            attention_config = {
                "output_attentions": True,
                "attn_implementation": "eager",
            }

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if not getattr(self.tokenizer, "chat_template", None):
            raise ValueError(
                f"Tokenizer for {model_name} has no chat_template. "
                "MTR-Bench requires a chat/instruction model."
            )

        model_kwargs: dict[str, Any] = {
            "torch_dtype": "auto",
            "trust_remote_code": trust_remote_code,
            **attention_config,
        }
        if device == "auto":
            model_kwargs["device_map"] = "auto"
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, **model_kwargs
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, **model_kwargs
            )
            self.model.to(device)
        self.tensor_device = self._resolve_tensor_device()

    def _initial_tensor_device(self) -> str:
        torch = self._torch
        if self.device != "auto":
            return self.device
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    def _resolve_tensor_device(self) -> str:
        if self.device != "auto":
            return self.device
        if hasattr(self.model, "hf_device_map"):
            for entry in self.model.hf_device_map.values():
                if isinstance(entry, int):
                    return f"cuda:{entry}"
                if isinstance(entry, str) and entry not in ("cpu", "disk"):
                    return f"cuda:{entry}" if entry.isdigit() else entry
            if "cpu" in self.model.hf_device_map.values():
                return "cpu"
        model_device = str(getattr(self.model, "device", "cpu"))
        return "cpu" if model_device == "meta" else model_device

    def render_prompt(self, messages: list[dict[str, str]]) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if self.think_mode:
            prompt += "<think>\n"
        return prompt

    def generate(
        self, messages: list[dict[str, str]], *, seed: int
    ) -> tuple[str, dict[str, Any]]:
        torch = self._torch
        prompt = self.render_prompt(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.tensor_device)
        input_tokens = int(inputs["input_ids"].shape[1])
        if (
            self.max_context_length is not None
            and input_tokens > self.max_context_length
        ):
            raise ValueError(
                f"MTR conversation has {input_tokens} tokens, exceeding --max_context_length "
                f"{self.max_context_length}. Refusing to truncate multi-turn state."
            )

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        press = make_press(
            self.press_name, device=self.tensor_device, **self.press_kwargs
        )
        if hasattr(press, "reset_timing"):
            press.reset_timing()
        output_attentions = _press_needs_attentions(press)
        press_context = (
            press(self.model)
            if not isinstance(press, self._none_press_type)
            else contextlib.nullcontext()
        )
        sampling_kwargs: dict[str, Any] = {"do_sample": self.do_sampling}
        if self.do_sampling:
            sampling_kwargs.update(
                temperature=self.temperature,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
            )

        with torch.inference_mode(), press_context:
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                output_attentions=output_attentions,
                **sampling_kwargs,
            )

        generated_ids = output[0, input_tokens:]
        raw_text = self.tokenizer.decode(
            generated_ids, skip_special_tokens=True
        ).strip()
        response = raw_text.split("</think>")[-1].strip()
        metrics = {
            "seed": seed,
            "input_token_count": input_tokens,
            "output_token_count": int(generated_ids.shape[0]),
            "cache_budget": self.cache_budget,
            "cache_mode": "recompute",
        }
        if hasattr(press, "get_timing_metrics"):
            metrics.update(
                {
                    f"press_{key}": value
                    for key, value in press.get_timing_metrics().items()
                }
            )
        del output, generated_ids, inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return raw_text, {"parsed_output": response, **metrics}

    def start_session(self, messages: list[dict[str, str]], *, seed: int):
        if not self.uses_persistent_cache:
            raise RuntimeError("start_session() requires --cache_mode persistent")
        return KVPressPersistentSession(self, messages=messages, seed=seed)


class KVPressPersistentSession:
    """One MTR dialogue backed by a single continuously growing KV cache."""

    def __init__(
        self, generator: KVPressGenerator, *, messages: list[dict[str, str]], seed: int
    ):
        self.generator = generator
        self.torch = generator._torch
        self.messages = list(messages)
        self.seed = seed
        self.press = None
        self.cache = None
        self._press_context = None
        self._next_logits = None
        self._logical_token_ids: list[int] = []
        self._rendered_prompt = ""
        self._last_generated_ids: list[int] = []
        self._last_decoded = ""

    def __enter__(self):
        from transformers import DynamicCache

        torch = self.torch
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        self.press = make_press(
            self.generator.press_name,
            device=self.generator.tensor_device,
            **self.generator.press_kwargs,
        )
        if hasattr(self.press, "reset_timing"):
            self.press.reset_timing()
        self.cache = DynamicCache()
        self._press_context = (
            self.press(self.generator.model)
            if not isinstance(self.press, self.generator._none_press_type)
            else contextlib.nullcontext()
        )
        self._press_context.__enter__()

        self._rendered_prompt = self.generator.render_prompt(self.messages)
        prompt_ids = self.generator.tokenizer(
            self._rendered_prompt,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"][0].tolist()
        if not prompt_ids:
            raise ValueError("MTR chat template produced an empty initial prompt")
        self._check_context_limit(len(prompt_ids))
        self._next_logits = self._forward_ids(prompt_ids, initial_prefill=True)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._press_context is not None:
            self._press_context.__exit__(exc_type, exc_value, traceback)
        self._next_logits = None
        self.cache = None
        self.press = None
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
        return False

    def _check_context_limit(self, additional_tokens: int) -> None:
        limit = self.generator.max_context_length
        resulting_length = len(self._logical_token_ids) + additional_tokens
        if limit is not None and resulting_length > limit:
            raise ValueError(
                f"Persistent MTR dialogue would reach {resulting_length} logical tokens, "
                f"exceeding --max_context_length {limit}."
            )

    def _physical_cache_length(self) -> int:
        if self.cache is None:
            return 0
        try:
            return int(self.cache.get_seq_length())
        except Exception:
            return 0

    def _forward_ids(self, token_ids: list[int], *, initial_prefill: bool = False):
        if not token_ids:
            return self._next_logits
        torch = self.torch
        self._check_context_limit(len(token_ids))
        output_attentions = _press_needs_attentions(self.press)

        if initial_prefill:
            chunks = [token_ids]
        else:
            # q_len=1 is essential: KVPress must treat monitor feedback as a
            # continuation, not as a fresh prefill that resets RPC/SCOPE state.
            chunks = [[token_id] for token_id in token_ids]

        logits = None
        with torch.inference_mode():
            for chunk in chunks:
                start_position = len(self._logical_token_ids)
                input_ids = torch.tensor(
                    [chunk], dtype=torch.long, device=self.generator.tensor_device
                )
                positions = torch.arange(
                    start_position,
                    start_position + len(chunk),
                    dtype=torch.long,
                    device=self.generator.tensor_device,
                )
                outputs = self.generator.model(
                    input_ids=input_ids,
                    past_key_values=self.cache,
                    position_ids=positions.unsqueeze(0),
                    cache_position=positions,
                    use_cache=True,
                    output_attentions=output_attentions,
                )
                self.cache = outputs.past_key_values
                logits = outputs.logits[0, -1]
                self._logical_token_ids.extend(chunk)
        return logits

    def _sample_token(self, logits, *, seed: int) -> int:
        torch = self.torch
        scores = logits.float().clone()
        penalty = self.generator.repetition_penalty
        if penalty != 1.0 and self._logical_token_ids:
            seen = torch.tensor(
                sorted(set(self._logical_token_ids)),
                dtype=torch.long,
                device=scores.device,
            )
            seen_scores = scores[seen]
            scores[seen] = torch.where(
                seen_scores < 0,
                seen_scores * penalty,
                seen_scores / penalty,
            )

        if not self.generator.do_sampling:
            return int(scores.argmax().item())

        scores /= self.generator.temperature
        probabilities = torch.softmax(scores, dim=-1)
        if self.generator.top_p < 1.0:
            sorted_probabilities, sorted_indices = torch.sort(
                probabilities, descending=True
            )
            cumulative = torch.cumsum(sorted_probabilities, dim=-1)
            remove = cumulative > self.generator.top_p
            remove[1:] = remove[:-1].clone()
            remove[0] = False
            probabilities[sorted_indices[remove]] = 0
            probabilities /= probabilities.sum()
        rng = torch.Generator(device=probabilities.device)
        rng.manual_seed(seed + len(self._logical_token_ids))
        return int(torch.multinomial(probabilities, 1, generator=rng).item())

    def _eos_token_ids(self) -> set[int]:
        eos = self.generator.model.generation_config.eos_token_id
        if eos is None:
            eos = self.generator.tokenizer.eos_token_id
        if isinstance(eos, int):
            return {eos}
        return {int(token_id) for token_id in (eos or [])}

    def generate(self, *, seed: int) -> tuple[str, dict[str, Any]]:
        if self._next_logits is None:
            raise RuntimeError("Persistent session has not been initialized")
        before_cache = self._physical_cache_length()
        before_logical = len(self._logical_token_ids)
        before_timing = (
            self.press.get_timing_metrics()
            if hasattr(self.press, "get_timing_metrics")
            else {}
        )
        generated_ids: list[int] = []
        eos_ids = self._eos_token_ids()

        logits = self._next_logits
        for _ in range(self.generator.max_new_tokens):
            token_id = self._sample_token(logits, seed=seed)
            generated_ids.append(token_id)
            # Forward every sampled token, including EOS, so the next user turn
            # truly starts from the complete assistant-turn cache.
            logits = self._forward_ids([token_id])
            if token_id in eos_ids:
                break
        self._next_logits = logits
        self._last_generated_ids = generated_ids

        decoded = self.generator.tokenizer.decode(
            generated_ids, skip_special_tokens=True
        )
        self._last_decoded = decoded
        raw_text = decoded.strip()
        response = raw_text.split("</think>")[-1].strip()
        after_timing = (
            self.press.get_timing_metrics()
            if hasattr(self.press, "get_timing_metrics")
            else {}
        )
        metrics = {
            "seed": seed,
            "input_token_count": before_logical,
            "output_token_count": len(generated_ids),
            "cache_budget": self.generator.cache_budget,
            "cache_mode": "persistent",
            "logical_sequence_length": len(self._logical_token_ids),
            "physical_cache_length_before_turn": before_cache,
            "physical_cache_length_after_turn": self._physical_cache_length(),
            "output_token_ids": generated_ids,
        }
        for key, value in after_timing.items():
            metrics[f"press_{key}"] = value - before_timing.get(key, 0)

        assistant_content = ("<think>\n" if self.generator.think_mode else "") + decoded
        self.messages.append({"role": "assistant", "content": assistant_content})
        return raw_text, {"parsed_output": response, **metrics}

    def append_feedback(self, feedback: str) -> None:
        if not self.messages or self.messages[-1]["role"] != "assistant":
            raise RuntimeError("Monitor feedback must follow an assistant turn")
        self.messages.append({"role": "user", "content": feedback})
        next_prompt = self.generator.render_prompt(self.messages)
        next_prompt_ids = self.generator.tokenizer(
            next_prompt,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"][0].tolist()
        cached_length = len(self._logical_token_ids)
        if next_prompt_ids[:cached_length] != self._logical_token_ids:
            raise ValueError(
                "Tokenizer chat template does not preserve the already cached token prefix across turns. "
                "Exact persistent KV-cache continuation is therefore impossible for this tokenizer; "
                "use --cache_mode recompute."
            )
        suffix_ids = next_prompt_ids[cached_length:]
        if not suffix_ids:
            raise ValueError("Chat template produced no tokens for monitor feedback")
        self._next_logits = self._forward_ids(suffix_ids)
        self._rendered_prompt = next_prompt


def replay_monitor(
    handler: Any, turns: list[dict[str, Any]], *, question_id: str
) -> None:
    """Restore official monitor state and detect incompatible/corrupt checkpoints."""
    for index, turn in enumerate(turns, start=1):
        result, feedback = handler.parse_response(str(turn["output"]))
        if result != turn.get("result") or feedback != turn.get("feedback"):
            raise RuntimeError(
                f"Cannot resume question {question_id}: replayed monitor state differs "
                f"at turn {index}. Stored ({turn.get('result')!r}, {turn.get('feedback')!r}); "
                f"replayed ({result!r}, {feedback!r}). Use a new --run_tag or --overwrite."
            )


def run_question(
    *,
    question: dict[str, Any],
    requested_game_type: str,
    category: str,
    max_round: int,
    base_seed: int,
    existing_turns: list[dict[str, Any]],
    get_game_handler: Any,
    generator: Any,
    checkpoint: Any,
) -> dict[str, Any]:
    question_id = str(question["question_id"])
    game_type = question_game_type(category, requested_game_type, question)
    limit = turn_limit(category, question, max_round)

    # MTR monitors use Python's module-level random generator. A stable,
    # question-local seed plus deterministic replay makes resume reliable.
    random.seed(_stable_seed(base_seed, requested_game_type, question_id, "monitor"))
    handler = get_game_handler(game_type=game_type, question=question)
    turns = list(existing_turns)
    replay_monitor(handler, turns, question_id=question_id)

    if turns and should_stop(
        category,
        handler,
        turns[-1].get("result"),
        turns[-1].get("feedback"),
        len(turns),
        limit,
    ):
        return {"question_id": question["question_id"], "turns": turns}

    persistent = bool(getattr(generator, "uses_persistent_cache", False))
    if persistent and turns:
        # JSONL checkpoints contain the dialogue, not tensor-valued KV state.
        # Replaying text through a compressed cache would not reproduce the
        # original online decisions, so restart only this incomplete task.
        logger.warning(
            "Restarting incomplete persistent-cache question %s from turn 1; "
            "KV tensors are not serialized in JSONL checkpoints",
            question_id,
        )
        turns = []
        random.seed(
            _stable_seed(base_seed, requested_game_type, question_id, "monitor")
        )
        handler = get_game_handler(game_type=game_type, question=question)

    initial_messages = build_messages(
        question, [], category=category, max_round=max_round
    )
    initial_seed = _stable_seed(base_seed, requested_game_type, question_id, "prefill")
    session_context = (
        generator.start_session(initial_messages, seed=initial_seed)
        if persistent
        else contextlib.nullcontext(None)
    )
    with session_context as session:
        while len(turns) < limit:
            round_number = len(turns) + 1
            generation_seed = _stable_seed(
                base_seed, requested_game_type, question_id, round_number, "model"
            )
            monitor_random_state = random.getstate()
            if persistent:
                raw_text, generation = session.generate(seed=generation_seed)
            else:
                messages = build_messages(
                    question, turns, category=category, max_round=max_round
                )
                raw_text, generation = generator.generate(
                    messages, seed=generation_seed
                )
            # Model implementations may use Python's global RNG. Keep that from
            # changing the official monitor's deterministic environment trajectory.
            random.setstate(monitor_random_state)
            output = str(generation.pop("parsed_output"))
            try:
                result, feedback = handler.parse_response(output)
            except Exception as exc:
                result = "Invalid"
                feedback = (
                    f"Invalid response: monitor raised {type(exc).__name__}: {exc}"
                )

            turn = {
                "round": round_number,
                "raw_output": raw_text,
                "output": output,
                "result": result,
                "feedback": feedback,
                "kvpress": generation,
            }
            turns.append(turn)
            answer = {"question_id": question["question_id"], "turns": turns}
            checkpoint(answer)
            if should_stop(category, handler, result, feedback, len(turns), limit):
                break
            if persistent:
                session.append_feedback(str(feedback))
    return {"question_id": question["question_id"], "turns": turns}


def result_paths(
    *,
    result_dir: Path,
    category: str,
    game_type: str,
    difficulty: str,
    max_round: int,
    model_name: str,
    press_name: str,
    cache_budget: int,
    seed: int,
    run_tag: str,
    configuration_tag: str = "",
) -> tuple[Path, Path, Path]:
    stem = f"{_safe_name(model_name)}__{press_name}__budget{cache_budget}"
    if configuration_tag:
        stem += f"__{_safe_name(configuration_tag)}"
    stem += f"__seed{seed}"
    if run_tag:
        stem += f"__{_safe_name(run_tag)}"
    directory = result_dir / category / game_type / difficulty / str(max_round)
    return (
        directory / f"{stem}.jsonl",
        directory / f"{stem}_eval.json",
        directory / f"{stem}_questions.jsonl",
    )


def configuration_tag(args: argparse.Namespace) -> str:
    parts: list[str] = [f"cache-{args.cache_mode}"]
    if args.press_name in ("rkv", "rkvlsh"):
        parts.extend((f"hash{args.n_hash_buckets}", f"lam{args.lam:g}"))
    elif args.press_name in ("snapkv", "snapkv_press", "pyramidkv"):
        parts.append(f"window{args.snapkv_window_size}")
    elif args.press_name == "turboquant":
        parts.append(f"bits{args.n_bits}")
    elif args.press_name == "scope":
        parts.extend(
            (
                f"window{args.snapkv_window_size}",
                f"decodebudget{args.scope_decoding_cache_budget}",
                f"interval{args.scope_compress_interval}",
                f"decodewindow{args.scope_decoding_window_size}",
            )
        )
    elif args.press_name == "rpc":
        parts.extend(
            (
                f"window{args.rpc_window_size}",
                f"interval{args.rpc_compress_interval}",
                f"kernel{args.rpc_kernel_size}",
            )
        )
    parts.append(f"maxtokens{args.max_new_tokens}")
    if args.do_sampling:
        parts.extend((f"temp{args.temperature:g}", f"topp{args.top_p:g}"))
    else:
        parts.append("greedy")
    if args.num_samples > 0:
        parts.append(f"numsamples{args.num_samples}")
        parts.append(args.sampling_strategy)
    if args.dataset_block_index > 0:
        parts.append(f"block{args.dataset_block_index}_size{args.dataset_block_size}")
    return "__".join(parts)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def select_questions(
    questions: list[dict[str, Any]], *, num_samples: int, seed: int
) -> list[dict[str, Any]]:
    selected = list(questions)
    random.Random(seed).shuffle(selected)
    if num_samples > 0:
        if num_samples > len(selected):
            raise ValueError(
                f"num_samples={num_samples} exceeds dataset size {len(selected)}"
            )
        selected = selected[:num_samples]
    return selected


def resolve_categories(requested: list[str]) -> list[str]:
    if "all" in requested:
        if len(requested) != 1:
            raise ValueError(
                "--category all cannot be combined with individual categories"
            )
        return list(CATEGORIES)
    return requested


def discover_cases(
    mtr_root: Path,
    *,
    categories: list[str],
    game_types: list[str],
    difficulties: list[str],
    max_rounds: list[int],
) -> list[BenchmarkCase]:
    all_games = game_types == ["all"]
    if "all" in game_types and not all_games:
        raise ValueError(
            "--game_type all cannot be combined with individual game types"
        )

    cases: list[BenchmarkCase] = []
    found_games: set[str] = set()
    for category in categories:
        category_dir = mtr_root / "problem" / category
        if not category_dir.is_dir():
            raise FileNotFoundError(f"MTR category directory not found: {category_dir}")
        available_games = sorted(
            path.name for path in category_dir.iterdir() if path.is_dir()
        )
        selected_games = (
            available_games
            if all_games
            else [game for game in game_types if game in available_games]
        )
        found_games.update(selected_games)
        for game_type in selected_games:
            for difficulty in difficulties:
                question_file = category_dir / game_type / f"{difficulty}.jsonl"
                if not question_file.is_file():
                    raise FileNotFoundError(
                        f"MTR question file not found: {question_file}"
                    )
                for max_round in max_rounds:
                    cases.append(
                        BenchmarkCase(
                            category=category,
                            game_type=game_type,
                            difficulty=difficulty,
                            max_round=max_round,
                            question_file=question_file,
                        )
                    )

    if not all_games:
        missing = sorted(set(game_types) - found_games)
        if missing:
            raise ValueError(
                "Requested game types were not found in the selected categories: "
                + ", ".join(missing)
            )
    if not cases:
        raise ValueError(
            "No MTR-Bench cases matched the requested categories and game types"
        )
    return cases


def build_task_pool(cases: Iterable[BenchmarkCase]) -> list[BenchmarkTask]:
    pool: list[BenchmarkTask] = []
    for case in cases:
        pool.extend(
            BenchmarkTask(case=case, question=question)
            for question in load_jsonl(case.question_file)
        )
    return pool


def select_task_pool(
    pool: list[BenchmarkTask],
    *,
    num_samples: int,
    seed: int,
    sampling_strategy: str,
    dataset_block_index: int,
    dataset_block_size: int,
) -> tuple[list[BenchmarkTask], dict[str, int]]:
    if num_samples > len(pool):
        raise ValueError(
            f"num_samples={num_samples} exceeds the full task pool size {len(pool)}"
        )

    rng = random.Random(seed)
    if num_samples > 0 and sampling_strategy == "stratified":
        by_case: dict[BenchmarkCase, list[BenchmarkTask]] = {}
        for task in pool:
            by_case.setdefault(task.case, []).append(task)
        case_order = list(by_case)
        rng.shuffle(case_order)
        for tasks in by_case.values():
            rng.shuffle(tasks)

        sampled = []
        while len(sampled) < num_samples:
            made_progress = False
            for case in case_order:
                if by_case[case]:
                    sampled.append(by_case[case].pop())
                    made_progress = True
                    if len(sampled) == num_samples:
                        break
            if not made_progress:
                break
    else:
        sampled = list(pool)
        rng.shuffle(sampled)
        if num_samples > 0:
            sampled = sampled[:num_samples]

    # Blocks should contain a deterministic mixture of cases instead of being
    # grouped by the stratified round-robin order.
    rng.shuffle(sampled)

    block_start = 0
    block_end = len(sampled)
    if dataset_block_index > 0:
        block_start = (dataset_block_index - 1) * dataset_block_size
        if block_start >= len(sampled):
            raise ValueError(
                f"dataset_block_index={dataset_block_index} with dataset_block_size={dataset_block_size} "
                f"starts at task {block_start + 1}, beyond the sampled pool of {len(sampled)}"
            )
        block_end = min(block_start + dataset_block_size, len(sampled))

    return sampled[block_start:block_end], {
        "full_pool_size": len(pool),
        "sampled_pool_size": len(sampled),
        "block_start": block_start + 1 if sampled else 0,
        "block_end": block_end,
        "selected_pool_size": block_end - block_start,
    }


def group_tasks_by_case(
    tasks: Iterable[BenchmarkTask],
) -> dict[BenchmarkCase, list[dict[str, Any]]]:
    grouped: dict[BenchmarkCase, list[dict[str, Any]]] = {}
    for task in tasks:
        grouped.setdefault(task.case, []).append(task.question)
    return grouped


def evaluator_questions(questions: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Match MTR's evaluator, which coerces answer IDs to integers."""
    normalized: list[dict[str, Any]] = []
    for question in questions:
        copied = dict(question)
        question_id = copied.get("question_id")
        if isinstance(question_id, str) and question_id.isdigit():
            copied["question_id"] = int(question_id)
        normalized.append(copied)
    return normalized


def add_evaluation_metadata(
    eval_file: Path,
    *,
    args: argparse.Namespace,
    mtr_root: Path,
    category: str,
    game_type: str,
    difficulty: str,
    max_round: int,
    selected_questions: int,
    selection_summary: dict[str, int],
) -> None:
    with eval_file.open(encoding="utf-8") as handle:
        evaluation = json.load(handle)
    evaluation["kvpress_config"] = {
        "pipeline": "MTR-Bench game_handlers.py + answer_evaluator.py",
        "mtr_root": str(mtr_root),
        "model_name": args.model_name,
        "category": category,
        "game_type": game_type,
        "difficulty": difficulty,
        "max_round": max_round,
        "selected_questions": selected_questions,
        "press_name": args.press_name,
        "cache_budget": args.cache_budget,
        "cache_mode": args.cache_mode,
        "configuration": configuration_tag(args),
        "max_context_length": args.max_context_length,
        "seed": args.seed,
        "run_tag": args.run_tag,
        "sampling_strategy": args.sampling_strategy,
        "global_selection": selection_summary,
        "dataset_block_index": args.dataset_block_index,
        "dataset_block_size": args.dataset_block_size,
    }
    with eval_file.open("w", encoding="utf-8") as handle:
        json.dump(evaluation, handle, indent=2, ensure_ascii=False)


def write_aggregate_evaluation(
    eval_paths: list[Path],
    *,
    args: argparse.Namespace,
    categories: list[str],
    selection_summary: dict[str, int],
) -> Optional[Path]:
    if not eval_paths:
        return None
    evaluations = [json.loads(path.read_text(encoding="utf-8")) for path in eval_paths]
    total_questions = sum(int(item.get("total_questions", 0)) for item in evaluations)
    successful_games = sum(int(item.get("successful_games", 0)) for item in evaluations)
    detailed_results = [
        result
        for evaluation in evaluations
        for result in evaluation.get("detailed_results", [])
    ]
    average_turns = (
        sum(int(result.get("num_turns", 0)) for result in detailed_results)
        / len(detailed_results)
        if detailed_results
        else 0.0
    )

    by_category: dict[str, dict[str, int | float]] = {}
    for evaluation in evaluations:
        category = evaluation["kvpress_config"]["category"]
        summary = by_category.setdefault(
            category,
            {"total_questions": 0, "successful_games": 0, "accuracy": 0.0},
        )
        summary["total_questions"] += int(evaluation.get("total_questions", 0))
        summary["successful_games"] += int(evaluation.get("successful_games", 0))
    for summary in by_category.values():
        count = int(summary["total_questions"])
        summary["accuracy"] = int(summary["successful_games"]) / count if count else 0.0

    scope = "all" if args.category == ["all"] else "-".join(categories)
    difficulty = "-".join(args.difficulty)
    rounds = "-".join(str(value) for value in args.max_round)
    stem = (
        f"aggregate__{_safe_name(args.model_name)}__{args.press_name}"
        f"__budget{args.cache_budget}__{_safe_name(configuration_tag(args))}"
        f"__categories-{_safe_name(scope)}__difficulty-{_safe_name(difficulty)}"
        f"__rounds-{rounds}__seed{args.seed}"
    )
    if args.run_tag:
        stem += f"__{_safe_name(args.run_tag)}"
    aggregate_path = args.result_dir / f"{stem}.json"
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate = {
        "total_questions": total_questions,
        "successful_games": successful_games,
        "accuracy": successful_games / total_questions if total_questions else 0.0,
        "average_turns": average_turns,
        "evaluated_case_count": len(evaluations),
        "by_category": by_category,
        "global_selection": selection_summary,
        "evaluation_files": [str(path) for path in eval_paths],
    }
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return aggregate_path


def evaluate_mtr(
    args: argparse.Namespace, generator: Optional[Any] = None
) -> list[Path]:
    from tqdm import tqdm

    mtr_root = args.mtr_root.expanduser().resolve()
    modules = load_mtr_modules(mtr_root)
    categories = resolve_categories(args.category)
    cases = discover_cases(
        mtr_root,
        categories=categories,
        game_types=args.game_type,
        difficulties=args.difficulty,
        max_rounds=args.max_round,
    )
    full_pool = build_task_pool(cases)
    selected_tasks, selection_summary = select_task_pool(
        full_pool,
        num_samples=args.num_samples,
        seed=args.seed,
        sampling_strategy=args.sampling_strategy,
        dataset_block_index=args.dataset_block_index,
        dataset_block_size=args.dataset_block_size,
    )
    selected_by_case = group_tasks_by_case(selected_tasks)
    logger.info(
        "Selected %d tasks from %d total (%d after --num_samples); block range %d-%d",
        selection_summary["selected_pool_size"],
        selection_summary["full_pool_size"],
        selection_summary["sampled_pool_size"],
        selection_summary["block_start"],
        selection_summary["block_end"],
    )

    if generator is None:
        generator = KVPressGenerator(
            model_name=args.model_name,
            device=args.device,
            press_name=args.press_name,
            cache_budget=args.cache_budget,
            max_new_tokens=args.max_new_tokens,
            max_context_length=args.max_context_length,
            do_sampling=args.do_sampling,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            trust_remote_code=args.trust_remote_code,
            think_mode=args.think_mode,
            n_hash_buckets=args.n_hash_buckets,
            lam=args.lam,
            n_bits=args.n_bits,
            snapkv_window_size=args.snapkv_window_size,
            scope_decoding_cache_budget=args.scope_decoding_cache_budget,
            scope_compress_interval=args.scope_compress_interval,
            scope_decoding_window_size=args.scope_decoding_window_size,
            rpc_window_size=args.rpc_window_size,
            rpc_compress_interval=args.rpc_compress_interval,
            rpc_kernel_size=args.rpc_kernel_size,
            cache_mode=args.cache_mode,
        )

    eval_paths: list[Path] = []
    for case in cases:
        selected = selected_by_case.get(case, [])
        if not selected:
            continue
        answer_file, eval_file, selected_question_file = result_paths(
            result_dir=args.result_dir,
            category=case.category,
            game_type=case.game_type,
            difficulty=case.difficulty,
            max_round=case.max_round,
            model_name=args.model_name,
            press_name=args.press_name,
            cache_budget=args.cache_budget,
            seed=args.seed,
            run_tag=args.run_tag,
            configuration_tag=configuration_tag(args),
        )
        if args.overwrite:
            for path in (answer_file, eval_file, selected_question_file):
                path.unlink(missing_ok=True)
        selected_ids = {str(question["question_id"]) for question in selected}
        existing = {
            question_id: answer
            for question_id, answer in load_latest_answers(answer_file).items()
            if question_id in selected_ids
        }
        write_jsonl(selected_question_file, evaluator_questions(selected))

        description = f"{case.category}/{case.game_type}/{case.difficulty}/rounds={case.max_round}"
        for question in tqdm(selected, desc=description):
            question_id = str(question["question_id"])

            def checkpoint(answer: dict[str, Any]) -> None:
                append_answer_checkpoint(answer_file, answer)
                existing[question_id] = answer

            answer = run_question(
                question=question,
                requested_game_type=case.game_type,
                category=case.category,
                max_round=case.max_round,
                base_seed=_stable_seed(
                    args.seed,
                    case.category,
                    case.game_type,
                    case.difficulty,
                    case.max_round,
                ),
                existing_turns=existing.get(question_id, {}).get("turns", []),
                get_game_handler=modules.game_handlers.get_game_handler,
                generator=generator,
                checkpoint=checkpoint,
            )
            existing[question_id] = answer

        rewrite_latest_answers(answer_file, existing)
        modules.answer_evaluator.evaluate_answers(
            question_file=str(selected_question_file),
            answer_file=str(answer_file),
            eval_file=str(eval_file),
            game_type=case.game_type,
        )
        add_evaluation_metadata(
            eval_file,
            args=args,
            mtr_root=mtr_root,
            category=case.category,
            game_type=case.game_type,
            difficulty=case.difficulty,
            max_round=case.max_round,
            selected_questions=len(selected),
            selection_summary=selection_summary,
        )
        logger.info("Saved answers to %s", answer_file)
        logger.info("Saved MTR evaluation to %s", eval_file)
        eval_paths.append(eval_file)
    aggregate_path = write_aggregate_evaluation(
        eval_paths,
        args=args,
        categories=categories,
        selection_summary=selection_summary,
    )
    if aggregate_path is not None:
        logger.info("Saved aggregate MTR evaluation to %s", aggregate_path)
    return eval_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a Transformers + KVPress model on the official multi-turn MTR-Bench pipeline."
    )
    parser.add_argument(
        "--mtr_root",
        type=Path,
        required=True,
        help="Checkout of LittleCirc1e/mtr_bench",
    )
    parser.add_argument("--model_name", required=True)
    parser.add_argument(
        "--category",
        nargs="+",
        choices=(*CATEGORIES, "all"),
        default=["information_query"],
        help="One or more categories, or 'all'",
    )
    parser.add_argument(
        "--game_type",
        nargs="+",
        default=["all"],
        help="One or more game types, or 'all' to discover every type in the selected categories",
    )
    parser.add_argument("--difficulty", nargs="+", default=["easy"])
    parser.add_argument("--max_round", nargs="+", type=int, default=[10])
    parser.add_argument(
        "--num_samples",
        type=int,
        default=0,
        help="Total questions sampled across the complete requested pool; 0 selects the full pool",
    )
    parser.add_argument(
        "--sampling_strategy",
        choices=("stratified", "random"),
        default="stratified",
        help="Stratified balances the global sample across selected game/difficulty/round cases",
    )
    parser.add_argument(
        "--dataset_block_index",
        type=int,
        default=0,
        help="1-based block of the globally sampled pool; 0 selects all sampled questions",
    )
    parser.add_argument("--dataset_block_size", type=int, default=50)
    parser.add_argument(
        "--result_dir", type=Path, default=Path(__file__).parent / "mtr_results"
    )
    parser.add_argument("--run_tag", default="")
    parser.add_argument(
        "--overwrite", action=argparse.BooleanOptionalAction, default=False
    )

    parser.add_argument("--device", default="auto")
    parser.add_argument("--press_name", choices=PRESS_NAMES, default="knorm")
    parser.add_argument("--cache_budget", type=int, default=512)
    parser.add_argument(
        "--cache_mode",
        choices=("persistent", "recompute"),
        default="persistent",
        help=(
            "persistent prefills once and extends one KV cache across all dialogue turns; "
            "recompute rebuilds the full transcript before every assistant turn"
        ),
    )
    parser.add_argument("--max_new_tokens", type=int, default=16384)
    parser.add_argument("--max_context_length", type=int)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--do_sampling", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument(
        "--think_mode", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--trust_remote_code", action=argparse.BooleanOptionalAction, default=True
    )

    parser.add_argument("--n_hash_buckets", type=int, default=6)
    parser.add_argument("--lam", type=float, default=0.1)
    parser.add_argument("--n_bits", type=int, default=4)
    parser.add_argument("--snapkv_window_size", type=int, default=64)
    parser.add_argument("--scope_decoding_cache_budget", type=int, default=0)
    parser.add_argument("--scope_compress_interval", type=int, default=32)
    parser.add_argument("--scope_decoding_window_size", type=int, default=8)
    parser.add_argument("--rpc_window_size", type=int, default=32)
    parser.add_argument("--rpc_compress_interval", type=int, default=128)
    parser.add_argument("--rpc_kernel_size", type=int, default=7)
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = build_parser().parse_args()
    if any(rounds <= 0 for rounds in args.max_round):
        raise ValueError("--max_round values must be positive")
    if args.num_samples < 0:
        raise ValueError("--num_samples must be non-negative")
    if args.dataset_block_index < 0:
        raise ValueError("--dataset_block_index must be non-negative")
    if args.dataset_block_size <= 0:
        raise ValueError("--dataset_block_size must be positive")
    if args.do_sampling and args.temperature <= 0:
        raise ValueError("--temperature must be positive when sampling")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top_p must be in (0, 1]")
    if args.repetition_penalty <= 0:
        raise ValueError("--repetition_penalty must be positive")
    evaluate_mtr(args)


if __name__ == "__main__":
    main()
