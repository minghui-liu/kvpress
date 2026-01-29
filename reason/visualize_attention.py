# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Visualize summed attention weights as a TEXT HEATMAP (HTML).

Shows BOTH prefix and generated tokens, each colored by attention received.
    Red = low attention, Yellow = medium, Green = high attention

Usage (matching evaluate.py interface):
    python visualize_attention.py \
        --dataset=math500 \
        --model_name=deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
        --press_name=full \
        --cache_budget=1536 \
        --num_samples=1 \
        --random_seed=1 \
        --max_new_tokens=32768
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
from datasets import load_dataset
from fire import Fire
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

# Import formatters from evaluate.py's directory
from math500 import math500_formatter
from gsm8k import gsm8k_formatter
from aime24 import aime24_formatter
from aime25 import aime25_formatter

logger = logging.getLogger(__name__)

# Dataset configs (same as evaluate.py)
DATASET_DICT = {
    "gsm8k": ("openai/gsm8k", "main", "test"),
    "math500": ("HuggingFaceH4/MATH-500", None, "test"),
    "aime25": ("math-ai/aime25", None, "test"),
    "aime24": ("math-ai/aime24", None, "test"),
}

FORMATTER_DICT = {
    "gsm8k": gsm8k_formatter,
    "math500": math500_formatter,
    "aime25": aime25_formatter,
    "aime24": aime24_formatter,
}


def score_to_color(normalized: float) -> str:
    """Convert normalized score (0-1) to RGB color. Red=low, Yellow=mid, Green=high."""
    normalized = max(0, min(1, normalized))
    
    if normalized < 0.5:
        r = 255
        g = int(255 * (normalized * 2))
        b = 0
    else:
        r = int(255 * (1 - (normalized - 0.5) * 2))
        g = 255
        b = 0
    
    return f"rgb({r},{g},{b})"


def normalize_scores_percentile(scores: List[float]) -> List[float]:
    """Normalize scores using percentile ranking for better color distribution."""
    if not scores:
        return []
    
    non_zero = [s for s in scores if s > 0]
    if not non_zero:
        return [0.0] * len(scores)
    
    sorted_scores = sorted(set(non_zero))
    rank_map = {s: i / (len(sorted_scores) - 1) if len(sorted_scores) > 1 else 0.5 
                for i, s in enumerate(sorted_scores)}
    
    normalized = []
    for s in scores:
        if s == 0:
            normalized.append(0.0)
        else:
            normalized.append(rank_map.get(s, 0.5))
    
    return normalized


def build_token_spans(tokens: List[str], scores: List[float], is_generated: bool = False) -> str:
    """Build HTML spans for tokens with attention-based coloring."""
    normalized_scores = normalize_scores_percentile(scores)
    
    token_spans = []
    for pos, token_text in enumerate(tokens):
        score = scores[pos] if pos < len(scores) else 0
        norm_score = normalized_scores[pos] if pos < len(normalized_scores) else 0
        
        color = score_to_color(norm_score)
        
        # Escape HTML and handle whitespace visually
        display_text = token_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        display_text = display_text.replace('\n', '↵')
        if display_text == ' ':
            display_text = '·'
        
        # Add border for generated tokens
        border = "border-bottom: 2px solid #8f8;" if is_generated else ""
        
        token_type = "GEN" if is_generated else "PREFIX"
        title = f"pos={pos} attn={score:.4f} norm={norm_score:.3f} [{token_type}]"
        
        token_spans.append(
            f'<span class="token" style="background-color: {color}; color: black; {border}" '
            f'title="{title}">{display_text}</span>'
        )
    
    return ''.join(token_spans)


def create_html_heatmap(
    prefix_tokens: List[str],
    prefix_scores: List[float],
    gen_tokens: List[str],
    gen_scores: List[float],
    model_name: str,
    input_text: str,
) -> str:
    """Create HTML with BOTH prefix and generated tokens colored by attention."""
    
    prefix_len = len(prefix_tokens)
    num_generated = len(gen_tokens)
    
    # Stats for prefix
    prefix_min = min(prefix_scores) if prefix_scores else 0
    prefix_max = max(prefix_scores) if prefix_scores else 1
    prefix_avg = sum(prefix_scores) / len(prefix_scores) if prefix_scores else 0
    
    # Stats for generated
    gen_min = min(gen_scores) if gen_scores else 0
    gen_max = max(gen_scores) if gen_scores else 1
    gen_avg = sum(gen_scores) / len(gen_scores) if gen_scores else 0
    
    html_parts = []
    html_parts.append('<!DOCTYPE html>')
    html_parts.append('<html><head>')
    html_parts.append('<meta charset="UTF-8">')
    html_parts.append(f'<title>Attention Heatmap - {model_name.split("/")[-1]}</title>')
    html_parts.append('''<style>
body { 
    font-family: Arial, sans-serif; 
    max-width: 1400px; 
    margin: 0 auto; 
    padding: 20px; 
    background: #2d2d2d; 
    color: white; 
}
.token {
    padding: 2px 1px;
    border-radius: 2px;
    display: inline;
}
.stats {
    background: #1a1a1a;
    padding: 15px;
    border-radius: 5px;
    margin: 10px 0;
}
.legend {
    margin-top: 15px;
    font-size: 12px;
}
.legend span {
    padding: 2px 8px;
    margin-right: 10px;
}
.content {
    font-family: monospace;
    line-height: 1.8;
    background: #1a1a1a;
    padding: 15px;
    border-radius: 5px;
    word-wrap: break-word;
}
.section {
    margin: 20px 0;
    padding: 15px;
    border: 1px solid #444;
    border-radius: 8px;
}
h2 { color: #88f; margin-top: 30px; }
h3 { color: #8f8; }
</style>''')
    html_parts.append('</head><body>')
    
    html_parts.append(f'<h1>Attention Weight Distribution</h1>')
    html_parts.append(f'<p><strong>Model:</strong> {model_name}</p>')
    html_parts.append(f'<p><strong>Total:</strong> {prefix_len} prefix tokens + {num_generated} generated tokens = {prefix_len + num_generated} total</p>')
    
    # Legend at top
    html_parts.append('<div class="legend">')
    html_parts.append('<strong>Legend:</strong> ')
    html_parts.append('<span style="background: rgb(255,0,0); color: white;">Low attention</span>')
    html_parts.append('<span style="background: rgb(255,255,0); color: black;">Medium</span>')
    html_parts.append('<span style="background: rgb(0,255,0); color: black;">High attention</span>')
    html_parts.append('<span style="border-bottom: 2px solid #8f8; padding: 2px 5px; background: #333;">Generated token</span>')
    html_parts.append('</div>')
    
    # ========== PREFIX SECTION ==========
    html_parts.append('<div class="section">')
    html_parts.append('<h2>📝 Prefix (Input Question)</h2>')
    html_parts.append('<div class="stats">')
    html_parts.append(f'<p><strong>{prefix_len} tokens</strong> — Colored by attention received from all {num_generated} generated tokens</p>')
    html_parts.append(f'<p>Attention: Min={prefix_min:.4f} | Max={prefix_max:.4f} | Avg={prefix_avg:.4f}</p>')
    html_parts.append('</div>')
    
    html_parts.append('<div class="content">')
    html_parts.append(build_token_spans(prefix_tokens, prefix_scores, is_generated=False))
    html_parts.append('</div>')
    html_parts.append('</div>')
    
    # ========== GENERATED SECTION ==========
    html_parts.append('<div class="section">')
    html_parts.append('<h2>🤖 Generated Response</h2>')
    html_parts.append('<div class="stats">')
    html_parts.append(f'<p><strong>{num_generated} tokens</strong> — Colored by attention received from subsequent tokens</p>')
    html_parts.append(f'<p>Attention: Min={gen_min:.4f} | Max={gen_max:.4f} | Avg={gen_avg:.4f}</p>')
    html_parts.append('</div>')
    
    html_parts.append('<div class="content">')
    html_parts.append(build_token_spans(gen_tokens, gen_scores, is_generated=True))
    html_parts.append('</div>')
    html_parts.append('</div>')
    
    # ========== TOP TOKENS TABLE ==========
    # Combine all tokens for ranking
    all_tokens = prefix_tokens + gen_tokens
    all_scores = prefix_scores + gen_scores
    
    top_k = min(30, len(all_tokens))
    top_indices = np.argsort(all_scores)[-top_k:][::-1]
    
    html_parts.append(f'<h2>🏆 Top {top_k} Most-Attended Tokens</h2>')
    html_parts.append('<div class="stats">')
    html_parts.append('<table style="width: 100%; border-collapse: collapse;">')
    html_parts.append('<tr><th style="text-align:left; padding:5px;">Rank</th><th style="text-align:left; padding:5px;">Position</th><th style="text-align:left; padding:5px;">Type</th><th style="text-align:left; padding:5px;">Token</th><th style="text-align:left; padding:5px;">Attention</th></tr>')
    
    for rank, idx in enumerate(top_indices):
        token = all_tokens[idx] if idx < len(all_tokens) else "<unk>"
        token_escaped = token.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('\n', '↵')
        score = all_scores[idx]
        token_type = "PREFIX" if idx < prefix_len else "GEN"
        type_color = "#88f" if idx < prefix_len else "#8f8"
        html_parts.append(f'<tr><td style="padding:5px;">{rank+1}</td><td style="padding:5px;">{idx}</td><td style="padding:5px; color:{type_color};">{token_type}</td><td style="padding:5px; font-family:monospace;">{token_escaped}</td><td style="padding:5px;">{score:.4f}</td></tr>')
    
    html_parts.append('</table>')
    html_parts.append('</div>')
    
    html_parts.append('</body></html>')
    
    return '\n'.join(html_parts)


class StopOnBoxed(StoppingCriteria):
    """Stop when \\boxed{...} is detected (same as evaluate.py)."""
    def __init__(self, prompt_len: int, tokenizer):
        super().__init__()
        self.prompt_len = prompt_len
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        gen_ids = input_ids[0, self.prompt_len:].tolist()
        if len(gen_ids) < 5:
            return False
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        if "\\boxed{" in text:
            idx = text.rfind("\\boxed{")
            if "}" in text[idx + 7:]:
                return True
        return False


class AttentionAccumulator:
    """Accumulates attention weights on ALL tokens across generation steps."""
    
    def __init__(self, model, prefix_len: int, max_seq_len: int, device: str, debug: bool = False):
        self.model = model
        self.prefix_len = prefix_len
        self.max_seq_len = max_seq_len
        self.device = device
        self.debug = debug
        self.accumulated_attn = None  # [max_seq_len] - attention received per position
        self._hooks = []
        self._step_count = 0
        self._current_seq_len = prefix_len
        
    def _attention_hook(self, layer_idx: int):
        """Creates a hook that accumulates attention on ALL positions."""
        def hook(module, input, kwargs, output):
            try:
                if len(output) > 1 and output[1] is not None:
                    attn_weights = output[1]
                    
                    if attn_weights is not None and attn_weights.numel() > 0:
                        kv_len = attn_weights.shape[-1]
                        
                        # Skip prefill (when this is the first forward pass)
                        if kv_len <= self.prefix_len:
                            return output
                        
                        # Initialize accumulator if needed
                        if self.accumulated_attn is None:
                            self.accumulated_attn = torch.zeros(
                                self.max_seq_len,
                                dtype=torch.float32, device='cpu'
                            )
                            if self.debug:
                                print(f"[DEBUG] Initialized accumulator for max_seq_len={self.max_seq_len}")
                        
                        # Get attention from the last query token to ALL previous positions
                        # attn_weights shape: [batch, num_heads, 1, kv_len]
                        # Sum across all heads for this layer
                        attn_to_all = attn_weights[0, :, -1, :].sum(dim=0)  # [kv_len]
                        
                        # Accumulate (only up to kv_len positions exist so far)
                        actual_len = min(kv_len, self.max_seq_len)
                        self.accumulated_attn[:actual_len] += attn_to_all[:actual_len].float().cpu()
                        
                        if layer_idx == 0:
                            self._step_count += 1
                            self._current_seq_len = kv_len
                            if self.debug and self._step_count % 200 == 0:
                                print(f"[DEBUG] Step {self._step_count}, seq_len={kv_len}")
            except Exception as e:
                if self.debug:
                    print(f"[DEBUG] Hook error at layer {layer_idx}: {e}")
            
            return output
        return hook
    
    def register_hooks(self):
        for layer_idx, layer in enumerate(self.model.model.layers):
            hook = layer.self_attn.register_forward_hook(
                self._attention_hook(layer_idx),
                with_kwargs=True
            )
            self._hooks.append(hook)
        print(f"Registered {len(self._hooks)} attention hooks")
    
    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
    
    def get_step_count(self):
        return self._step_count
    
    def get_prefix_attention(self):
        """Get summed attention on prefix tokens."""
        if self.accumulated_attn is None:
            return None
        return self.accumulated_attn[:self.prefix_len].numpy()
    
    def get_generated_attention(self, num_generated: int):
        """Get summed attention on generated tokens."""
        if self.accumulated_attn is None:
            return None
        return self.accumulated_attn[self.prefix_len:self.prefix_len + num_generated].numpy()
    
    def get_all_attention(self, total_len: int):
        """Get summed attention on all tokens."""
        if self.accumulated_attn is None:
            return None
        return self.accumulated_attn[:total_len].numpy()


def visualize_attention(
    dataset: str = "math500",
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    device: Optional[str] = None,
    press_name: str = "full",
    cache_budget: int = 1536,
    num_samples: int = 1,
    random_seed: int = 1,
    max_new_tokens: int = 32768,
    max_context_length: Optional[int] = None,
    save_dir: Optional[str] = None,
    debug: bool = False,
):
    """
    Run generation and create TEXT HEATMAP (HTML) showing attention on BOTH prefix and generated tokens.
    """
    
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Save to text_heatmaps directory
    if save_dir is None:
        save_dir = Path(__file__).parent.parent / "text_heatmaps"
    else:
        save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Load dataset
    assert dataset in DATASET_DICT, f"Dataset {dataset} not found"
    hf_name, data_dir, data_split = DATASET_DICT[dataset]
    ds = load_dataset(hf_name, data_dir=data_dir, split=data_split)
    
    if num_samples > 0:
        ds = ds.shuffle(seed=random_seed).select(range(min(num_samples, len(ds))))
    
    formatter = FORMATTER_DICT[dataset]
    
    print(f"Loading model {model_name} with eager attention...")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    
    for i, example in enumerate(tqdm(ds, desc="Processing samples")):
        input_text, gt_answer = formatter(example)
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(device)
        
        if max_context_length is not None:
            inputs = {k: v[:, :max_context_length] for k, v in inputs.items()}
        
        prefix_len = inputs["input_ids"].shape[1]
        max_seq_len = prefix_len + max_new_tokens
        
        print(f"\n{'='*60}")
        print(f"Sample {i}: Prefix length = {prefix_len} tokens")
        print(f"{'='*60}")
        
        accumulator = AttentionAccumulator(model, prefix_len, max_seq_len, device, debug=debug)
        accumulator.register_hooks()
        
        try:
            stopping = StoppingCriteriaList([
                StopOnBoxed(prompt_len=prefix_len, tokenizer=tokenizer)
            ])
            
            print(f"Generating up to {max_new_tokens} tokens...")
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    stopping_criteria=stopping,
                    use_cache=True,
                    output_attentions=True,
                    return_dict_in_generate=True,
                )
            
            generated_ids = outputs.sequences[0][prefix_len:]
            num_generated = len(generated_ids)
            steps_captured = accumulator.get_step_count()
            print(f"Generated {num_generated} tokens, captured {steps_captured} attention steps")
            
            # Get attention scores
            prefix_attn = accumulator.get_prefix_attention()
            gen_attn = accumulator.get_generated_attention(num_generated)
            
            if prefix_attn is None:
                print("Warning: No attention weights captured!")
                continue
            
            # Get tokens as text
            prefix_tokens = [tokenizer.decode([tid], skip_special_tokens=False) 
                           for tid in inputs["input_ids"][0].tolist()]
            gen_tokens = [tokenizer.decode([tid], skip_special_tokens=False) 
                         for tid in generated_ids.tolist()]
            
            # Create HTML heatmap
            html_content = create_html_heatmap(
                prefix_tokens=prefix_tokens,
                prefix_scores=prefix_attn.tolist(),
                gen_tokens=gen_tokens,
                gen_scores=gen_attn.tolist() if gen_attn is not None else [0] * num_generated,
                model_name=model_name,
                input_text=input_text,
            )
            
            # Save HTML
            model_short = model_name.replace("/", "_")
            html_path = save_dir / f"attn_full_q{i}_{model_short}_seed{random_seed}.html"
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"✅ Saved HTML heatmap to {html_path}")
            
            # Save raw data
            data_path = save_dir / f"attn_full_q{i}_{model_short}_seed{random_seed}.npz"
            np.savez_compressed(
                data_path,
                prefix_attention=prefix_attn,
                generated_attention=gen_attn if gen_attn is not None else np.zeros(num_generated),
                prefix_len=prefix_len,
                num_generated=num_generated,
                prefix_tokens=np.array(prefix_tokens, dtype=object),
                gen_tokens=np.array(gen_tokens, dtype=object),
            )
            print(f"✅ Saved data to {data_path}")
            
        finally:
            accumulator.remove_hooks()
    
    print(f"\n{'='*60}")
    print(f"All heatmaps saved to {save_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    cache_dir = "/fs/nexus-scratch/minghui/.cache/huggingface"
    if not os.environ.get("HF_HOME"):
        os.environ["HF_HOME"] = cache_dir
    Fire(visualize_attention)
