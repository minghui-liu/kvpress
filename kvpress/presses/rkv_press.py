# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import math
import json
import os
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.llama.modeling_llama import repeat_kv, rotate_half

from kvpress.presses.scorer_press import ScorerPress


@dataclass
class RKVPress(ScorerPress):
    """
    RKV (https://www.arxiv.org/pdf/2505.24133)
    """

    cache_budget: int = 0
    compress_interval: int = 64 # aka. the buffer size 
    # compression_ratio: float = 0.0
    window_size: int = 8 # number of observation tokens always kept in the cache
    kernel_size: int = 5
    attn_csv_path: str = "attn_loss.csv"
    prune_step: int = 0

    def __post_init__(self):
        super().__post_init__()
        self.accumulated_tokens = 0  # Initialize accumulated tokens for compression interval
        self.hidden_size = None  # Will be set based on model type
        self.acc_hidden_states = None  # Will be initialized when hidden_size is known

        # Tokenizer for decoding tokens (will be set during inference)
        self.tokenizer = None
        self.input_tokens = None

        # Qualitative analysis mode (same as RKVLSHPress)
        self.enable_qualitative_analysis = False
        self.qualitative_data = []
        self.current_sample_id = 0
        self.current_sample_data = []
        self.qualitative_output_file = None
        self.save_dir = "ranking_analysis" 

    @staticmethod
    def compute_window_attention(module, hidden_states, keys, window_size, position_embeddings):
        """
        Compute the last window_size queries and associated attention weights for the first q_len - window_size keys.
        """
        bsz, q_len, _ = hidden_states.shape
        num_heads = module.config.num_attention_heads
        head_dim = module.head_dim
        num_key_value_groups = num_heads // module.config.num_key_value_heads

        # Get last window_size queries
        if hasattr(module, "q_proj"):
            query_states = module.q_proj(hidden_states[:, -window_size:])
        elif hasattr(module, "qkv_proj"):
            qkv = module.qkv_proj(hidden_states[:, -window_size:])
            query_states = qkv[..., : num_heads * head_dim]
        else:
            raise NotImplementedError(f"SnapKV not yet implemented for {module.__class__}.")
        
        query_states = query_states.view(bsz, window_size, num_heads, head_dim).transpose(1, 2)

        # Apply RoPE
        cos, sin = position_embeddings
        cos, sin = cos[:, -window_size:], sin[:, -window_size:]
        query_states = (query_states * cos.unsqueeze(1)) + (rotate_half(query_states) * sin.unsqueeze(1))

        # Compute attention for first q_len - window_size tokens
        key_states = repeat_kv(keys, num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
        attention_mask = torch.ones_like(attn_weights) * float("-1e9")
        attention_mask = torch.triu(attention_mask, diagonal=q_len - window_size + 1)
        attn_weights += attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.bfloat16).to(query_states.dtype)
        attn_weights = attn_weights[..., :-window_size]

        return attn_weights


    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        is_prefill: bool,
        kwargs,
    ) -> torch.Tensor:

        bsz, num_key_value_heads, q_len, _ = keys.shape
        num_key_value_groups = module.config.num_attention_heads // num_key_value_heads

        assert q_len > self.window_size, "Query length should be greater than the window size"
        if attentions is not None:
            attn_weights = attentions[..., -self.window_size :, : -self.window_size]
        else:
            attn_weights = self.compute_window_attention(
                module, hidden_states, keys, self.window_size, kwargs["position_embeddings"]
            )
        scores = attn_weights.mean(dim=-2)   
        # Average per group (https://github.com/FasterDecoding/SnapKV/issues/22)
        scores = scores.view(bsz, num_key_value_heads, num_key_value_groups, q_len - self.window_size)
        scores = scores.max(dim=-2).values
        # Stablization and Importance Estimation
        scores = F.max_pool1d(scores, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=1)
        # Redundancy Estimation via Semantic Similarity
        
        # normalize keys by dividing the l2 norm of keys + eps (1e-8) 
        eps = 1e-8
        keys_norm = keys.norm(dim=-1, keepdim=True) + eps
        keys = keys / keys_norm

        # Original Algorithm: directly using the cosine similarity
        # compute the cosine similarity between keys
        keys_flat = keys.view(bsz, num_key_value_heads, -1, keys.shape[-1])
        keys_flat = keys_flat[:, :, : -self.window_size, :]  # Exclude the last window_size keys
        keys_similarity = torch.einsum("bhqd,bhkd->bhqk", keys_flat, keys_flat)
        # zero out the diagonal (self-similarity)
        mask = torch.eye(keys_similarity.shape[-1], device=keys_similarity.device).unsqueeze(0).unsqueeze(0)
        keys_similarity = keys_similarity * (1 - mask)

        redundency = keys_similarity.mean(dim=-1)  # Average over the key dimension
        redundency = F.softmax(redundency, dim=-1, dtype=torch.float32).to(scores.dtype)

        scores = scores + redundency
        # Add back the observation window. Use max score to make sure the window is not pruned.
        scores = F.pad(scores, (0, self.window_size), value=scores.max().item())
        return scores

    @staticmethod
    def get_avg_attention_for_index(indices, attn_weights, b=0, k=0):
        idx = indices[b, :, :, 0]
        attn_weights = attn_weights.to(torch.float32)
        max_valid_key = attn_weights.shape[-1]
        total = torch.zeros((), device=attn_weights.device, dtype=torch.float32)
        for kvh in range(idx.shape[0]):
            key_idx = int(idx[kvh, k].item())
            if key_idx >= max_valid_key:
                continue
            total = total + attn_weights[b, :, :, key_idx].sum()
        return total

    @staticmethod
    def get_avg_attention_for_all_indices(indices, attn_weights, b=0):
        attn_weights = attn_weights.to(torch.float32)
        total = torch.zeros((), device=attn_weights.device, dtype=torch.float32)
        num_positions = indices.shape[2]
        for k in range(num_positions):
            total = total + RKVPress.get_avg_attention_for_index(indices, attn_weights, b=b, k=k)
        return total
    

    def _get_hidden_size(self, module, device="cuda"):
        """Get hidden size based on model type."""
        if self.hidden_size is None:
            # Detect model type from config
            model_type = getattr(module.config, 'model_type', '').lower()
            model_name = getattr(module.config, 'name_or_path', '').lower()
            
            # Check for llama3 models
            if 'llama' in model_type or 'llama' in model_name or 'nemotron' in model_name:
                self.hidden_size = 4096
            # Check for qwen-7b models
            elif 'qwen' in model_type or 'qwen' in model_name:
                if '7b' in model_name or '7b' in str(getattr(module.config, 'hidden_size', 0)):
                    self.hidden_size = 3584
                else:
                    # Default for other Qwen models
                    self.hidden_size = getattr(module.config, 'hidden_size', 4096)
            else:
                # Default: use config hidden_size or fallback to 4096
                self.hidden_size = getattr(module.config, 'hidden_size', 4096)
        
        # Always ensure acc_hidden_states exists and is on correct device
        if self.acc_hidden_states is None or self.acc_hidden_states.device != device:
            self.acc_hidden_states = torch.zeros(
                (1, self.compress_interval, self.hidden_size), dtype=torch.bfloat16, device=device
            )
        
        return self.hidden_size

    def compress_decoding(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kv_len = keys.shape[2]
        if self.cache_budget == 0:
            if getattr(module, "layer_idx", 0) == 0:
                self.track_retained_cache_positions(kv_len, list(range(kv_len)))
            return keys, values
        layer_idx = getattr(module, "layer_idx", 0)
        if self.cache_budget >= kv_len:
            # All tokens retained, track if needed
            if layer_idx == 0:
                if self.tokenizer is not None and self.input_tokens is not None and kv_len <= len(self.input_tokens):
                    all_token_ids = self.input_tokens[:kv_len].cpu().tolist()
                    retained_token_ids = all_token_ids.copy()
                    self.track_generation_step(all_token_ids, retained_token_ids, self.tokenizer)
                elif self.tokenizer is not None and self.input_tokens is not None:
                    all_token_ids = self.input_tokens.cpu().tolist() + list(range(len(self.input_tokens), kv_len))
                    retained_token_ids = all_token_ids.copy()
                    self.track_generation_step(all_token_ids, retained_token_ids, self.tokenizer)
                self.track_retained_cache_positions(kv_len, list(range(kv_len)))
            return keys, values
        
        # Initialize hidden size if not set (must be done before using acc_hidden_states)
        device = hidden_states.device
        self._get_hidden_size(module, device=device)
        
        if self.accumulated_tokens < self.compress_interval:
            if getattr(module, "layer_idx", -1) == 0:
                self.accumulated_tokens += 1
            # # print(f"[DEBUG] hidden_states shape: {hidden_states.shape}, acc_hidden_states shape: {self.acc_hidden_states.shape}, accumulated_tokens: {self.accumulated_tokens}")
            self.acc_hidden_states[:, self.accumulated_tokens - 1, :] = hidden_states
            return keys, values

        # Compute scores
        # scores = self.score(module, hidden_states, keys, values, attentions, False, kwargs)
        scores = self.score(module, self.acc_hidden_states[:, -self.window_size:, :], keys, values, attentions, False, kwargs)
        # Get indices of KV pairs with the lowest scores
        indices = scores.topk(self.cache_budget, dim=-1).indices
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)

        # Track token retention/eviction at first layer only
        if layer_idx == 0:
            # Map position indices to actual token IDs
            if self.tokenizer is not None and self.input_tokens is not None:
                all_token_ids = self.get_tracked_cache_token_ids(kv_len)
                retained_positions = indices[0, 0, :, 0].cpu().tolist()
                retained_token_ids = [all_token_ids[pos] for pos in retained_positions]
                self.track_generation_step(all_token_ids, retained_token_ids, self.tokenizer)

            # Log qualitative decisions if enabled
            self.log_qualitative_decisions(indices, kv_len, scores)
            self.track_retained_cache_positions(
                kv_len,
                indices[0, 0, :, 0].detach().cpu().tolist(),
            )

        # Prune keys and values
        keys = keys.gather(2, indices).contiguous()
        values = values.gather(2, indices).contiguous()
        # remove nan in keys and values
        keys = torch.nan_to_num(keys, nan=0.0)  
        values = torch.nan_to_num(values, nan=0.0)
        
        if layer_idx == 0:
            self.accumulated_tokens = 0  # Reset after compression
            device = hidden_states.device
            self.acc_hidden_states = torch.zeros(
                (1, self.compress_interval, self.hidden_size), dtype=torch.bfloat16, device=device
            ) # Reset accumulated hidden states

        # Save ranking data ONLY if tokenizer is set (track_tokens == True)
        if self.tokenizer is not None:
            self.save_ranking_data(scores, indices, kv_len, False)

        return keys, values

    def enable_qualitative_mode(self, output_file=None, model_name=None, press_name=None):
        """Enable qualitative analysis mode to track token retention decisions."""
        self.enable_qualitative_analysis = True

        # Set up output file path
        if output_file is None:
            p = press_name or "rkv"
            m = f"_{model_name}" if model_name else ""
            output_file = f"token_decisions_{p}{m}_budget{self.cache_budget}.jsonl"

        self.qualitative_output_file = os.path.join(self.save_dir, output_file)
        os.makedirs(self.save_dir, exist_ok=True)

        # Clear the file if it exists (start fresh)
        if os.path.exists(self.qualitative_output_file):
            os.remove(self.qualitative_output_file)

        print(f"[RKV] Qualitative analysis mode enabled")
        print(f"[RKV] Incremental output will be saved to: {self.qualitative_output_file}")

    def save_current_sample_incremental(self):
        """Save the current sample's qualitative data incrementally to file."""
        if not self.enable_qualitative_analysis:
            return

        if len(self.current_sample_data) == 0:
            return

        # Create sample entry
        sample_entry = {
            'sample_id': self.current_sample_id,
            'num_eviction_steps': len(self.current_sample_data),
            'eviction_steps': self.current_sample_data
        }

        # Append to JSONL file (one JSON object per line)
        with open(self.qualitative_output_file, 'a') as f:
            f.write(json.dumps(sample_entry) + '\n')

        print(f"[RKV] Sample {self.current_sample_id} qualitative data saved ({len(self.current_sample_data)} eviction steps)")

        # Keep in main list for summary generation later
        self.qualitative_data.extend(self.current_sample_data)

        # Clear current sample data to free memory
        self.current_sample_data = []

    def next_sample(self):
        """Mark the start of a new sample for qualitative analysis."""
        # Save current sample data before moving to next
        if self.enable_qualitative_analysis:
            self.save_current_sample_incremental()

        self.current_sample_id += 1

    def set_tokenizer_and_tokens(self, tokenizer, input_tokens):
        """Set tokenizer and input tokens for text decoding."""
        self.tokenizer = tokenizer
        self.input_tokens = input_tokens

    def log_qualitative_decisions(self, indices, kv_len, scores):
        """
        Log detailed token retention/eviction decisions.

        For each eviction step, logs:
        - All tokens in the cache (position, text, token_id)
        - Which tokens were retained vs evicted
        - Attention + redundancy scores

        This allows analysis of what kinds of tokens RKV drops.
        """
        if not self.enable_qualitative_analysis or self.tokenizer is None:
            return

        # Get all token IDs in the current cache
        all_token_ids = self.get_tracked_cache_token_ids(kv_len)

        # Get which positions were retained (kept in cache)
        retained_positions = set(indices[0, 0, :, 0].cpu().tolist())
        all_positions = list(range(kv_len))

        # Get scores (excluding window which is always kept)
        # scores from score() is padded to kv_len; slice off window padding
        final_scores = scores[0, 0, :-self.window_size].detach().cpu().float().numpy()
        num_scored = len(final_scores)  # kv_len - window_size

        # Build detailed token list (JSON only stores IDs, not text to keep file small)
        all_tokens = []
        retained_tokens = []
        evicted_tokens = []

        # Track repetitive/filler tokens of interest
        repetitive_keywords = {'wait', 'so', 'but'}
        repetitive_tokens_all = []
        repetitive_tokens_retained = []
        repetitive_tokens_evicted = []

        for pos in all_positions:
            token_id = all_token_ids[pos] if pos < len(all_token_ids) else None

            # Decode token to check if it's a repetitive keyword
            if token_id is not None:
                token_text = self.tokenizer.decode([token_id], skip_special_tokens=False).strip().lower()
                is_repetitive = token_text in repetitive_keywords
            else:
                is_repetitive = False

            # Window positions (pos >= num_scored) are always retained and have no scores
            in_window = pos >= num_scored
            token_info = {
                'position': pos,
                'token_id': token_id,
                'retained': pos in retained_positions,
                'in_window': in_window,
                'final_score': float(final_scores[pos]) if not in_window else None,
                'is_repetitive_keyword': is_repetitive,  # Flag for wait/so/but
            }

            all_tokens.append(token_info)
            if is_repetitive:
                repetitive_tokens_all.append(token_info)

            if pos in retained_positions:
                retained_tokens.append(token_info)
                if is_repetitive:
                    repetitive_tokens_retained.append(token_info)
            else:
                evicted_tokens.append(token_info)
                if is_repetitive:
                    repetitive_tokens_evicted.append(token_info)

        # Calculate repetitive keyword densities
        num_total = len(all_positions)
        repetitive_density_all = len(repetitive_tokens_all) / num_total if num_total > 0 else 0
        repetitive_density_retained = len(repetitive_tokens_retained) / len(retained_tokens) if len(retained_tokens) > 0 else 0
        repetitive_density_evicted = len(repetitive_tokens_evicted) / len(evicted_tokens) if len(evicted_tokens) > 0 else 0

        # Create log entry for this eviction step
        log_entry = {
            'sample_id': self.current_sample_id,
            'eviction_step': len(self.current_sample_data),  # Track which eviction step within this sample
            'kv_len': kv_len,
            'cache_budget': self.cache_budget,
            'method': 'RKV',

            # All tokens with their status
            'all_tokens': all_tokens,

            # Separated lists for easier analysis
            'retained_tokens': retained_tokens,
            'evicted_tokens': evicted_tokens,

            # Summary statistics
            'num_total': num_total,
            'num_retained': len(retained_tokens),
            'num_evicted': len(evicted_tokens),

            # Repetitive keyword tracking (wait/so/but)
            'repetitive_keywords_tracked': list(repetitive_keywords),
            'num_repetitive_all': len(repetitive_tokens_all),
            'num_repetitive_retained': len(repetitive_tokens_retained),
            'num_repetitive_evicted': len(repetitive_tokens_evicted),
            'repetitive_density_all': repetitive_density_all,
            'repetitive_density_retained': repetitive_density_retained,
            'repetitive_density_evicted': repetitive_density_evicted,
        }

        # Append to current sample data (will be saved incrementally)
        self.current_sample_data.append(log_entry)

        # Print summary if verbose (only for first sample and first few steps)
        if self.current_sample_id == 0 and len(self.current_sample_data) <= 3:  # Only print first few
            print(f"[Qualitative] Sample {self.current_sample_id}, Step {log_entry['eviction_step']}: "
                  f"Retained {len(retained_tokens)}/{kv_len} tokens")

    def save_qualitative_analysis(self):
        """
        Generate final summary file after incremental saves are complete.

        The detailed data has already been saved incrementally to JSONL file.
        This method generates a human-readable summary.
        """
        if not self.enable_qualitative_analysis:
            print("[RKV] Qualitative analysis not enabled, skipping save")
            return

        if len(self.qualitative_data) == 0:
            print("[RKV] No qualitative data collected")
            return

        print(f"[RKV] Qualitative analysis complete:")
        print(f"  - Total samples: {self.current_sample_id + 1}")
        print(f"  - Total eviction steps: {len(self.qualitative_data)}")
        print(f"  - Incremental data saved to: {self.qualitative_output_file}")

        # Generate human-readable summary
        summary_path = self.qualitative_output_file.replace('.jsonl', '_summary.txt')
        self._save_qualitative_summary(summary_path)
        print(f"  - Summary saved to: {summary_path}")

    def _save_qualitative_summary(self, summary_file):
        """
        Generate and save a human-readable summary.

        Shows examples of retained vs evicted tokens to help identify:
        - What kinds of tokens RKV keeps
        - What kinds of tokens RKV drops

        Decodes token IDs to text for readability.
        """
        if self.tokenizer is None:
            print("[RKV] Cannot generate summary without tokenizer")
            return

        with open(summary_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"Token Retention/Eviction Analysis: RKV\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Method: RKV (Attention + Redundancy via Cosine Similarity)\n")
            f.write(f"Cache Budget: {self.cache_budget}\n")
            f.write(f"Total Eviction Steps: {len(self.qualitative_data)}\n\n")

            # Show first few eviction steps in detail
            for i, entry in enumerate(self.qualitative_data[:5], 1):
                f.write("\n" + "=" * 80 + "\n")
                f.write(f"Eviction Step {i}: Sample {entry['sample_id']}, Step {entry['eviction_step']}\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Total tokens: {entry['num_total']}\n")
                f.write(f"Retained: {entry['num_retained']}\n")
                f.write(f"Evicted: {entry['num_evicted']}\n\n")

                # Show repetitive keyword statistics
                f.write(f"Repetitive Keywords Tracked: {entry.get('repetitive_keywords_tracked', [])}\n")
                f.write(f"Repetitive tokens in all: {entry.get('num_repetitive_all', 0)} "
                       f"({entry.get('repetitive_density_all', 0):.2%})\n")
                f.write(f"Repetitive tokens retained: {entry.get('num_repetitive_retained', 0)} "
                       f"({entry.get('repetitive_density_retained', 0):.2%})\n")
                f.write(f"Repetitive tokens evicted: {entry.get('num_repetitive_evicted', 0)} "
                       f"({entry.get('repetitive_density_evicted', 0):.2%})\n\n")

                # Sort retained by score (highest first)
                retained = sorted(entry['retained_tokens'], key=lambda x: x['final_score'], reverse=True)
                # Sort evicted by score (lowest first)
                evicted = sorted(entry['evicted_tokens'], key=lambda x: x['final_score'])

                # Show examples of RETAINED tokens
                f.write("-" * 80 + "\n")
                f.write("RETAINED TOKENS (Top 15 by score):\n")
                f.write("-" * 80 + "\n")
                for token in retained[:15]:
                    # Decode token ID to text
                    token_text = self.tokenizer.decode([token['token_id']], skip_special_tokens=False) if token['token_id'] is not None else "<UNK>"
                    f.write(f"[Pos {token['position']:4d}] ID:{token['token_id']:6d} '{token_text[:30]:<30}' | Score: {token['final_score']:.4f}\n")

                # Show examples of EVICTED tokens
                f.write("\n" + "-" * 80 + "\n")
                f.write("EVICTED TOKENS (Top 15 by score - these had lowest scores):\n")
                f.write("-" * 80 + "\n")
                for token in evicted[:15]:
                    # Decode token ID to text
                    token_text = self.tokenizer.decode([token['token_id']], skip_special_tokens=False) if token['token_id'] is not None else "<UNK>"
                    f.write(f"[Pos {token['position']:4d}] ID:{token['token_id']:6d} '{token_text[:30]:<30}' | Score: {token['final_score']:.4f}\n")

            # Overall statistics
            f.write("\n\n" + "=" * 80 + "\n")
            f.write("Overall Statistics\n")
            f.write("=" * 80 + "\n")
            total_retained = sum(entry['num_retained'] for entry in self.qualitative_data)
            total_evicted = sum(entry['num_evicted'] for entry in self.qualitative_data)
            total_repetitive_all = sum(entry.get('num_repetitive_all', 0) for entry in self.qualitative_data)
            total_repetitive_retained = sum(entry.get('num_repetitive_retained', 0) for entry in self.qualitative_data)
            total_repetitive_evicted = sum(entry.get('num_repetitive_evicted', 0) for entry in self.qualitative_data)

            f.write(f"Total tokens retained: {total_retained}\n")
            f.write(f"Total tokens evicted: {total_evicted}\n\n")

            f.write(f"Repetitive keyword (wait/so/but) statistics:\n")
            f.write(f"  Total repetitive tokens: {total_repetitive_all}\n")
            f.write(f"  Repetitive tokens retained: {total_repetitive_retained} "
                   f"({total_repetitive_retained/total_retained:.2%} of retained)\n")
            f.write(f"  Repetitive tokens evicted: {total_repetitive_evicted} "
                   f"({total_repetitive_evicted/total_evicted:.2%} of evicted)\n")
            f.write(f"\nThis shows how effectively RKV identifies and drops repetitive filler words.\n")

        print(f"[RKV] Summary saved to: {summary_file}")
