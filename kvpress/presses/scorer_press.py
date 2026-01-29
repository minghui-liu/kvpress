# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
import json
import os
from dataclasses import dataclass

import torch
import numpy as np
from torch import nn
from transformers.models.llama.modeling_llama import repeat_kv, rotate_half

from kvpress.presses.base_press import BasePress

logger = logging.getLogger(__name__)


@dataclass
class ScorerPress(BasePress):
    """
    Default press method for using a score method.
    Any ScorerPress subclass must implement the `score` method that computes a tensor of scores for each key-value pair
    The KV pairs with the lowest scores will be pruned in the `compress` method.
    The cache is uniformly pruned across all heads and layers using the compression_ratio parameter.
    """

    compression_ratio: float = 0.0
    cache_budget: int = 0

    def __post_init__(self):
        # Initialize BasePress timing counters and local state
        super().__post_init__()
        assert 0 <= self.compression_ratio < 1, "Compression ratio must be between 0 and 1"
        # Track last evicted indices per (layer_idx, attn_head) to avoid duplicate CSV rows
        self._last_evicted_indices: dict[tuple[int, int], tuple[int, ...]] = {}
        
        # Initialize ranking data collection
        self.ranking_data = []
        self.save_dir = "ranking_analysis"

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
        """
        Compute a tensor of scores with shape (bsz, num_key_value_heads, q_len)
        The KV pairs with lowest scores will be pruned in the `compress` method.
        """
        raise NotImplementedError

    def save_ranking_data(self, scores, indices, kv_len, is_prefill):
        """Save ranking data for analysis."""
        # Only save if tokenizer is set (tracking enabled)
        if self.tokenizer is None:
            return
        # Create directory only when needed (first time saving)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
        try:
            # Convert tensors to numpy
            scores_np = scores.cpu().numpy().flatten()
            indices_np = indices.cpu().numpy().flatten()
            
            # Get rankings (higher score = higher rank)
            rankings = np.argsort(scores_np)[::-1]  # Sort in descending order
            
            # Get token text information if available
            token_texts = []
            top_10_tokens = []
            bottom_10_tokens = []
            
            if self.tokenizer is not None and self.input_tokens is not None:
                # Decode all tokens
                for i in range(min(kv_len, len(self.input_tokens))):
                    token_text = self.tokenizer.decode([self.input_tokens[i]], skip_special_tokens=True)
                    token_texts.append({
                        'index': i,
                        'text': token_text,
                        'score': scores_np[i] if i < len(scores_np) else 0.0
                    })
                
                # Get top 10 tokens (highest scores)
                top_10_indices = rankings[:10]
                for idx in top_10_indices:
                    if idx < len(self.input_tokens):
                        token_text = self.tokenizer.decode([self.input_tokens[idx]], skip_special_tokens=True)
                        top_10_tokens.append({
                            'index': int(idx),
                            'text': token_text,
                            'score': float(scores_np[idx])
                        })
                
                # Get bottom 10 tokens (lowest scores)
                bottom_10_indices = rankings[-10:]
                for idx in bottom_10_indices:
                    if idx < len(self.input_tokens):
                        token_text = self.tokenizer.decode([self.input_tokens[idx]], skip_special_tokens=True)
                        bottom_10_tokens.append({
                            'index': int(idx),
                            'text': token_text,
                            'score': float(scores_np[idx])
                        })
            
            # Create ranking entry
            ranking_entry = {
                'scores': scores_np.tolist(),
                'rankings': rankings.tolist(),
                'selected_indices': indices_np.tolist(),
                'sequence_length': kv_len,
                'cache_budget': self.cache_budget,
                'is_prefill': is_prefill,
                'compression_ratio': self.compression_ratio,
                'token_texts': token_texts,
                'top_10_tokens': top_10_tokens,
                'bottom_10_tokens': bottom_10_tokens
            }
            
            # Add to ranking data
            self.ranking_data.append(ranking_entry)
            
            # Save individual ranking data
            ranking_file = os.path.join(self.save_dir, f"ranking_data_{len(self.ranking_data)}.json")
            with open(ranking_file, 'w') as f:
                json.dump(ranking_entry, f, indent=2)
                
        except Exception as e:
            print(f"Error saving ranking data: {e}")
    
    def save_all_ranking_data(self, filename="all_ranking_data.json"):
        """Save all collected ranking data to a single file."""
        # Only save if tokenizer is set (tracking enabled)
        if self.tokenizer is None:
            return
        # Create directory only when needed
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
        try:
            output_file = os.path.join(self.save_dir, filename)
            with open(output_file, 'w') as f:
                json.dump(self.ranking_data, f, indent=2)
            print(f"Saved {len(self.ranking_data)} ranking entries to {output_file}")
        except Exception as e:
            print(f"Error saving all ranking data: {e}")
    
    def reset_ranking_data(self):
        """Reset collected ranking data."""
        self.ranking_data = []

    def compute_attention_loss(
        self,
        module: nn.Module,
        attentions: torch.Tensor | None,
        indices: torch.Tensor,
        window_size: int = 0,
    ) -> None:
        """
        Shared helper to log pre-prune vs kept attention mass per attention head using
        model-provided attentions. This does not change pruning; it's for CSV logging.

        Parameters
        ----------
        module : nn.Module
            Layer module to read head/group counts
        attentions : torch.Tensor | None
            Layer attentions [B, H_attn, Q, K]; if None, returns immediately
        indices : torch.Tensor
            Kept positions per KV head [B, H_kv, topk] or expanded [B, H_kv, topk, head_dim]
        window_size : int
            Number of last keys to treat as always-kept (exclude from prunable span)
        """
        if self.latency:
            return
        # Shapes
        b = 0
        H_attn = attentions.shape[1]
        Q = attentions.shape[2]
        K = attentions.shape[3]
        K_valid = K - window_size if window_size > 0 else K
        if K_valid <= 0:
            return

        # Canonicalize kept indices to [H_kv, topk]
        kept_all = indices[b]
        if kept_all.dim() == 2:
            # [H_kv, topk] (common case)
            pass
        elif kept_all.dim() == 3:
            # [H_kv, topk, something] -> take first column
            kept_all = kept_all[:, :, 0]

        H_kv = kept_all.shape[0]
        if H_kv == 0:
            return
        heads_per_group = max(1, H_attn // H_kv)

        # Build keep mask per KV head over prunable K (vectorized)
        device = attentions.device
        valid_mask = kept_all < K_valid
        safe_idx = kept_all.clone()
        safe_idx[~valid_mask] = 0
        keep_mask = torch.zeros(H_kv, K_valid, dtype=torch.bool, device=device)
        kvh_idx = torch.arange(H_kv, device=device).unsqueeze(1).expand_as(safe_idx)
        keep_mask[kvh_idx[valid_mask], safe_idx[valid_mask]] = True  # [H_kv, K_valid]
        drop_mask = ~keep_mask  # [H_kv, K_valid]

        # Map KV masks to attention heads by repeating per group
        keep_mask_attn = keep_mask.repeat_interleave(heads_per_group, dim=0)  # [H_attn, K_valid]
        # Guard in case H_attn is not a perfect multiple
        if keep_mask_attn.shape[0] > H_attn:
            keep_mask_attn = keep_mask_attn[:H_attn]

        # Slice attentions to query window and K_valid
        q_start = Q - window_size if window_size > 0 else 0
        attn = attentions[b, :, q_start:, :K_valid]  # [H_attn, Q', K_valid]

        # Vectorized sums per attention head
        pre_sum = attn.sum(dim=(1, 2), dtype=torch.float64)  # [H_attn]
        kept_mass = (attn * keep_mask_attn.unsqueeze(1)).sum(dim=(1, 2), dtype=torch.float64)  # [H_attn]
        kept_fraction = torch.where(pre_sum > 0, kept_mass / pre_sum, torch.zeros_like(pre_sum)).cpu().tolist()

        # Precompute per-KV-head metadata for CSV
        kept_counts = valid_mask.sum(dim=1).tolist()  # per KV head
        diff_indices_counts = drop_mask.sum(dim=1).tolist()  # per KV head
        # Evicted positions strings per KV head
        try:
            drop_idx_all = [torch.nonzero(drop_mask[kvh], as_tuple=False).squeeze(-1).tolist() for kvh in range(H_kv)]
        except Exception:
            drop_idx_all = [[] for _ in range(H_kv)]
        evicted_str_all = [" ".join(str(int(x)) for x in lst) if lst else "" for lst in drop_idx_all]

        # Write rows per attention head (cheap loop; heavy math already done)
        layer_idx = int(getattr(module, "layer_idx", -1))
        for h in range(H_attn):
            kvh = min(h // heads_per_group, H_kv - 1)
            # Duplicate suppression by (layer, head)
            key = (layer_idx, int(h))
            current_tuple = tuple(int(x) for x in drop_idx_all[kvh]) if drop_idx_all[kvh] else tuple()
            if self._last_evicted_indices.get(key) == current_tuple:
                continue
            self._last_evicted_indices[key] = current_tuple

            self.prune_step = getattr(self, "prune_step", 0) + 1
            self.write_data(
                csv_path=getattr(self, "csv_path", ""),
                prune_step=self.prune_step,
                layer_idx=layer_idx,
                head_idx=int(h),
                kv_len_pre=int(K_valid),
                attn_len=int(kept_counts[kvh]),
                diff_indices=int(diff_indices_counts[kvh]),
                attn_pre=1.0,
                attn_post=float(kept_fraction[h]),
                evicted_positions=evicted_str_all[kvh],
            )

    def compress_prefilling(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if self.cache_budget <= 0:
            return keys, values

        q_len = hidden_states.shape[1]
        if self.cache_budget >= q_len:
            return keys, values

        # Compute scores
        scores = self.score(module, hidden_states, keys, values, attentions, True, kwargs)
        # Get indices of KV pairs with the lowest scores
        indices = scores.topk(self.cache_budget, dim=-1).indices
        # Centralized logging: write CSV using (possibly computed) attentions
        if not getattr(self, "latency", False):
            self.compute_attention_loss(module, attentions, indices, window_size=getattr(self, "window_size", 0))
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)

        # Prune keys and values
        keys = keys.gather(2, indices).contiguous()
        values = values.gather(2, indices).contiguous()
        return keys, values
    

    def compress_decoding(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
 
        if self.cache_budget == 0:
            return keys, values

        kv_len = keys.shape[2]
        layer_idx = getattr(module, "layer_idx", 0)
        
        if self.cache_budget >= kv_len:
            # All tokens retained - no compression needed, skip tracking to avoid excessive entries
            return keys, values

        # Compute scores
        scores = self.score(module, hidden_states, keys, values, attentions, False, kwargs)
        # Get indices of KV pairs with the lowest scores
        indices = scores.topk(self.cache_budget, dim=-1).indices
        # Centralized logging: write CSV using (possibly computed) attentions
        if not getattr(self, "latency", False):
            self.compute_attention_loss(module, attentions, indices, window_size=getattr(self, "window_size", 0))
        
        # Track token retention/eviction at first layer only (only if tokenizer is set)
        if layer_idx == 0 and self.tokenizer is not None:
            # Map position indices to actual token IDs
            if kv_len <= len(self.input_tokens):
                all_token_ids = self.input_tokens[:kv_len].cpu().tolist()
                retained_positions = indices[0, 0, :].cpu().tolist()  # Get retained position indices
                retained_token_ids = [all_token_ids[pos] for pos in retained_positions if pos < len(all_token_ids)]
            else:
                # If kv_len > input_tokens, we have generated tokens (use position indices as placeholders)
                all_token_ids = self.input_tokens.cpu().tolist() + list(range(len(self.input_tokens), kv_len))
                retained_positions = indices[0, 0, :].cpu().tolist()
                retained_token_ids = [all_token_ids[pos] if pos < len(all_token_ids) else pos for pos in retained_positions]
            self.track_generation_step(all_token_ids, retained_token_ids, self.tokenizer)
        
        # Save ranking data only if tokenizer is set (tracking enabled)
        if self.tokenizer is not None:
            self.save_ranking_data(scores, indices, kv_len, False)

        indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)

        # Prune keys and values
        keys = keys.gather(2, indices).contiguous()
        values = values.gather(2, indices).contiguous()
        return keys, values
