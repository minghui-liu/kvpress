# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
from dataclasses import dataclass

import torch
import numpy as np
from torch import nn

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
        # Initialize BasePress timing/tracking fields (e.g., measure_latency)
        super().__post_init__()
        assert 0 <= self.compression_ratio < 1, "Compression ratio must be between 0 and 1"
        
        # Initialize in-memory ranking data collection.
        self.ranking_data = []
        self.save_dir = "ranking_analysis"
        
        # Tokenizer for decoding tokens (will be set during inference)
        self.tokenizer = None
        self.input_tokens = None

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

    def set_tokenizer_and_tokens(self, tokenizer, input_tokens):
        """Set tokenizer and input tokens for text decoding."""
        self.tokenizer = tokenizer
        self.input_tokens = input_tokens

    def save_ranking_data(self, scores, indices, kv_len, is_prefill):
        """Collect ranking data in memory without writing auxiliary files."""
        # Only save if tokenizer is set (tracking enabled)
        if self.tokenizer is None:
            return
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
                
        except Exception as e:
            print(f"Error saving ranking data: {e}")
    
    def save_all_ranking_data(self, filename="all_ranking_data.json"):
        """Retained for compatibility; auxiliary ranking files are disabled."""
        return
    
    def reset_ranking_data(self):
        """Reset collected ranking data."""
        self.ranking_data = []

    def compress_prefilling(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        q_len = hidden_states.shape[1]
        layer_idx = getattr(module, "layer_idx", 0)
        if self.cache_budget <= 0:
            if layer_idx == 0:
                self.track_retained_cache_positions(q_len, list(range(q_len)))
            return keys, values

        if self.cache_budget >= q_len:
            if layer_idx == 0:
                self.track_retained_cache_positions(q_len, list(range(q_len)))
            return keys, values

        # Compute scores
        score_result = self.score(module, hidden_states, keys, values, attentions, True, kwargs)
        # Handle case where score returns tuple (scores, attn_weights) or just scores
        if isinstance(score_result, tuple):
            scores = score_result[0]
        else:
            scores = score_result
        # Get indices of KV pairs with the lowest scores
        indices = scores.topk(self.cache_budget, dim=-1).indices

        if layer_idx == 0:
            retained_positions = indices[0, 0].detach().cpu().tolist()
            self.track_retained_cache_positions(q_len, retained_positions)
        
        # Save ranking data
        #self.save_ranking_data(scores, indices, q_len, True)
        
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
        kv_len = keys.shape[2]
        if self.cache_budget == 0:
            if getattr(module, "layer_idx", 0) == 0:
                self.track_retained_cache_positions(kv_len, list(range(kv_len)))
            return keys, values

        if self.cache_budget >= kv_len:
            # All tokens retained, track if needed (only if tokenizer is set)
            if getattr(module, "layer_idx", 0) == 0 and self.tokenizer is not None:  # Only track at first layer to avoid duplicates
                all_token_ids = self.get_tracked_cache_token_ids(kv_len)
                retained_token_ids = all_token_ids.copy()
                self.track_generation_step(all_token_ids, retained_token_ids, self.tokenizer)
                self.track_retained_cache_positions(kv_len, list(range(kv_len)))
            return keys, values

        # Compute scores
        score_result = self.score(module, hidden_states, keys, values, attentions, False, kwargs)
        # Handle case where score returns tuple (scores, attn_weights) or just scores
        if isinstance(score_result, tuple):
            scores = score_result[0]
        else:
            scores = score_result
        # Get indices of KV pairs with the lowest scores
        indices = scores.topk(self.cache_budget, dim=-1).indices
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)
        
        # Track token retention/eviction at first layer only (only if tokenizer is set)
        if getattr(module, "layer_idx", 0) == 0 and self.tokenizer is not None:  # Only track at first layer to avoid duplicates
            all_token_ids = self.get_tracked_cache_token_ids(kv_len)
            retained_positions = indices[0, 0, :, 0].cpu().tolist()
            retained_token_ids = [all_token_ids[pos] for pos in retained_positions]
            self.track_generation_step(all_token_ids, retained_token_ids, self.tokenizer)
            self.track_retained_cache_positions(kv_len, retained_positions)

        # Save ranking data only if tokenizer is set (tracking enabled)
        if self.tokenizer is not None:
            self.save_ranking_data(scores, indices, kv_len, False)

        # Prune keys and values
        keys = keys.gather(2, indices).contiguous()
        values = values.gather(2, indices).contiguous()
        return keys, values
