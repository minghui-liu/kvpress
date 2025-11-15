# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
import json
import os
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
        assert 0 <= self.compression_ratio < 1, "Compression ratio must be between 0 and 1"
        
        # Initialize ranking data collection
        self.ranking_data = []
        self.save_dir = "ranking_analysis"
        os.makedirs(self.save_dir, exist_ok=True)
        
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
        """Save ranking data for analysis."""
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
 
        if self.cache_budget == 0:
            return keys, values

        kv_len = keys.shape[2]
        if self.cache_budget >= kv_len:
            # All tokens retained, track if needed
            if getattr(module, "layer_idx", 0) == 0:  # Only track at first layer to avoid duplicates
                # Map position indices to actual token IDs
                if kv_len <= len(self.input_tokens):
                    all_token_ids = self.input_tokens[:kv_len].cpu().tolist()
                    retained_token_ids = all_token_ids.copy()
                else:
                    # If kv_len > input_tokens, we have generated tokens
                    all_token_ids = self.input_tokens.cpu().tolist() + list(range(len(self.input_tokens), kv_len))
                    retained_token_ids = all_token_ids.copy()
                self.track_generation_step(all_token_ids, retained_token_ids, self.tokenizer)
            return keys, values

        # Compute scores
        scores = self.score(module, hidden_states, keys, values, attentions, False, kwargs)
        # Get indices of KV pairs with the lowest scores
        indices = scores.topk(self.cache_budget, dim=-1).indices
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)
        
        # Track token retention/eviction at first layer only
        if getattr(module, "layer_idx", 0) == 0:  # Only track at first layer to avoid duplicates
            # Map position indices to actual token IDs
            if kv_len <= len(self.input_tokens):
                all_token_ids = self.input_tokens[:kv_len].cpu().tolist()
                retained_positions = indices[0, 0, :, 0].cpu().tolist()  # Get retained position indices
                retained_token_ids = [all_token_ids[pos] for pos in retained_positions]
            else:
                # If kv_len > input_tokens, we have generated tokens (use position indices as placeholders)
                all_token_ids = self.input_tokens.cpu().tolist() + list(range(len(self.input_tokens), kv_len))
                retained_positions = indices[0, 0, :, 0].cpu().tolist()
                retained_token_ids = [all_token_ids[pos] if pos < len(self.input_tokens) else pos for pos in retained_positions]
            self.track_generation_step(all_token_ids, retained_token_ids, self.tokenizer)

        # Save ranking data
        self.save_ranking_data(scores, indices, kv_len, False)

        # Prune keys and values
        keys = keys.gather(2, indices).contiguous()
        values = values.gather(2, indices).contiguous()
        return keys, values
