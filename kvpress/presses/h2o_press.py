# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn

from kvpress.presses.scorer_press import ScorerPress

logger = logging.getLogger(__name__)


@dataclass
class H2OPress(ScorerPress):
    """
    The H2O score is defined as the average attention weight over all prompt tokens.
    The cache budget is divided between heavy hitters and an always-retained
    window of the most recent tokens, as in the original H2O policy.
    Requires output_attentions=True and attn_implementation="eager" to have access to attentions
    This approach is a faithful implementation of H2O (https://arxiv.org/abs/2306.14048).
    """

    cache_budget: int = 0
    window_size: int = 64
    output_attentions: bool = True
    attn_csv_path: str = "attn_loss.csv"
    prune_step: int = 0

    def __post_init__(self):
        if self.window_size < 0:
            raise ValueError("window_size must be non-negative")
        if not self.output_attentions:
            logger.warning(
                "Model will not return attentions in its output to save memory. "
                "Set output_attentions=True if attentions are needed in the output."
            )
        super().__post_init__()
        self.acc_attn_by_layer = {}
        self.n_tokens_in_sum_by_layer = {}

    def reset_timing(self):
        super().reset_timing()
        self.acc_attn_by_layer = {}
        self.n_tokens_in_sum_by_layer = {}

    def _layer_idx(self, module: nn.Module) -> int:
        return getattr(module, "layer_idx", 0)

    def _layer_state_key(self, module: nn.Module) -> int:
        layer_idx = getattr(module, "layer_idx", None)
        return layer_idx if layer_idx is not None else id(module)

    def _get_layer_state(self, module: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
        layer_key = self._layer_state_key(module)
        return self.acc_attn_by_layer[layer_key], self.n_tokens_in_sum_by_layer[layer_key]

    def _set_layer_state(
        self,
        module: nn.Module,
        acc_attn: torch.Tensor,
        n_tokens_in_sum: torch.Tensor,
    ):
        layer_key = self._layer_state_key(module)
        self.acc_attn_by_layer[layer_key] = acc_attn
        self.n_tokens_in_sum_by_layer[layer_key] = n_tokens_in_sum

    def _select_heavy_and_recent(self, scores: torch.Tensor) -> torch.Tensor:
        """Select heavy hitters from the past plus a fixed recent window."""
        kv_len = scores.shape[-1]
        keep_count = min(self.cache_budget, kv_len)
        recent_count = min(self.window_size, keep_count)
        heavy_count = keep_count - recent_count

        selected = []
        if heavy_count:
            past_end = kv_len - recent_count
            heavy_indices = scores[..., :past_end].topk(heavy_count, dim=-1).indices
            selected.append(heavy_indices)
        if recent_count:
            recent_indices = torch.arange(
                kv_len - recent_count,
                kv_len,
                device=scores.device,
                dtype=torch.long,
            )
            recent_indices = recent_indices.view(1, 1, -1).expand(*scores.shape[:-1], -1)
            selected.append(recent_indices)

        return torch.cat(selected, dim=-1)

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
        assert attentions is not None, 'Set output_attentions=True and attn_implementation="eager" to use this hook'
        bsz, num_key_value_heads, n_tokens, _ = keys.shape
        acc_attn, n_tokens_in_sum = self._get_layer_state(module)
        scores = acc_attn / n_tokens_in_sum
        scores = scores.view(bsz, num_key_value_heads, -1, n_tokens).mean(2)
        return scores
    

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

        # save the accumulated attention weights
        assert attentions is not None, 'Set output_attentions=True and attn_implementation="eager" to use this hook'
        bsz, n_heads, _, q_len = attentions.shape
        n_kv_groups = module.num_key_value_groups
        n_kv_heads = n_heads // n_kv_groups

        acc_attn = attentions.sum(2)
        # reshape attentions to bsz, n_kv_heads, n_kv_groups, q_len
        acc_attn = acc_attn.view(bsz, -1, n_kv_groups, q_len)
        # average over the n_kv_groups dimension
        acc_attn = acc_attn.mean(2) # bsz, n_kv_heads, q_len

        n_tokens_in_sum = torch.arange(q_len, 0, -1).to(attentions.device, attentions.dtype)
        n_tokens_in_sum = n_tokens_in_sum.unsqueeze(0).unsqueeze(0).expand(bsz, n_kv_heads, -1) # bsz, n_kv_heads, q_len
        self._set_layer_state(module, acc_attn, n_tokens_in_sum)

        if self.cache_budget >= q_len:
            return keys, values
  
        # Compute scores
        scores = self.score(module, hidden_states, keys, values, attentions, True, kwargs)
        indices = self._select_heavy_and_recent(scores)
        if getattr(module, "layer_idx", 0) == 0:
            self.track_retained_cache_positions(q_len, indices[0, 0].detach().cpu().tolist())

        # Prune keys and values
        kv_indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim) # bsz, num_key_value_heads, cache_budget, head_dim
        keys = keys.gather(2, kv_indices).contiguous()
        values = values.gather(2, kv_indices).contiguous()

        # Prune acc attention weights and n_tokens_in_sum
        # expand second dimension from n_kv_heads to num_heads
        acc_attn, n_tokens_in_sum = self._get_layer_state(module)
        acc_attn = acc_attn.gather(2, indices).contiguous()
        n_tokens_in_sum = n_tokens_in_sum.gather(2, indices).contiguous()
        self._set_layer_state(module, acc_attn, n_tokens_in_sum)

        return keys, values


    def compress_decoding(self, module, hidden_states, keys, values, attentions, kwargs):
        kv_len = keys.shape[2]
        layer_idx = getattr(module, "layer_idx", 0)
        if self.cache_budget == 0:
            if layer_idx == 0:
                self.track_retained_cache_positions(kv_len, list(range(kv_len)))
            return keys, values
    
        # add to the accumulated attention weights
        acc_attn, n_tokens_in_sum = self._get_layer_state(module)
        n_existing = acc_attn.shape[2]
        bsz, n_heads, _, q_len = attentions.shape

        n_kv_groups = module.num_key_value_groups
        n_kv_heads = n_heads // n_kv_groups

        new_acc_attn = attentions.sum(2) # bsz, n_heads, q_len
        # reshape attentions to bsz, n_kv_heads, n_kv_groups, q_len
        new_acc_attn = new_acc_attn.view(bsz, -1, n_kv_groups, q_len)
        # average over the n_kv_groups dimension
        new_acc_attn = new_acc_attn.mean(2) # bsz, n_kv_heads, q_len
        new_acc_attn[:, :, :n_existing] += acc_attn
        new_n_tokens_in_sum = torch.ones(bsz, n_kv_heads, q_len, device=attentions.device, dtype=attentions.dtype)
        new_n_tokens_in_sum[:, :, :n_existing] += n_tokens_in_sum
        self._set_layer_state(module, new_acc_attn, new_n_tokens_in_sum)

        if self.cache_budget >= q_len:
            # All tokens retained, track if needed
            if layer_idx == 0 and self.input_tokens is not None:
                if kv_len <= len(self.input_tokens):
                    all_token_ids = self.input_tokens[:kv_len].cpu().tolist()
                    retained_token_ids = all_token_ids.copy()
                else:
                    all_token_ids = self.input_tokens.cpu().tolist() + list(range(len(self.input_tokens), kv_len))
                    retained_token_ids = all_token_ids.copy()
                self.track_generation_step(all_token_ids, retained_token_ids, self.tokenizer)
                self.track_retained_cache_positions(kv_len, list(range(kv_len)))
            return keys, values

        # Compute scores
        scores = self.score(module, hidden_states, keys, values, attentions, False, kwargs)
        indices = self._select_heavy_and_recent(scores)

        # Track token retention/eviction at first layer only
        if layer_idx == 0 and self.input_tokens is not None:
            # Map position indices to actual token IDs
            if kv_len <= len(self.input_tokens):
                all_token_ids = self.input_tokens[:kv_len].cpu().tolist()
                retained_positions = indices[0, 0, :].cpu().tolist()  # Get retained position indices
                retained_token_ids = [all_token_ids[pos] for pos in retained_positions]
            else:
                # If kv_len > input_tokens, we have generated tokens
                all_token_ids = self.input_tokens.cpu().tolist() + list(range(len(self.input_tokens), kv_len))
                retained_positions = indices[0, 0, :].cpu().tolist()
                retained_token_ids = [all_token_ids[pos] if pos < len(self.input_tokens) else pos for pos in retained_positions]
            self.track_generation_step(all_token_ids, retained_token_ids, self.tokenizer)
            self.track_retained_cache_positions(kv_len, retained_positions)

        # Prune keys and values
        kv_indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim) # bsz, num_key_value_heads, cache_budget, head_dim
        keys = keys.gather(2, kv_indices).contiguous()
        values = values.gather(2, kv_indices).contiguous()

        # Prune acc attention weights and n_tokens_in_sum
        acc_attn, n_tokens_in_sum = self._get_layer_state(module)
        acc_attn = acc_attn.gather(2, indices).contiguous()
        n_tokens_in_sum = n_tokens_in_sum.gather(2, indices).contiguous()
        self._set_layer_state(module, acc_attn, n_tokens_in_sum)

        return keys, values


    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: Tuple):
        output = super().forward_hook(module, input, kwargs, output)
        # attentions are needed as input for the hook, but unless the user wants to return them in the output,
        # we can remove them to save memory
        if not self.output_attentions:
            output = (output[0], None)

        return output
