# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass

import torch
from torch import nn

from kvpress.presses.scorer_press import ScorerPress


@dataclass
class StreamingLLMPress(ScorerPress):
    """
    Prune a fixed number of KV pairs at the beginning and end of the sequence (https://arxiv.org/abs/2309.17453)
    We keep the first n_sink tokens and the last n_local tokens.
    n_local is computed using the compression ratio.

    Note that the original implementation https://github.com/mit-han-lab/streaming-llm additionally rerotates keys.
    This can be achieved by using
    press = KeyRerotationPress(press=StreamingLLMPress(compression_ratio, n_sink))
    """

    compression_ratio: float = 0.0
    cache_budget: int = 0
    n_sink: int = 4
    attn_csv_path: str = "attn_loss.csv"
    prune_step: int = 0
    output_attentions: bool = True

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
        
        if is_prefill:
            q_len = hidden_states.shape[1]
            assert q_len > self.n_sink, f"Input should contain more tokens than n_sink={self.n_sink}"
            n_pruned = q_len - self.cache_budget
            scores = torch.ones_like(keys[..., 0])
            scores[:, :, self.n_sink : self.n_sink + n_pruned] = 0
        else:
            # during generation, we keep the first n_sink tokens and the last n_local tokens
            n_local = self.cache_budget - self.n_sink
            scores = torch.zeros_like(keys[..., 0])
            scores[:, :, : self.n_sink] = 1
            scores[:, :, -n_local:] = 1

        # Debug prints similar to RKV/H2O
        full_len = scores.shape[-1]
        kept_len = scores.sum(dim=-1).int().min().item()  # minimum kept across batch/heads
        print("---" * 10)
        print(f"[DEBUG] (PRE) keys shape: {keys.shape}, values shape: {values.shape}")
        print(f"[DEBUG] diff indices: {full_len - kept_len}")


        return scores

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
            return keys, values

        # Compute scores and select kept indices
        scores = self.score(module, hidden_states, keys, values, attentions, False, kwargs)
        indices = scores.topk(self.cache_budget, dim=-1).indices  # [B, Hkv, K]

        # Debug counts
        # try:
        #     print("---" * 10)
        #     print(f"[DEBUG] (PRE) keys shape: {keys.shape}, values shape: {values.shape}")
        #     full_len = scores.shape[-1]
        #     kept_len = indices.shape[2]
        #     print(f"[DEBUG] diff indices: {full_len - kept_len}")
        # except Exception:
        #     pass

        # Gather pruned keys/values
        kv_indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)
        keys = keys.gather(2, kv_indices).contiguous()
        values = values.gather(2, kv_indices).contiguous()
        return keys, values
