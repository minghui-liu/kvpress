# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import torch
from torch import nn

from kvpress.presses.scorer_press import ScorerPress


@dataclass
class EvictPress(ScorerPress):
    """
    Evict KV entries whose average attention mass exceeds a threshold.

    Notes
    -----
    - We compute the average attention mass per KV head and key position by
      averaging over queries and (grouped) attention heads. This keeps values
      in [0, 1] and makes the threshold interpretable.
    - If the number of under-threshold keys is lower than the cache budget,
      the threshold is ignored for that head to keep the cache size stable.
    """

    cache_budget: int = 0
    compression_ratio: float = 0.0
    evict_threshold: float = 0.1

    def __post_init__(self):
        super().__post_init__()
        # Initialize attention capping hooks if needed
        self.attention_hooks = []

    def _compute_average_attention(
        self,
        attentions: torch.Tensor,
        keys: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-KV-head average attention mass over queries.

        Returns
        -------
        torch.Tensor
            Shape [B, H_kv, K]
        """
        # attentions: [B, H_attn, Q, K]
        attn_head = attentions.mean(dim=2)  # [B, H_attn, K]
        bsz, H_attn, kv_len = attn_head.shape
        H_kv = keys.shape[1]

        if H_attn >= H_kv:
            heads_per_group = max(1, H_attn // H_kv)
            H_use = H_kv * heads_per_group
            attn_head = attn_head[:, :H_use]
            attn_grouped = attn_head.view(bsz, H_kv, heads_per_group, kv_len).mean(dim=2)
            return attn_grouped

        # Fallback: repeat heads to cover all KV heads
        repeat = (H_kv + H_attn - 1) // H_attn
        attn_head = attn_head.repeat(1, repeat, 1)[:, :H_kv]
        return attn_head

    def _resolve_budget(self, kv_len: int) -> int:
        if self.cache_budget and self.cache_budget > 0:
            return min(self.cache_budget, kv_len)
        if self.compression_ratio and self.compression_ratio > 0:
            keep = int(kv_len * (1.0 - self.compression_ratio))
            return max(1, min(keep, kv_len))
        return kv_len

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
        if attentions is None:
            return torch.zeros(*keys.shape[:-1], device=keys.device, dtype=keys.dtype)

        avg_attn = self._compute_average_attention(attentions, keys)  # [B, H_kv, K]
        scores = -avg_attn.float()

        if self.evict_threshold is not None:
            over = avg_attn >= self.evict_threshold
            # Apply threshold only when enough keys remain under it for the budget
            budget = self._resolve_budget(avg_attn.shape[-1])
            under_counts = (~over).sum(dim=-1, keepdim=True)
            allow_threshold = under_counts >= budget
            if allow_threshold.any():
                scores = scores.masked_fill(over & allow_threshold, float("-inf"))
            with torch.no_grad():
                b = 0
                over_counts = over[b].sum(dim=-1)
                allow_counts = allow_threshold[b].sum().item()
                avg_min = float(avg_attn[b].min().item())
                avg_max = float(avg_attn[b].max().item())
            print(
                "[EVICT] score: "
                f"kv_len={avg_attn.shape[-1]} budget={budget} "
                f"threshold={self.evict_threshold} "
                f"heads_over_mean={over_counts.float().mean().item():.2f} "
                f"heads_over_max={int(over_counts.max().item())} "
                f"allow_heads={int(allow_counts)} "
                f"avg_attn_min={avg_min:.4f} avg_attn_max={avg_max:.4f}"
            )

        return scores.to(keys.dtype)

    def compress_prefilling(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if attentions is None:
            return keys, values

        kv_len = keys.shape[2]
        budget = self._resolve_budget(kv_len)
        if budget <= 0:
            return keys, values

        scores = self.score(module, hidden_states, keys, values, attentions, True, kwargs)
        indices = scores.topk(budget, dim=-1).indices
        if not getattr(self, "latency", False):
            self.compute_attention_loss(module, attentions, indices, window_size=getattr(self, "window_size", 0))
        with torch.no_grad():
            kept = indices[0].numel()
        print(
            "[EVICT] prefill: "
            f"layer={getattr(module, 'layer_idx', -1)} "
            f"kv_len={kv_len} kept={kept} budget={budget}"
        )

        indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)
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
        if attentions is None:
            return keys, values

        kv_len = keys.shape[2]
        budget = self._resolve_budget(kv_len)
        if budget <= 0:
            return keys, values

        scores = self.score(module, hidden_states, keys, values, attentions, False, kwargs)
        indices = scores.topk(budget, dim=-1).indices
        if not getattr(self, "latency", False):
            self.compute_attention_loss(module, attentions, indices, window_size=getattr(self, "window_size", 0))
        with torch.no_grad():
            kept = indices[0].numel()
        print(
            "[EVICT] decode: "
            f"layer={getattr(module, 'layer_idx', -1)} "
            f"kv_len={kv_len} kept={kept} budget={budget}"
        )

        indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)
        keys = keys.gather(2, indices).contiguous()
        values = values.gather(2, indices).contiguous()
        return keys, values
