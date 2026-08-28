# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.llama.modeling_llama import repeat_kv, rotate_half

from kvpress.presses.base_press import BasePress


@dataclass
class ZipCachePress(BasePress):
    """
    ZipCache: mixed-precision KV cache quantization with salient token identification
    (https://arxiv.org/abs/2405.14256).

    Like TurboQuantPress, this is NOT a token-pruning method: every token is kept and
    the "compression" comes from quantization error. It is implemented as simulated
    quantization (quantize then immediately dequantize) so that accuracy is directly
    comparable to the eviction presses at a matched bit budget.

    Three ideas from the paper are implemented:

    1. Channel-separable token-wise quantization.
       Keys and values are quantized per token, with min/max taken along the channel
       (head_dim) axis, optionally in groups of `group_size` channels. This avoids the
       outlier-channel problem that per-tensor quantization suffers from.

    2. Normalized attention score for saliency.
       Raw accumulated attention is biased by the causal mask: an early token is visible
       to many queries and so accumulates a large score purely by position. ZipCache
       divides each token's accumulated attention by the NUMBER OF QUERIES that could
       attend to it, which removes that positional bias. This normalization is the main
       accuracy win of the paper and is what distinguishes it from an H2O-style score.

    3. Mixed precision.
       The top `salient_ratio` fraction of tokens by normalized score are stored at
       `high_bits`; everything else at `low_bits`. The paper's headline configuration is
       4-bit salient / 2-bit non-salient, i.e. high_bits=4, low_bits=2.

    Efficiency approximation
    ------------------------
    The paper estimates saliency from a small probe of recent queries rather than the
    full attention matrix. We do the same: `probe_window` trailing queries are used. If
    the model already returns attentions (output_attentions=True, eager attention) we
    reuse them; otherwise the probe attention is recomputed from the hidden states, the
    same way SnapKVPress does.

    Decoding
    --------
    During decoding a single new query gives a very noisy saliency estimate, so the
    `recent_window` most recent tokens are always treated as salient and the probe-based
    ranking decides the rest. Recent tokens dominate attention in practice, so this
    matches the paper's behaviour while staying cheap.

    Parameters
    ----------
    high_bits : int
        Bit-width for salient tokens. Default 4.
    low_bits : int
        Bit-width for non-salient tokens. Default 2.
    salient_ratio : float
        Fraction of tokens kept at high_bits, in [0, 1]. Default 0.1.
    probe_window : int
        Number of trailing queries used to estimate saliency. Default 32.
    recent_window : int
        Most recent tokens always treated as salient. Default 32.
    group_size : int
        Channels per quantization group. 0 means one group over the whole head_dim.
    cache_budget : int
        If > 0, `salient_ratio` is derived from it so that the average bit-width matches
        a token-pruning method with this budget. See `_resolve_salient_ratio`.
    """

    high_bits: int = 4
    low_bits: int = 2
    salient_ratio: float = 0.1
    probe_window: int = 32
    recent_window: int = 32
    group_size: int = 0
    cache_budget: int = 0

    def __post_init__(self):
        super().__post_init__()
        if not (0.0 <= self.salient_ratio <= 1.0):
            raise ValueError("salient_ratio must be in [0, 1]")
        if self.high_bits < self.low_bits:
            raise ValueError("high_bits must be >= low_bits")

    # ------------------------------------------------------------------
    # Budget bookkeeping
    # ------------------------------------------------------------------

    def _resolve_salient_ratio(self, seq_len: int) -> float:
        """
        With cache_budget B on seq_len N tokens, a pruning method keeps B/N of the
        fp16 cache. ZipCache spends  r*high + (1-r)*low  bits per token, so matching
        that budget means solving

            r*high + (1-r)*low = 16 * B / N   ->   r = (16*B/N - low) / (high - low)

        Clamped to [0, 1]. cache_budget = 0 uses salient_ratio directly.
        """
        if self.cache_budget <= 0 or self.high_bits == self.low_bits:
            return self.salient_ratio
        target = 16.0 * self.cache_budget / max(seq_len, 1)
        r = (target - self.low_bits) / (self.high_bits - self.low_bits)
        return float(min(1.0, max(0.0, r)))

    # ------------------------------------------------------------------
    # Quantization
    # ------------------------------------------------------------------

    def _quantize_dequantize(self, x: torch.Tensor, n_bits: int) -> torch.Tensor:
        """
        Channel-separable token-wise asymmetric min-max quantize-dequantize.
        x: (batch, heads, seq, head_dim). Groups run along head_dim.
        """
        if n_bits >= 16:
            return x
        orig_shape = x.shape
        head_dim = orig_shape[-1]
        g = self.group_size if self.group_size and self.group_size > 0 else head_dim
        if head_dim % g != 0:
            g = head_dim  # fall back to a single group rather than padding
        x = x.reshape(*orig_shape[:-1], head_dim // g, g)

        min_val = x.amin(dim=-1, keepdim=True)
        max_val = x.amax(dim=-1, keepdim=True)
        levels = 2 ** n_bits - 1
        scale = ((max_val - min_val) / levels).clamp(min=1e-8)
        x_q = ((x - min_val) / scale).round().clamp(0, levels)
        out = x_q * scale + min_val
        return out.reshape(orig_shape)

    # ------------------------------------------------------------------
    # Saliency
    # ------------------------------------------------------------------

    @staticmethod
    def _probe_attention(module, hidden_states, keys, probe_window, position_embeddings):
        """
        Attention weights of the last `probe_window` queries against ALL kv positions.

        Adapted from SnapKVPress.compute_window_attention, but it deliberately does NOT
        drop the trailing window columns: ZipCache needs a score for every kv position,
        including the most recent ones.
        Returns (bsz, num_heads, actual_window, kv_len).
        """
        bsz = hidden_states.shape[0]
        num_heads = module.config.num_attention_heads
        head_dim = module.head_dim
        num_key_value_groups = num_heads // module.config.num_key_value_heads
        kv_len = keys.shape[2]

        if hasattr(module, "q_proj"):
            query_states = module.q_proj(hidden_states[:, -probe_window:])
        elif hasattr(module, "qkv_proj"):
            qkv = module.qkv_proj(hidden_states[:, -probe_window:])
            query_states = qkv[..., : num_heads * head_dim]
        else:
            raise NotImplementedError(f"ZipCache not yet implemented for {module.__class__}.")

        actual_window = query_states.shape[1]
        query_states = query_states.view(bsz, actual_window, num_heads, head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        cos, sin = cos[:, -actual_window:], sin[:, -actual_window:]
        query_states = (query_states * cos.unsqueeze(1)) + (rotate_half(query_states) * sin.unsqueeze(1))

        key_states = repeat_kv(keys, num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
        # Causal mask: probe query t (absolute position kv_len - actual_window + t)
        # may not attend beyond itself.
        mask = torch.ones_like(attn_weights) * float("-inf")
        mask = torch.triu(mask, diagonal=kv_len - actual_window + 1)
        attn_weights = attn_weights + mask
        return F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

    def _normalized_scores(self, module, hidden_states, keys, attentions, kwargs) -> torch.Tensor:
        """
        ZipCache's normalized attention score, per kv head.
        Returns (bsz, num_kv_heads, kv_len); higher means more salient.
        """
        bsz, num_kv_heads, kv_len, _ = keys.shape
        num_heads = module.config.num_attention_heads
        num_key_value_groups = num_heads // num_kv_heads

        if attentions is not None:
            attn = attentions[..., -self.probe_window :, :]
        else:
            attn = self._probe_attention(
                module, hidden_states, keys, self.probe_window, kwargs["position_embeddings"]
            )

        attn = attn.float()
        window = attn.shape[-2]
        # Accumulated attention per kv position.
        acc = attn.sum(dim=-2)                                   # (bsz, num_heads, kv_len)

        # --- the normalization that defines ZipCache -------------------------
        # Probe queries sit at absolute positions [kv_len - window, kv_len). Key j is
        # visible to query q iff j <= q, so the number of probe queries that could
        # attend to j is  kv_len - max(j, kv_len - window).
        idx = torch.arange(kv_len, device=attn.device)
        start = kv_len - window
        n_visible = (kv_len - torch.maximum(idx, torch.full_like(idx, start))).clamp(min=1)
        scores = acc / n_visible.to(acc.dtype)                   # (bsz, num_heads, kv_len)

        # Average across query heads within each kv group (GQA).
        scores = scores.view(bsz, num_kv_heads, num_key_value_groups, kv_len).mean(dim=2)
        return scores

    def _salient_mask(self, module, hidden_states, keys, attentions, kwargs) -> torch.Tensor:
        """Boolean (bsz, num_kv_heads, kv_len, 1): True where the token gets high_bits."""
        bsz, num_kv_heads, kv_len, _ = keys.shape
        ratio = self._resolve_salient_ratio(kv_len)

        if ratio >= 1.0:
            return torch.ones(bsz, num_kv_heads, kv_len, 1, dtype=torch.bool, device=keys.device)
        n_salient = int(math.ceil(ratio * kv_len))

        recent = min(self.recent_window, kv_len)
        mask = torch.zeros(bsz, num_kv_heads, kv_len, dtype=torch.bool, device=keys.device)
        if recent > 0:
            mask[..., -recent:] = True                 # recent tokens always salient

        extra = n_salient - recent
        if extra > 0:
            try:
                scores = self._normalized_scores(module, hidden_states, keys, attentions, kwargs)
            except (KeyError, NotImplementedError):
                # No position_embeddings / unsupported module: fall back to pure recency.
                fallback = min(n_salient, kv_len)
                mask[...] = False
                if fallback > 0:
                    mask[..., -fallback:] = True
                return mask.unsqueeze(-1)
            scores = scores.masked_fill(mask, float("-inf"))     # don't re-pick recent
            extra = min(extra, kv_len - recent)
            if extra > 0:
                topk = scores.topk(extra, dim=-1).indices
                mask.scatter_(-1, topk, True)
        elif extra < 0:
            # Budget smaller than the recency window: keep only the newest n_salient.
            mask[...] = False
            if n_salient > 0:
                mask[..., -n_salient:] = True

        return mask.unsqueeze(-1)

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    def _zipcache(self, module, hidden_states, keys, values, attentions, kwargs):
        mask = self._salient_mask(module, hidden_states, keys, attentions, kwargs)
        out = []
        for x in (keys, values):
            hi = self._quantize_dequantize(x, self.high_bits)
            lo = self._quantize_dequantize(x, self.low_bits)
            out.append(torch.where(mask, hi, lo))
        return out[0], out[1]

    # ------------------------------------------------------------------
    # BasePress interface
    # ------------------------------------------------------------------

    def compress_prefilling(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._zipcache(module, hidden_states, keys, values, attentions, kwargs)

    def compress_decoding(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._zipcache(module, hidden_states, keys, values, attentions, kwargs)
