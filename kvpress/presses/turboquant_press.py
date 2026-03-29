# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from kvpress.presses.base_press import BasePress


@dataclass
class TurboQuantPress(BasePress):
    """
    TurboQuant: rotation-based KV cache quantization (Google Research).

    Applies a randomized Hadamard transform (H * diag(signs)) to keys and values
    before quantization, which spreads outliers across all dimensions and makes the
    distribution more uniform, significantly improving low-bit quantization quality.

    This is not a token-pruning method — all tokens are kept. The "compression" comes
    from quantization noise introduced by low-bit (n_bits=4 or 8) representation.

    Pipeline per attention layer (prefill & decode):
        1. Apply random sign flip:       x' = diag(signs) * x
        2. Hadamard transform + scale:   x_rot = H * x' / sqrt(d)
        3. Per-token min-max quantize:   x_q = quantize(x_rot, n_bits)
        4. Dequantize:                   x_dq = dequantize(x_q)
        5. Inverse rotation:             x_out = diag(signs) * H * x_dq / sqrt(d)

    Budget equivalence
    ------------------
    To match a token-pruning method with cache_budget B on a sequence of N tokens,
    the equivalent bit-width is:  n_bits = (B / N) * 16  (since float16 = 16 bits)

    Common mappings:
        compression 25%  (e.g. 1024 budget / 4096 tokens)  →  INT4  (n_bits=4)
        compression 50%  (e.g. 1024 budget / 2048 tokens)  →  INT8  (n_bits=8)

    If cache_budget > 0, n_bits is computed automatically from the actual sequence
    length at runtime using the formula above, clamped to [1, 16].
    Set cache_budget = 0 (default) to use n_bits directly.

    Parameters
    ----------
    n_bits : int
        Number of bits for quantization when cache_budget == 0. Default: 4.
    cache_budget : int
        If > 0, overrides n_bits: computes bits = round(cache_budget / seq_len * 16).
        Set this to compare fairly against token-pruning methods. Default: 0.
    seed : int
        Random seed for the rotation sign vector, shared across all layers. Default: 42.

    Reference: TurboQuant (Google Research)
    """

    n_bits: int = 4
    cache_budget: int = 0
    seed: int = 42

    def __post_init__(self):
        super().__post_init__()
        # Cache sign vectors per head_dim to avoid recomputation
        self._sign_cache: dict = field(default_factory=dict)
        self._sign_cache = {}

    # ------------------------------------------------------------------
    # Rotation helpers
    # ------------------------------------------------------------------

    def _get_signs(self, head_dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Return (and cache) the random ±1 sign vector for a given head_dim."""
        if head_dim not in self._sign_cache:
            gen = torch.Generator()
            gen.manual_seed(self.seed)
            signs = (torch.randint(0, 2, (head_dim,), generator=gen) * 2 - 1).float()
            self._sign_cache[head_dim] = signs
        return self._sign_cache[head_dim].to(device=device, dtype=dtype)

    @staticmethod
    def _next_power_of_2(n: int) -> int:
        return 1 << ((n - 1).bit_length()) if n > 1 else 1

    @staticmethod
    def _fast_hadamard(x: torch.Tensor) -> torch.Tensor:
        """
        In-place iterative Walsh-Hadamard transform along the last dimension.
        Requires last dim to be a power of 2.
        Result is un-normalized (caller divides by sqrt(n)).
        """
        n = x.shape[-1]
        batch = x.shape[:-1]
        h = 1
        while h < n:
            x = x.view(*batch, n // (2 * h), 2, h)
            a = x[..., 0, :]   # (..., groups, h)
            b = x[..., 1, :]
            x = torch.stack([a + b, a - b], dim=-2)  # (..., groups, 2, h)
            x = x.view(*batch, n)
            h *= 2
        return x

    def _rotate(self, x: torch.Tensor) -> torch.Tensor:
        """Forward rotation:  x_rot = H * diag(signs) * x / sqrt(pad_dim)."""
        orig_dim = x.shape[-1]
        signs = self._get_signs(orig_dim, x.device, x.dtype)
        x = x * signs                                   # sign flip

        pad_dim = self._next_power_of_2(orig_dim)
        if orig_dim != pad_dim:
            x = F.pad(x, (0, pad_dim - orig_dim))

        x = self._fast_hadamard(x) / (pad_dim ** 0.5)

        return x[..., :orig_dim]

    def _rotate_inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse rotation:  x_out = diag(signs) * H * x / sqrt(pad_dim)."""
        orig_dim = x.shape[-1]
        signs = self._get_signs(orig_dim, x.device, x.dtype)

        pad_dim = self._next_power_of_2(orig_dim)
        if orig_dim != pad_dim:
            x = F.pad(x, (0, pad_dim - orig_dim))

        x = self._fast_hadamard(x) / (pad_dim ** 0.5)
        x = x[..., :orig_dim] * signs                  # sign flip

        return x

    # ------------------------------------------------------------------
    # Quantization helpers
    # ------------------------------------------------------------------

    def _quantize_dequantize(self, x: torch.Tensor, n_bits: int) -> torch.Tensor:
        """
        Per-token (per-sequence-position) asymmetric min-max quantization.
        Quantizes to n_bits and immediately dequantizes back to float,
        introducing the quantization error that TurboQuant would cause.
        """
        # x shape: (batch, heads, seq, head_dim)
        min_val = x.amin(dim=-1, keepdim=True)
        max_val = x.amax(dim=-1, keepdim=True)
        scale = (max_val - min_val) / (2 ** n_bits - 1)
        scale = scale.clamp(min=1e-8)

        x_q = ((x - min_val) / scale).round().clamp(0, 2 ** n_bits - 1)
        return x_q * scale + min_val

    def _resolve_n_bits(self, seq_len: int) -> int:
        """
        Compute effective bit-width.
        If cache_budget > 0: n_bits = round(cache_budget / seq_len * 16), clamped to [1, 16].
        Otherwise use self.n_bits directly.
        """
        if self.cache_budget > 0:
            return max(1, min(16, round(self.cache_budget / seq_len * 16)))
        return self.n_bits

    # ------------------------------------------------------------------
    # Core TurboQuant pipeline
    # ------------------------------------------------------------------

    def _turboquant(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate → quantize → dequantize → inverse-rotate."""
        seq_len = x.shape[2]
        n_bits = self._resolve_n_bits(seq_len)
        x_rot = self._rotate(x)
        x_rot_dq = self._quantize_dequantize(x_rot, n_bits)
        return self._rotate_inverse(x_rot_dq)

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
        return self._turboquant(keys), self._turboquant(values)

    def compress_decoding(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._turboquant(keys), self._turboquant(values)
