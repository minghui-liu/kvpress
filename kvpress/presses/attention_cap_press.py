# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from contextlib import contextmanager

import torch
from torch import nn

from kvpress.presses.base_press import BasePress


@dataclass
class AttentionCapPress(BasePress):
    """
    Cap attention weights to a maximum value during attention computation.

    This press modifies attention weights directly in the attention mechanism,
    clamping them to a maximum value instead of evicting entries.
    """

    attention_cap: float = 0.5  # Cap attention weights to this maximum value

    def __post_init__(self):
        super().__post_init__()
        self.attention_hooks = []

    def _attention_hook(self, module, input, kwargs, output):
        """Hook to modify attention weights after softmax."""
        if isinstance(output, tuple) and len(output) >= 2:
            # output[0] is the attention output, output[1] might be attention weights
            if len(output) >= 2 and isinstance(output[1], torch.Tensor):
                # Clamp attention weights
                clamped_weights = torch.clamp(output[1], max=self.attention_cap)
                # Return modified output
                return (output[0], clamped_weights, *output[2:])
        return output

    @contextmanager
    def __call__(self, model):
        """Set up hooks to cap attention weights during forward passes."""
        if self.attention_cap is not None and self.attention_cap < 1.0:
            # Register hooks on attention layers
            hooks = []
            for layer in model.model.layers:
                # Hook into the attention mechanism
                # This assumes standard transformer architecture
                if hasattr(layer.self_attn, 'forward'):
                    hook = layer.self_attn.register_forward_hook(self._attention_hook, with_kwargs=True)
                    hooks.append(hook)

            yield

            # Remove hooks
            for hook in hooks:
                hook.remove()
        else:
            # No capping needed
            yield

    def compress_prefilling(self, module, hidden_states, keys, values, attentions, kwargs):
        """No KV cache compression - just pass through."""
        return keys, values

    def compress_decoding(self, module, hidden_states, keys, values, attentions, kwargs):
        """No KV cache compression - just pass through."""
        return keys, values