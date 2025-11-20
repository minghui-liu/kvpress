# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from kvpress.presses.base_press import BasePress
from torch import nn


@dataclass
class NonePress(BasePress):
    """
    A no-op press that does absolutely nothing.
    Returns keys and values unchanged, no compression, no tracking.
    Use this when you want to run inference without any KV cache compression.
    """
    
    def compress_prefilling(
        self,
        module: nn.Module,
        hidden_states,
        keys,
        values,
        attentions,
        kwargs: dict,
    ):
        """No compression - return keys and values as-is"""
        return keys, values
    
    def compress_decoding(
        self,
        module: nn.Module,
        hidden_states,
        keys,
        values,
        attentions,
        kwargs: dict,
    ):
        """No compression - return keys and values as-is"""
        return keys, values

