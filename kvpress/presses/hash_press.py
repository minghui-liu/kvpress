# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass

import torch
from torch import nn
import collections

from kvpress.presses.base_press import BasePress


@dataclass
class HashPress(BasePress):
    """Prune KV pairs with Locality Sensitive Hashing (https://arxiv.org/pdf/)"""

    # compression_ratio: float = 0.0
    # hash_size: int = 8

    def __post_init__(self):
        self.hash_table = collections.defaultdict(list)
        self.hyperplanes = None
        self.hash_size = 8
        self.compression_ratio = 0.0

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        self.hash_table.clear()
        bsz, num_key_value_heads, n_tokens, hidden_size = keys.shape

        num_buckets = int(n_tokens * self.compression_ratio)
        if num_buckets < 1:
            num_buckets = 1

        # calculate the dimension needed for the hash
        
        self.hash_size = int(torch.log2(torch.tensor(num_buckets)).item())
        self.hyperplanes = torch.randn(hidden_size, self.hash_size, dtype=keys.dtype, device=keys.device)

        # hash the keys and group them into buckets

        for head in range(num_key_value_heads):
            self.index(keys[:, head])

        self.index(keys)

        # for each bucket, calculate the average key and value
        compressed_keys = []
        compressed_values = []
        compressed_lens = []
        for bucket in self.hash_table.values():
            if len(bucket) == 0:
                continue
            bucket_keys = keys[bucket]
            bucket_values = values[bucket]
            compressed_lens.append(len(bucket))
            compressed_keys.append(torch.mean(bucket_keys, dim=0))
            compressed_values.append(torch.mean(bucket_values, dim=0))

        print(f"[DEBUG] num_buckets: {num_buckets}, hash_size: {self.hash_size}, compressed_keys: {len(compressed_keys)}")
              
        compressed_keys = torch.stack(compressed_keys)
        compressed_values = torch.stack(compressed_values)
        
        return compressed_keys, compressed_values

    def hash_fn(self, vectors):
        """
        Locality sensitive hashing function
        """
        projections = vectors @ self.hyperplanes
        hash_code = projections >= 0
        return hash_code

    def hamming_dist(self, hash1, hash2):
        """
        Compute the hamming distance between two hash codes
        """
        return torch.sum(hash1 != hash2, dim=-1)
    
    def index(self, vectors):
        for idx, vector in enumerate(vectors):
            hash_code = self.hash_fn(vector)
            self.hash_table[hash_code].append(idx)
    
    def query(self, query_vector):
        query_hash_code = self.hash_fn(query_vector)
        return self.hash_table.get(query_hash_code, [])