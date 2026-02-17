# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import math
import json
import os
from dataclasses import dataclass

import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from transformers.models.llama.modeling_llama import repeat_kv, rotate_half

from kvpress.presses.scorer_press import ScorerPress


@dataclass
class RKVLSHPress(ScorerPress):
    """
    RKV (https://www.arxiv.org/pdf/2505.24133)
    """

    cache_budget: int = 0
    compress_interval: int = 64 # aka. the buffer size 
    # compression_ratio: float = 0.0
    window_size: int = 8 # number of observation tokens always kept in the cache
    kernel_size: int = 5
    n_hash_buckets: int=6
    cos_hamming_distance_bucket: torch.Tensor=None
    powers_of_two: torch.Tensor=None
    num_buckets: int=None
    proj_matrix: torch.Tensor=None  # Cached projection matrix for LSH
    proj_matrix_head_dim: int=None  # Track head_dim for which proj_matrix was created
    cos_bucket_cached: torch.Tensor=None  # Cached cos_bucket on current device
    cos_bucket_device: str=None  # Track device for cached cos_bucket
    cos_bucket_dtype: torch.dtype=None  # Track dtype for cached cos_bucket
    powers_of_two_cached: torch.Tensor=None  # Cached powers_of_two on current device
    powers_of_two_device: str=None  # Track device for cached powers_of_two
    lam: float = 0.1

    def __post_init__(self):
        super().__post_init__()
        self.accumulated_tokens = 0  # Initialize accumulated tokens for compression interval
        self.hidden_size = None  # Will be set based on model type
        self.acc_hidden_states = None  # Will be initialized when hidden_size is known
        
        # Initialize ranking data collection
        # Only create directory if tracking is enabled (will be set later via set_tokenizer_and_tokens)
        self.ranking_data = []
        self.save_dir = "ranking_analysis"
        # Don't create directory here - only create when actually needed (when tokenizer is set)
        
        # Tokenizer for decoding tokens (will be set during inference)
        self.tokenizer = None
        self.input_tokens = None
        
        # Bucket tracking for analysis
        self.track_buckets = False
        self.bucket_counts = None  # Will be initialized when tracking is enabled
        self.bucket_counts_per_sample = []  # Store counts for each sample

        # Qualitative analysis mode
        self.enable_qualitative_analysis = False
        self.qualitative_data = []  # Store detailed token retention/eviction data
        self.current_sample_id = 0

    def enable_bucket_tracking(self):
        """Enable bucket count tracking for analysis."""
        self.track_buckets = True
        if self.num_buckets is not None:
            self.bucket_counts = np.zeros(self.num_buckets, dtype=np.int64)
        print(f"[RKV-LSH] Bucket tracking enabled (num_buckets={self.num_buckets})")
    
    def reset_bucket_counts(self):
        """Reset bucket counts for a new sample."""
        if self.track_buckets and self.num_buckets is not None:
            self.bucket_counts = np.zeros(self.num_buckets, dtype=np.int64)
    
    def get_bucket_counts(self):
        """Get current bucket counts and store for this sample."""
        if self.track_buckets and self.bucket_counts is not None:
            # Store a copy of current counts
            counts_copy = self.bucket_counts.copy()
            self.bucket_counts_per_sample.append(counts_copy)
            return counts_copy
        return None

    def enable_qualitative_mode(self):
        """Enable qualitative analysis mode to track token retention decisions."""
        self.enable_qualitative_analysis = True
        print(f"[RKV-LSH] Qualitative analysis mode enabled")

    def next_sample(self):
        """Mark the start of a new sample for qualitative analysis."""
        self.current_sample_id += 1
    
    def initialize_buckets(self, device=None):
        """
        Initialize cos_hamming_distance_bucket on the specified device.
        If device is None, uses CUDA if available, otherwise CPU.

        For large n_hash_buckets (>16), skips precomputing the full matrix
        and uses on-the-fly computation instead to avoid memory overflow.
        """
        # Determine device: use provided device, or CUDA if available, else CPU
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.num_buckets = 2 ** self.n_hash_buckets

        # For large bucket sizes (>16 bits = 65536 buckets), avoid precomputing
        # the full pairwise matrix to prevent memory overflow
        if self.n_hash_buckets > 16:
            print(f"[RKV-LSH] n_hash_buckets={self.n_hash_buckets} (num_buckets={self.num_buckets})")
            print(f"[RKV-LSH] Using on-the-fly Hamming distance computation to avoid memory overflow")
            self.cos_hamming_distance_bucket = None  # Signal to use on-the-fly computation
        else:
            # For reasonable bucket sizes, pre-compute the full pairwise matrix
            buckets = torch.arange(self.num_buckets, device=device)
            a = buckets.view(-1, 1)  # [N, 1]
            b = buckets.view(1, -1)  # [1, N]
            xor_vals = a ^ b
            # Use efficient bitwise popcount - with fallback for older PyTorch versions
            # This counts the number of set bits (Hamming weight) in each element
            if hasattr(torch, 'bitwise_popcount'):
                hamming = torch.bitwise_popcount(xor_vals).to(torch.int64)
            else:
                # Fallback: manual bit counting using bitwise operations (vectorized)
                # This works for older PyTorch versions that don't have bitwise_popcount
                hamming = torch.zeros_like(xor_vals, dtype=torch.int64)
                temp = xor_vals.clone()
                # Count bits by repeatedly shifting and masking
                # This is O(log(max_value)) but fully vectorized on GPU
                max_bits = self.n_hash_buckets
                for _ in range(max_bits):
                    hamming += (temp & 1).long()
                    temp = temp >> 1
            self.cos_hamming_distance_bucket = torch.cos(hamming / self.n_hash_buckets)

        # Use bit shifting instead of exponentiation for faster computation
        # 1 << [0, 1, 2, ...] = [1, 2, 4, 8, ...] = [2^0, 2^1, 2^2, ...]
        # Bit shifting is faster than exponentiation, compute as int then convert to bfloat16
        arange = torch.arange(self.n_hash_buckets, device=device, dtype=torch.int64)
        self.powers_of_two = (torch.tensor(1, device=device, dtype=torch.int64) << arange).to(torch.bfloat16)
        self.num_buckets = 2 ** self.n_hash_buckets

    @staticmethod
    def compute_window_attention(module, hidden_states, keys, window_size, position_embeddings):
        """
        Compute the last window_size queries and associated attention weights for the first q_len - window_size keys.
        """
        bsz, q_len, _ = hidden_states.shape
        num_heads = module.config.num_attention_heads
        head_dim = module.head_dim
        num_key_value_groups = num_heads // module.config.num_key_value_heads

        # Get last window_size queries
        if hasattr(module, "q_proj"):
            query_states = module.q_proj(hidden_states[:, -window_size:])
        elif hasattr(module, "qkv_proj"):
            qkv = module.qkv_proj(hidden_states[:, -window_size:])
            query_states = qkv[..., : num_heads * head_dim]
        else:
            raise NotImplementedError(f"SnapKV not yet implemented for {module.__class__}.")
        
        query_states = query_states.view(bsz, window_size, num_heads, head_dim).transpose(1, 2)

        # Apply RoPE
        cos, sin = position_embeddings
        cos, sin = cos[:, -window_size:], sin[:, -window_size:]
        query_states = (query_states * cos.unsqueeze(1)) + (rotate_half(query_states) * sin.unsqueeze(1))

        # Compute attention for first q_len - window_size tokens
        key_states = repeat_kv(keys, num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
        attention_mask = torch.ones_like(attn_weights) * float("-1e9")
        attention_mask = torch.triu(attention_mask, diagonal=q_len - window_size + 1)
        attn_weights += attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.bfloat16).to(query_states.dtype)
        attn_weights = attn_weights[..., :-window_size]

        return attn_weights


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

        bsz, num_key_value_heads, q_len, _ = keys.shape
        num_key_value_groups = module.config.num_attention_heads // num_key_value_heads

        assert q_len > self.window_size, "Query length should be greater than the window size"
        
        # If lam == 0, skip attention computation and only compute redundancy
        if self.lam == 0:
            # Skip attention computation entirely when lam=0 (only redundancy matters)
            scores = None
        else:
            # Compute attention weights
            if attentions is not None:
                attn_weights = attentions[..., -self.window_size :, : -self.window_size]
            else:
                attn_weights = self.compute_window_attention(
                    module, hidden_states, keys, self.window_size, kwargs["position_embeddings"]
                )
            scores = attn_weights.mean(dim=-2)   
            # Average per group (https://github.com/FasterDecoding/SnapKV/issues/22)
            scores = scores.view(bsz, num_key_value_heads, num_key_value_groups, q_len - self.window_size)
            scores = scores.max(dim=-2).values
            # Stablization and Importance Estimation
            scores = F.max_pool1d(scores, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=1)
        
        # Redundancy Estimation via Semantic Similarity
        # If lam == 1, skip redundancy computation (only attention matters)
        if self.lam == 1:
            # Skip redundancy computation entirely when lam=1
            redundancy = None
        else:
            # normalize keys by dividing the l2 norm of keys + eps (1e-8) 
            eps = 1e-8
            keys_norm = keys.norm(dim=-1, keepdim=True) + eps
            keys = keys / keys_norm

            ### Original Algorithm: directly using the cosine similarity
            # # compute the cosine similarity between keys
            # keys_flat = keys.view(bsz, num_key_value_heads, -1, keys.shape[-1])
            # keys_flat = keys_flat[:, :, : -self.window_size, :]  # Exclude the last window_size keys
            # keys_similarity = torch.einsum("bhqd,bhkd->bhqk", keys_flat, keys_flat)
            # # zero out the diagonal (self-similarity)
            # mask = torch.eye(keys_similarity.shape[-1], device=keys_similarity.device).unsqueeze(0).unsqueeze(0)
            # keys_similarity = keys_similarity * (1 - mask)

            # redundency = keys_similarity.mean(dim=-1)  # Average over the key dimension
            # redundency = F.softmax(redundency, dim=-1, dtype=torch.float32).to(scores.dtype)
     
            ### Modified Algorithm: implement LSH over that
            keys_flat = keys.view(bsz, num_key_value_heads, -1, keys.shape[-1])
            keys_flat = keys_flat[:, :, : -self.window_size, :]  # Exclude the last window_size keys

            # Construct LSH buckets with cached projection matrix
            head_dim = keys_flat.shape[-1]
            device = keys.device
            dtype = keys_flat.dtype
            
            # Cache projection matrix to avoid recreating it every time
            # Only recreate if head_dim changed (e.g., different model) or device/dtype mismatch
            if (self.proj_matrix is None or 
                self.proj_matrix_head_dim != head_dim or 
                str(self.proj_matrix.device) != str(device) or
                self.proj_matrix.dtype != dtype):
                self.proj_matrix = torch.randn(
                    head_dim, self.n_hash_buckets, 
                    device=device, dtype=dtype
                )
                self.proj_matrix_head_dim = head_dim
            
            q_len_flat, head_dim = keys_flat.shape[2], keys_flat.shape[3]
            keys_reshaped = keys_flat.reshape(-1, head_dim)  # [B*H*Q, D]
            hash_bits_flat = torch.matmul(keys_reshaped, self.proj_matrix)  # [B*H*Q, K] - faster than einsum
            hash_bits = hash_bits_flat.reshape(bsz, num_key_value_heads, q_len_flat, self.n_hash_buckets)  # [B, H, Q, K]
            # Convert to binary codes and compute integer hash codes in one step
            hash_codes = (hash_bits > 0).int()  # [B, H, Q, K]
            # Cache powers_of_two device transfer - only move if device changed
            device_str = str(hash_codes.device)
            if (self.powers_of_two_cached is None or 
                self.powers_of_two_device != device_str):
                self.powers_of_two_cached = self.powers_of_two.to(hash_codes.device)
                self.powers_of_two_device = device_str
            powers_of_two = self.powers_of_two_cached
            # Compute hash codes as integers: sum of binary bits weighted by powers of 2
            hash_codes_int = torch.sum(hash_codes * powers_of_two, dim=-1)  # [B, H, Q]

            # Fully vectorized computation on GPU - no CPU transfers, no Python loops
            # Shape: hash_codes_int is [B, H, Q]
            bsz, num_heads, q_len = hash_codes_int.shape

            # Flatten for batch processing: [B*H, Q]
            codes_flat = hash_codes_int.view(-1, q_len).long()  # [B*H, Q]
            device = codes_flat.device
            dtype = keys.dtype

            # According to paper: S_i' = (Σ_{i≠j} c_j cos(Hamming(i,j)/b)) / (Σ_j c_j)
            # Two paths: precomputed matrix (fast) or sparse on-the-fly computation (memory-efficient)
            if self.cos_hamming_distance_bucket is not None:
                # Fast path: use precomputed cosine hamming distance matrix
                # Vectorized bucket counting using scatter_add (fully GPU-accelerated)
                # Count tokens in each bucket for each batch-head: [B*H, num_buckets]
                counts = torch.zeros(bsz * num_heads, self.num_buckets, device=device, dtype=torch.bfloat16)
                counts.scatter_add_(1, codes_flat, torch.ones_like(codes_flat, dtype=torch.bfloat16))

                # Track bucket counts if enabled
                if self.track_buckets:
                    # Aggregate counts across all heads and batch
                    bucket_counts_total = counts.sum(dim=0).cpu().numpy().astype(np.int64)
                    if self.bucket_counts is None:
                        self.bucket_counts = bucket_counts_total
                    else:
                        self.bucket_counts += bucket_counts_total

                # Compute total counts per batch-head: [B*H, 1]
                total_counts = counts.sum(dim=1, keepdim=True)  # [B*H, 1]

                device_str = str(keys.device)
                if (self.cos_bucket_cached is None or
                    self.cos_bucket_device != device_str or
                    self.cos_bucket_dtype != dtype):
                    self.cos_bucket_cached = self.cos_hamming_distance_bucket.to(keys.device).to(dtype)
                    self.cos_bucket_device = device_str
                    self.cos_bucket_dtype = dtype
                cos_bucket = self.cos_bucket_cached  # [num_buckets, num_buckets]

                # Optimized: compute (counts @ cos_bucket - counts) / total_counts in one step
                # This excludes self-similarity (diagonal terms where cos(0) = 1.0)
                avg_cosine = (counts @ cos_bucket - counts) / (total_counts + 1e-8)  # [B*H, num_buckets]

                # Map each token's bucket code to its average cosine value
                redundancy_flat = avg_cosine.gather(1, codes_flat)  # [B*H, Q]
            else:
                # Memory-efficient path: fully sparse computation for large bucket spaces
                # Never materialize tensors with num_buckets dimension to avoid OOM

                # Get unique bucket codes that actually have tokens (across all batch-heads)
                unique_codes = torch.unique(codes_flat)  # [num_occupied_buckets]
                num_occupied = unique_codes.shape[0]

                # Create mapping from bucket codes to sparse indices
                # Use a dictionary-like sparse mapping (only stores occupied buckets)
                max_code = unique_codes.max().item() + 1
                code_to_idx = torch.full((max_code,), -1, device=device, dtype=torch.long)
                code_to_idx[unique_codes] = torch.arange(num_occupied, device=device)

                # Map codes_flat to sparse indices: [B*H, Q]
                codes_idx = code_to_idx[codes_flat]  # [B*H, Q]

                # Count tokens per unique bucket (sparse): [B*H, num_occupied]
                counts_sparse = torch.zeros(bsz * num_heads, num_occupied, device=device, dtype=dtype)
                counts_sparse.scatter_add_(1, codes_idx, torch.ones_like(codes_flat, dtype=dtype))

                # Track bucket counts if enabled
                if self.track_buckets:
                    # Aggregate counts for occupied buckets
                    bucket_counts_sparse = counts_sparse.sum(dim=0).cpu().numpy().astype(np.int64)
                    # Create full bucket counts array (only for tracking)
                    if self.bucket_counts is None:
                        self.bucket_counts = np.zeros(self.num_buckets, dtype=np.int64)
                    self.bucket_counts[unique_codes.cpu().numpy()] += bucket_counts_sparse

                # Compute total counts per batch-head: [B*H, 1]
                total_counts = counts_sparse.sum(dim=1, keepdim=True)  # [B*H, 1]

                # Compute hamming distances only between occupied buckets
                # XOR between all pairs of occupied buckets
                a = unique_codes.view(-1, 1)  # [num_occupied, 1]
                b = unique_codes.view(1, -1)  # [1, num_occupied]
                xor_vals = a ^ b  # [num_occupied, num_occupied]

                # Count bits using bitwise_popcount
                if hasattr(torch, 'bitwise_popcount'):
                    hamming = torch.bitwise_popcount(xor_vals).to(torch.float32)
                else:
                    # Fallback: manual bit counting
                    hamming = torch.zeros_like(xor_vals, dtype=torch.float32)
                    temp = xor_vals.clone()
                    for _ in range(self.n_hash_buckets):
                        hamming += (temp & 1).float()
                        temp = temp >> 1

                # Compute cosine similarity from hamming distance
                cos_hamming = torch.cos(hamming / self.n_hash_buckets).to(dtype)  # [num_occupied, num_occupied]

                # Compute weighted cosine similarity (sparse): [B*H, num_occupied]
                avg_cosine_sparse = (counts_sparse @ cos_hamming - counts_sparse) / (total_counts + 1e-8)

                # Map back to original tokens: [B*H, Q]
                redundancy_flat = avg_cosine_sparse.gather(1, codes_idx)  # [B*H, Q]

            # Reshape back to [B, H, Q]
            redundancy = redundancy_flat.view(bsz, num_heads, q_len)
            redundancy = F.softmax(redundancy, dim=-1, dtype=torch.bfloat16).to(keys.dtype)

        # Combine scores based on lambda
        if self.lam == 0:
            # Only redundancy (skip attention entirely)
            final_scores = redundancy
        elif self.lam == 1:
            # Only attention scores (skip redundancy entirely)
            final_scores = scores
        else:
            # Combination of both
            final_scores = self.lam * scores + (1 - self.lam) * redundancy

        # Store component scores for qualitative analysis
        if self.enable_qualitative_analysis:
            # Store both attention and redundancy scores (even if one is None)
            # This allows comparison between RKV (lam=1) and RKV-LSH (lam<1)
            self._last_attention_scores = scores.detach().cpu() if scores is not None else None
            self._last_redundancy_scores = redundancy.detach().cpu() if redundancy is not None else None
            self._last_final_scores = final_scores.detach().cpu()

        # Add back the observation window. Use max score to make sure the window is not pruned.
        # Keep max computation on GPU, only convert to Python scalar for padding value (required by F.pad)
        max_score = final_scores.max()
        final_scores = F.pad(final_scores, (0, self.window_size), value=float(max_score))
        return final_scores
    

    def _get_hidden_size(self, module, device="cuda"):
        """Get hidden size based on model type."""
        if self.hidden_size is None:
            # Detect model type from config
            model_type = getattr(module.config, 'model_type', '').lower()
            model_name = getattr(module.config, 'name_or_path', '').lower()
            
            # Check for llama3 models
            if 'llama' in model_type or 'llama' in model_name or 'nemotron' in model_name:
                self.hidden_size = 4096
            # Check for qwen-7b models
            elif 'qwen' in model_type or 'qwen' in model_name:
                if '7b' in model_name or '7b' in str(getattr(module.config, 'hidden_size', 0)):
                    self.hidden_size = 3584
                else:
                    # Default for other Qwen models
                    self.hidden_size = getattr(module.config, 'hidden_size', 4096)
            else:
                # Default: use config hidden_size or fallback to 4096
                self.hidden_size = getattr(module.config, 'hidden_size', 4096)
        
        # Always ensure acc_hidden_states exists and is on correct device
        if self.acc_hidden_states is None or self.acc_hidden_states.device != device:
            self.acc_hidden_states = torch.zeros(
                (1, self.compress_interval, self.hidden_size), dtype=torch.bfloat16, device=device
            )
        
        return self.hidden_size

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
        layer_idx = getattr(module, "layer_idx", 0)
        if self.cache_budget >= kv_len:
            # All tokens retained, track if needed
            # Only track if tokenizer is set (track_tokens == True)
            if layer_idx == 0:
                if self.tokenizer is not None and self.input_tokens is not None:
                    if kv_len <= len(self.input_tokens):
                        all_token_ids = self.input_tokens[:kv_len].cpu().tolist()
                        retained_token_ids = all_token_ids.copy()
                    else:
                        all_token_ids = self.input_tokens.cpu().tolist() + list(range(len(self.input_tokens), kv_len))
                        retained_token_ids = all_token_ids.copy()
                    self.track_generation_step(all_token_ids, retained_token_ids, self.tokenizer)
            return keys, values
        
        # Initialize hidden size if not set
        device = hidden_states.device
        self._get_hidden_size(module, device=device)
        
        if self.accumulated_tokens < self.compress_interval:
            if getattr(module, "layer_idx", -1) == 0:
                self.accumulated_tokens += 1
            # # print(f"[DEBUG] hidden_states shape: {hidden_states.shape}, acc_hidden_states shape: {self.acc_hidden_states.shape}, accumulated_tokens: {self.accumulated_tokens}")
            self.acc_hidden_states[:, self.accumulated_tokens - 1, :] = hidden_states
            return keys, values

        # Compute scores using LSH algorithm
        scores = self.score(module, self.acc_hidden_states[:, -self.window_size:, :], keys, values, attentions, False, kwargs)
        # Get indices of KV pairs with the lowest scores
        indices = scores.topk(self.cache_budget, dim=-1).indices
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)

        # Track token retention/eviction at first layer only
        # Only track if tokenizer is set (track_tokens == True)
        if layer_idx == 0:
            if self.tokenizer is not None and self.input_tokens is not None:
                # Map position indices to actual token IDs
                # CPU transfer only happens here for token tracking - computation stays on GPU
                if kv_len <= len(self.input_tokens):
                    all_token_ids = self.input_tokens[:kv_len].cpu().tolist()
                    retained_positions = indices[0, 0, :, 0].cpu().tolist()  # Get retained position indices
                    retained_token_ids = [all_token_ids[pos] for pos in retained_positions]
                else:
                    # If kv_len > input_tokens, we have generated tokens
                    all_token_ids = self.input_tokens.cpu().tolist() + list(range(len(self.input_tokens), kv_len))
                    retained_positions = indices[0, 0, :, 0].cpu().tolist()
                    retained_token_ids = [all_token_ids[pos] if pos < len(self.input_tokens) else pos for pos in retained_positions]
                self.track_generation_step(all_token_ids, retained_token_ids, self.tokenizer)

            # Log qualitative decisions if enabled
            self.log_qualitative_decisions(indices, kv_len)

        # Prune keys and values
        keys = keys.gather(2, indices).contiguous()
        values = values.gather(2, indices).contiguous()
        # remove nan in keys and values
        keys = torch.nan_to_num(keys, nan=0.0)  
        values = torch.nan_to_num(values, nan=0.0)
        
        if layer_idx == 0:
            self.accumulated_tokens = 0  # Reset after compression
            device = hidden_states.device
            self.acc_hidden_states = torch.zeros(
                (1, self.compress_interval, self.hidden_size), dtype=torch.bfloat16, device=device
            ) # Reset accumulated hidden states

        # Save ranking data ONLY if tokenizer is set (track_tokens == True)
        # When track_tokens == False, tokenizer is None, so no ranking data is saved
        if self.tokenizer is not None:
            self.save_ranking_data(scores, indices, kv_len, False)

        return keys, values

    def set_tokenizer_and_tokens(self, tokenizer, input_tokens):
        """Set tokenizer and input tokens for text decoding."""
        self.tokenizer = tokenizer
        self.input_tokens = input_tokens

    def log_qualitative_decisions(self, indices, kv_len):
        """
        Log detailed token retention/eviction decisions.

        For each eviction step, logs:
        - All tokens in the cache (position, text, token_id)
        - Which tokens were retained vs evicted
        - Attention scores (if lam > 0)
        - Redundancy scores (if lam < 1)
        - Final combined scores used for decision

        This allows analysis of what kinds of tokens each method drops.
        """
        if not self.enable_qualitative_analysis or self.tokenizer is None:
            return

        # Get all token IDs in the current cache
        if kv_len <= len(self.input_tokens):
            all_token_ids = self.input_tokens[:kv_len].cpu().tolist()
        else:
            all_token_ids = self.input_tokens.cpu().tolist() + list(range(len(self.input_tokens), kv_len))

        # Get which positions were retained (kept in cache)
        retained_positions = set(indices[0, 0, :, 0].cpu().tolist())
        all_positions = list(range(kv_len))

        # Get scores (excluding window which is always kept)
        final_scores = self._last_final_scores[0, 0, :-self.window_size].numpy()
        attention_scores = self._last_attention_scores[0, 0, :].numpy() if self._last_attention_scores is not None else None
        redundancy_scores = self._last_redundancy_scores[0, 0, :].numpy() if self._last_redundancy_scores is not None else None

        # Build detailed token list (JSON only stores IDs, not text to keep file small)
        all_tokens = []
        retained_tokens = []
        evicted_tokens = []

        for pos in all_positions:
            token_info = {
                'position': pos,
                'token_id': all_token_ids[pos] if pos < len(all_token_ids) else None,
                'retained': pos in retained_positions,
                'final_score': float(final_scores[pos]),
                'attention_score': float(attention_scores[pos]) if attention_scores is not None else None,
                'redundancy_score': float(redundancy_scores[pos]) if redundancy_scores is not None else None,
            }

            all_tokens.append(token_info)
            if pos in retained_positions:
                retained_tokens.append(token_info)
            else:
                evicted_tokens.append(token_info)

        # Create log entry for this eviction step
        log_entry = {
            'sample_id': self.current_sample_id,
            'eviction_step': len(self.qualitative_data),  # Track which eviction step this is
            'kv_len': kv_len,
            'cache_budget': self.cache_budget,
            'lambda': self.lam,
            'n_hash_buckets': self.n_hash_buckets,
            'method': 'RKV' if self.lam == 1.0 else ('RKV-LSH' if self.lam == 0.0 else f'Hybrid(λ={self.lam})'),

            # All tokens with their status
            'all_tokens': all_tokens,

            # Separated lists for easier analysis
            'retained_tokens': retained_tokens,
            'evicted_tokens': evicted_tokens,

            # Summary statistics
            'num_total': len(all_positions),
            'num_retained': len(retained_tokens),
            'num_evicted': len(evicted_tokens),
        }

        self.qualitative_data.append(log_entry)

        # Print summary if verbose
        if len(self.qualitative_data) <= 3:  # Only print first few
            print(f"[Qualitative] Sample {self.current_sample_id}, Step {log_entry['eviction_step']}: "
                  f"Retained {len(retained_tokens)}/{kv_len} tokens")

    def save_qualitative_analysis(self, output_file=None):
        """
        Save qualitative analysis data to JSON file.

        Output format:
        - Each entry represents one eviction step
        - Contains all tokens with their retention status and scores
        - Includes separate lists of retained/evicted tokens for easy analysis
        """
        if not self.enable_qualitative_analysis:
            print("[RKV-LSH] Qualitative analysis not enabled, skipping save")
            return

        if len(self.qualitative_data) == 0:
            print("[RKV-LSH] No qualitative data collected")
            return

        if output_file is None:
            method_name = 'rkv' if self.lam == 1.0 else ('rkvlsh' if self.lam == 0.0 else f'hybrid_lam{self.lam}')
            output_file = f"token_decisions_{method_name}_budget{self.cache_budget}_buckets{self.n_hash_buckets}.json"

        output_path = os.path.join(self.save_dir, output_file)
        os.makedirs(self.save_dir, exist_ok=True)

        # Add metadata
        output_data = {
            'metadata': {
                'method': 'RKV' if self.lam == 1.0 else ('RKV-LSH' if self.lam == 0.0 else f'Hybrid(λ={self.lam})'),
                'lambda': self.lam,
                'n_hash_buckets': self.n_hash_buckets,
                'cache_budget': self.cache_budget,
                'total_samples': self.current_sample_id + 1,
                'total_eviction_steps': len(self.qualitative_data),
            },
            'eviction_steps': self.qualitative_data
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"[RKV-LSH] Qualitative analysis saved to: {output_path}")
        print(f"  - Total samples: {self.current_sample_id + 1}")
        print(f"  - Total eviction steps: {len(self.qualitative_data)}")

        # Also save a human-readable summary
        self._save_qualitative_summary(output_path.replace('.json', '_summary.txt'))

    def _save_qualitative_summary(self, summary_file):
        """
        Generate and save a human-readable summary.

        Shows examples of retained vs evicted tokens to help identify:
        - What kinds of tokens this method keeps
        - What kinds of tokens this method drops

        Decodes token IDs to text for readability.
        """
        if self.tokenizer is None:
            print("[RKV-LSH] Cannot generate summary without tokenizer")
            return

        with open(summary_file, 'w') as f:
            method_name = 'RKV' if self.lam == 1.0 else ('RKV-LSH' if self.lam == 0.0 else f'Hybrid(λ={self.lam})')

            f.write("=" * 80 + "\n")
            f.write(f"Token Retention/Eviction Analysis: {method_name}\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Method: {method_name}\n")
            f.write(f"Lambda: {self.lam}\n")
            f.write(f"N Hash Buckets: {self.n_hash_buckets}\n")
            f.write(f"Cache Budget: {self.cache_budget}\n")
            f.write(f"Total Eviction Steps: {len(self.qualitative_data)}\n\n")

            # Show first few eviction steps in detail
            for i, entry in enumerate(self.qualitative_data[:5], 1):
                f.write("\n" + "=" * 80 + "\n")
                f.write(f"Eviction Step {i}: Sample {entry['sample_id']}, Step {entry['eviction_step']}\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Total tokens: {entry['num_total']}\n")
                f.write(f"Retained: {entry['num_retained']}\n")
                f.write(f"Evicted: {entry['num_evicted']}\n\n")

                # Sort retained by score (highest first)
                retained = sorted(entry['retained_tokens'], key=lambda x: x['final_score'], reverse=True)
                # Sort evicted by score (lowest first)
                evicted = sorted(entry['evicted_tokens'], key=lambda x: x['final_score'])

                # Show examples of RETAINED tokens
                f.write("-" * 80 + "\n")
                f.write("RETAINED TOKENS (Top 15 by score):\n")
                f.write("-" * 80 + "\n")
                for token in retained[:15]:
                    # Decode token ID to text
                    token_text = self.tokenizer.decode([token['token_id']], skip_special_tokens=False) if token['token_id'] is not None else "<UNK>"
                    f.write(f"[Pos {token['position']:4d}] ID:{token['token_id']:6d} '{token_text[:30]:<30}' | Score: {token['final_score']:.4f}")
                    if token['attention_score'] is not None:
                        f.write(f" | Attn: {token['attention_score']:.4f}, Red: {token['redundancy_score']:.4f}")
                    f.write("\n")

                # Show examples of EVICTED tokens
                f.write("\n" + "-" * 80 + "\n")
                f.write("EVICTED TOKENS (Top 15 by score - these had lowest scores):\n")
                f.write("-" * 80 + "\n")
                for token in evicted[:15]:
                    # Decode token ID to text
                    token_text = self.tokenizer.decode([token['token_id']], skip_special_tokens=False) if token['token_id'] is not None else "<UNK>"
                    f.write(f"[Pos {token['position']:4d}] ID:{token['token_id']:6d} '{token_text[:30]:<30}' | Score: {token['final_score']:.4f}")
                    if token['attention_score'] is not None:
                        f.write(f" | Attn: {token['attention_score']:.4f}, Red: {token['redundancy_score']:.4f}")
                    f.write("\n")

            # Overall statistics
            f.write("\n\n" + "=" * 80 + "\n")
            f.write("Overall Statistics\n")
            f.write("=" * 80 + "\n")
            total_retained = sum(entry['num_retained'] for entry in self.qualitative_data)
            total_evicted = sum(entry['num_evicted'] for entry in self.qualitative_data)
            f.write(f"Total tokens retained across all steps: {total_retained}\n")
            f.write(f"Total tokens evicted across all steps: {total_evicted}\n")

        print(f"[RKV-LSH] Summary saved to: {summary_file}")

    def save_ranking_data(self, scores, indices, kv_len, is_prefill):
        """Save ranking data for analysis. Only called when track_tokens=True (tokenizer is set)."""
        # Double-check: only save if tokenizer is set (tracking enabled)
        if self.tokenizer is None:
            return
        # Create directory only when needed (first time saving)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
        try:
            # Only move to CPU if we're actually saving ranking data
            # Keep operations on GPU as much as possible - only convert when needed for numpy
            # Use detach() to avoid gradient tracking, but keep on GPU until final conversion
            scores_detached = scores.detach()
            indices_detached = indices.detach()
            # Convert to numpy only at the last moment (CPU transfer happens here)
            scores_np = scores_detached.cpu().float().numpy().flatten()
            indices_np = indices_detached.cpu().float().numpy().flatten()
            
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
                        'index': int(i),
                        'text': str(token_text),
                        'score': float(scores_np[i]) if i < len(scores_np) else 0.0
                    })
                
                # Get top 10 tokens (highest scores)
                top_10_indices = rankings[:10]
                for idx in top_10_indices:
                    if idx < len(self.input_tokens):
                        token_text = self.tokenizer.decode([self.input_tokens[idx]], skip_special_tokens=True)
                        top_10_tokens.append({
                            'index': int(idx),
                            'text': str(token_text),
                            'score': float(scores_np[idx])
                        })
                
                # Get bottom 10 tokens (lowest scores)
                bottom_10_indices = rankings[-10:]
                for idx in bottom_10_indices:
                    if idx < len(self.input_tokens):
                        token_text = self.tokenizer.decode([self.input_tokens[idx]], skip_special_tokens=True)
                        bottom_10_tokens.append({
                            'index': int(idx),
                            'text': str(token_text),
                            'score': float(scores_np[idx])
                        })
            
            # Create ranking entry
            ranking_entry = {
                'scores': scores_np.astype(float).tolist(),
                'rankings': rankings.astype(int).tolist(),
                'selected_indices': indices_np.astype(int).tolist(),
                'sequence_length': int(kv_len),
                'cache_budget': int(self.cache_budget),
                'is_prefill': bool(is_prefill),
                'compression_ratio': float(self.compression_ratio),
                'token_texts': token_texts,
                'top_10_tokens': top_10_tokens,
                'bottom_10_tokens': bottom_10_tokens
            }
            
            # Add to ranking data
            self.ranking_data.append(ranking_entry)
            
            # Save individual ranking data
            class_name = self.__class__.__name__.lower()
            ranking_file = os.path.join(self.save_dir, f"ranking_data_{class_name}_budget{self.cache_budget}.json")
            with open(ranking_file, 'w') as f:
                json.dump(ranking_entry, f, indent=2)
                
        except Exception as e:
            print(f"Error saving ranking data: {e}")
    
    def save_all_ranking_data(self, filename=None):
        """Save all collected ranking data to a single file. Only called when track_tokens=True."""
        # Double-check: only save if tokenizer is set (tracking enabled)
        if self.tokenizer is None:
            return
        # Create directory only when needed
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
        try:
            if filename is None:
                class_name = self.__class__.__name__.lower()
                filename = f"all_ranking_data_{class_name}_budget{self.cache_budget}.json"
            output_file = os.path.join(self.save_dir, filename)
            with open(output_file, 'w') as f:
                json.dump(self.ranking_data, f, indent=2)
            print(f"All ranking data saved to: {output_file}")
        except Exception as e:
            print(f"Error saving all ranking data: {e}")
    
    def reset_ranking_data(self):
        """Reset collected ranking data."""
        self.ranking_data = []
