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
        self.ranking_data = []
        self.save_dir = "ranking_analysis"
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Tokenizer for decoding tokens (will be set during inference)
        self.tokenizer = None
        self.input_tokens = None 

    def initialize_buckets(self, device=None):
        """
        Initialize cos_hamming_distance_bucket on the specified device.
        If device is None, uses CUDA if available, otherwise CPU.
        """
        # Determine device: use provided device, or CUDA if available, else CPU
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # initialize cos_hamming_distance_bucket on the specified device
        buckets = torch.arange(2**self.n_hash_buckets, device=device)
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
        
        # Optimized hashing projection: use matmul which is faster than einsum for this pattern
        # einsum("bhqd,dk->bhqk") is equivalent to matmul after reshaping
        # For small sequences, matmul is often faster than einsum
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

        # Cache cos_bucket device transfer - only move if device or dtype changed
        device_str = str(keys.device)
        dtype = keys.dtype  # Use the same dtype as keys (typically bfloat16)
        if (self.cos_bucket_cached is None or 
            self.cos_bucket_device != device_str or
            self.cos_bucket_dtype != dtype):
            self.cos_bucket_cached = self.cos_hamming_distance_bucket.to(keys.device).to(dtype)
            self.cos_bucket_device = device_str
            self.cos_bucket_dtype = dtype
        cos_bucket = self.cos_bucket_cached  # [2**n_hash_buckets, 2**n_hash_buckets]
        
        # Fully vectorized computation on GPU - no CPU transfers, no Python loops
        # Shape: hash_codes_int is [B, H, Q]
        bsz, num_heads, q_len = hash_codes_int.shape
        
        # Flatten for batch processing: [B*H, Q]
        codes_flat = hash_codes_int.view(-1, q_len).long()  # [B*H, Q]
        
        # Vectorized bucket counting using scatter_add (fully GPU-accelerated)
        # Count tokens in each bucket for each batch-head: [B*H, num_buckets]
        counts = torch.zeros(bsz * num_heads, self.num_buckets, device=codes_flat.device, dtype=torch.bfloat16)
        counts.scatter_add_(1, codes_flat, torch.ones_like(codes_flat, dtype=torch.bfloat16))
        
        # Compute total counts per batch-head: [B*H, 1]
        total_counts = counts.sum(dim=1, keepdim=True)  # [B*H, 1]
        
        # According to paper: S_i' = (Σ_{i≠j} c_j cos(Hamming(i,j)/b)) / (Σ_j c_j)
        # Optimized: compute (counts @ cos_bucket - counts) / total_counts in one step
        # This excludes self-similarity (diagonal terms where cos(0) = 1.0)
        # Combined operation to reduce memory traffic
        avg_cosine = (counts @ cos_bucket - counts) / (total_counts + 1e-8)  # [B*H, num_buckets]
        
        # Map each token's bucket code to its average cosine value
        # Use gather for efficient indexing
        redundancy_flat = avg_cosine.gather(1, codes_flat)  # [B*H, Q]
        
        # Reshape back to [B, H, Q]
        redundancy = redundancy_flat.view(bsz, num_heads, q_len)
        redundancy = F.softmax(redundancy, dim=-1, dtype=torch.bfloat16).to(scores.dtype)

        scores = self.lam * scores + (1 - self.lam) * redundancy
        # Add back the observation window. Use max score to make sure the window is not pruned.
        # Keep max computation on GPU, only convert to Python scalar for padding value (required by F.pad)
        max_score = scores.max()
        scores = F.pad(scores, (0, self.window_size), value=float(max_score))
        return scores
    

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
            
            # Initialize acc_hidden_states with correct size
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
            # Only do CPU transfer if tokenizer is set (tracking enabled)
            if layer_idx == 0 and self.tokenizer is not None:
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

        # Compute scores
        # scores = self.score(module, hidden_states, keys, values, attentions, False, kwargs)
        scores = self.score(module, self.acc_hidden_states[:, -self.window_size:, :], keys, values, attentions, False, kwargs)
        # Get indices of KV pairs with the lowest scores
        indices = scores.topk(self.cache_budget, dim=-1).indices
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)

        # Track token retention/eviction at first layer only
        # Only do CPU transfer if tokenizer is set (tracking enabled)
        if layer_idx == 0 and self.tokenizer is not None:
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

        # Prune keys and values
        keys = keys.gather(2, indices).contiguous()
        values = values.gather(2, indices).contiguous()
        # remove nan in keys and values
        keys = torch.nan_to_num(keys, nan=0.0)  
        values = torch.nan_to_num(values, nan=0.0)
        
        if layer_idx == 0:
            self.accumulated_tokens = 0  # Reset after compression
        self.acc_hidden_states = torch.zeros(
            (1, self.compress_interval, self.hidden_size), dtype=torch.bfloat16, device=device
        ) # Reset accumulated hidden states

        # Save ranking data only if tokenizer is set (tracking enabled)
        if self.tokenizer is not None:
            self.save_ranking_data(scores, indices, kv_len, False)

        return keys, values

    def set_tokenizer_and_tokens(self, tokenizer, input_tokens):
        """Set tokenizer and input tokens for text decoding."""
        self.tokenizer = tokenizer
        self.input_tokens = input_tokens

    def save_ranking_data(self, scores, indices, kv_len, is_prefill):
        """Save ranking data for analysis."""
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
        """Save all collected ranking data to a single file."""
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

