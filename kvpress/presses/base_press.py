# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator

import torch
from torch import nn
from transformers import (
    LlamaForCausalLM,
    MistralForCausalLM,
    Phi3ForCausalLM,
    PreTrainedModel,
    QuantizedCache,
    Qwen2ForCausalLM,
)
from time import time

logger = logging.getLogger(__name__)


@dataclass
class BasePress:
    """
    Base class for all KV cache compression methods.
    The `forward_hook` method is called after the forward pass of an attention layer to update the cache.
    """
    
    def compress_prefilling(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        The core logic of the compression method during the pre-filling phase.

        Parameters
        ----------
        module :
            Transformer layer, see `hook` method for more details
        hidden_states :
            Hidden states of the layer
        keys :
            Keys of the cache (unquantized)
        values :
            Values of the cache (unquantized)
        attentions :
            Attention weights of the layer
        kwargs :
            Keyword arguments, as given to the forward pass of the layer

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Updated keys and values
        """
        raise NotImplementedError("compress method must be implemented in subclass")


    def compress_decoding(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        The core logic of the compression method during the decoding phase.
        Parameters
        ----------
        module :
            Transformer layer, see `hook` method for more details
        hidden_states :
            Hidden states of the layer
        keys :
            Keys of the cache (unquantized)
        values :
            Values of the cache (unquantized)
        attentions :
            Attention weights of the layer
        kwargs :
            Keyword arguments, as given to the forward pass of the layer  
        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Updated keys and values
        """
        raise NotImplementedError("compress method must be implemented in subclass")

    def __post_init__(self):
        """Initialize timing tracking attributes"""
        self.prefill_time: float = 0.0
        self.decoding_time: float = 0.0
        self.total_prefill_tokens: int = 0
        self.total_decoding_tokens: int = 0
        # Keyword tracking
        self.retained_token_indices: list = []  # List of sets of retained indices per compression step
        self.all_token_indices: list = []  # List of all token indices before compression
        
        # Per-step token tracking during generation
        self.generation_steps: list = []  # List of dicts with step info: {step, all_tokens, retained_tokens, evicted_tokens, newly_added_tokens}
        self.current_generation_step: int = 0
        self.previous_cache_tokens: set = set()  # Track what tokens were in the KV cache at the end of previous step (after compression)

    def reset_timing(self):
        """Reset timing counters"""
        self.prefill_time = 0.0
        self.decoding_time = 0.0
        self.total_prefill_tokens = 0
        self.total_decoding_tokens = 0
        # Reset keyword tracking
        self.retained_token_indices = []
        self.all_token_indices = []
        # Reset per-step tracking
        self.generation_steps = []
        self.current_generation_step = 0
        self.previous_cache_tokens = set()
    
    def track_retention(self, all_indices: list, retained_indices: list):
        """Track which tokens were retained in the cache"""
        self.all_token_indices.append(all_indices.copy())
        self.retained_token_indices.append(retained_indices.copy())
    
    def get_final_retained_indices(self) -> set:
        """Get the final set of retained token indices after all compressions"""
        if not self.retained_token_indices:
            return set()
        # Return the intersection of all retained indices (tokens that survived all compressions)
        final = set(self.retained_token_indices[0])
        for retained in self.retained_token_indices[1:]:
            final = final & set(retained)
        return final
    
    def track_generation_step(self, all_token_ids: list, retained_token_ids: list, tokenizer=None):
        """
        Track token retention/eviction at each generation step.
        
        During decoding:
        - Tokens are removed from KV cache during compression (evicted)
        - New tokens are added to KV cache (the newly generated token)
        
        Parameters
        ----------
        all_token_ids : list
            All token IDs in the KV cache at this step (before compression)
            This includes: previous_cache_tokens + newly generated token
        retained_token_ids : list
            Token IDs that were retained in the KV cache (after compression)
        tokenizer : optional
            Tokenizer to decode tokens to text
        """
        all_token_set = set(all_token_ids)
        current_retained_set = set(retained_token_ids)
        
        # Tokens that were in cache BEFORE compression = previous_cache + newly generated token
        # Tokens that were in cache AFTER compression = retained_token_ids
        
        # Compute tokens that were evicted (removed from cache during compression)
        # These are tokens that were in the cache before compression but not after
        evicted_token_ids = list(all_token_set - current_retained_set)
        
        # Compute tokens that were newly added to the cache
        # These are tokens in current retained set that weren't in previous cache
        # (typically the newly generated token that was added and retained)
        newly_added_token_ids = list(current_retained_set - self.previous_cache_tokens)
        
        # Also track what was in cache before compression for reference
        previous_cache_token_ids = list(self.previous_cache_tokens)
        
        step_info = {
            'step': self.current_generation_step,
            'previous_cache_tokens': previous_cache_token_ids.copy(),
            'all_tokens_before_compression': all_token_ids.copy(),
            'retained_tokens': retained_token_ids.copy(),
            'evicted_tokens': evicted_token_ids.copy(),
            'newly_added_tokens': newly_added_token_ids.copy()
        }
        
        # Decode tokens to text if tokenizer is available
        if tokenizer is not None:
            step_info['previous_cache_tokens_text'] = [tokenizer.decode([tid], skip_special_tokens=True) if isinstance(tid, int) else str(tid) for tid in previous_cache_token_ids]
            step_info['all_tokens_before_compression_text'] = [tokenizer.decode([tid], skip_special_tokens=True) if isinstance(tid, int) else str(tid) for tid in all_token_ids]
            step_info['retained_tokens_text'] = [tokenizer.decode([tid], skip_special_tokens=True) if isinstance(tid, int) else str(tid) for tid in retained_token_ids]
            step_info['evicted_tokens_text'] = [tokenizer.decode([tid], skip_special_tokens=True) if isinstance(tid, int) else str(tid) for tid in evicted_token_ids]
            step_info['newly_added_tokens_text'] = [tokenizer.decode([tid], skip_special_tokens=True) if isinstance(tid, int) else str(tid) for tid in newly_added_token_ids]
        
        # Update for next step: what's in cache now (after compression)
        self.previous_cache_tokens = current_retained_set.copy()
        
        self.generation_steps.append(step_info)
        self.current_generation_step += 1
    
    def get_generation_steps(self) -> list:
        """Get all tracked generation steps"""
        return self.generation_steps.copy()

    def get_timing_metrics(self):
        """Get timing metrics for performance analysis"""
        total_time = self.prefill_time + self.decoding_time
        output_tokens_per_second = self.total_decoding_tokens / self.decoding_time if self.decoding_time > 0 else 0.0
        
        return {
            "prefill_time": self.prefill_time,
            "decoding_time": self.decoding_time,
            "total_time": total_time,
            "total_prefill_tokens": self.total_prefill_tokens,
            "total_decoding_tokens": self.total_decoding_tokens,
            "output_tokens_per_second": output_tokens_per_second
        }
    
    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        """
        Default forward hook called after the forward pass of an attention layer.
        The hook calls the compress method to compress the KV cache while ensuring:
            - compression is only applied only during the pre-filling phase
            - KV cache quantization is handled correctly

        Parameters
        ----------
        module :
            Transformer attention layer.
        input :
            Input to the hook. This is the input to the forward pass of the layer.
        kwargs :
            Keyword arguments, as given to the forward pass of the layer.
        output :
            Output of the hook. This is the original output of the forward pass of the layer.

        Returns
        -------
            Modified output of the forward pass of the layer.

        """

        hidden_states = kwargs.get("hidden_states")
        if hidden_states is None:
            # If hidden_states is not in kwargs, this might be a different hook signature
            # Try to get from input
            if input and len(input) > 0:
                hidden_states = input[0]
            else:
                # Can't proceed without hidden_states
                return output
        
        cache = kwargs.get("past_key_value")
        if cache is None:
            # If past_key_value is not in kwargs, the cache might not be initialized yet
            # This can happen during the very first forward pass
            # Return output without compression
            return output
        
        q_len = hidden_states.shape[1]

        # Check if cache_position exists
        cache_position = kwargs.get("cache_position")
        if cache_position is not None:
            is_prefilling = cache_position[-1] <= q_len
        else:
            # If cache_position is not available, try to infer from cache state
            # During prefill, cache should be empty or growing
            if isinstance(cache, QuantizedCache):
                cache_len = cache._seen_tokens if hasattr(cache, '_seen_tokens') else 0
            else:
                cache_len = cache.get_seq_length() if hasattr(cache, 'get_seq_length') else 0
            is_prefilling = cache_len < q_len

        # Get keys and values from cache
        try:
            if isinstance(cache, QuantizedCache):
                keys = cache._dequantize(cache._quantized_key_cache[module.layer_idx])
                values = cache._dequantize(cache._quantized_value_cache[module.layer_idx])
            else:
                keys = cache.key_cache[module.layer_idx]
                values = cache.value_cache[module.layer_idx]
        except (KeyError, IndexError, AttributeError) as e:
            # Cache might not be properly initialized for this layer yet
            logger.warning(f"Could not access cache for layer {module.layer_idx}: {e}")
            return output

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        start=time()

        if is_prefilling:
            keys, values = self.compress_prefilling(module, hidden_states, keys, values, output[1], kwargs)
        else:
            keys, values = self.compress_decoding(module, hidden_states, keys, values, output[1], kwargs)

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        execution_time=time()-start

        # Track timing for prefill vs decoding phases
        if is_prefilling:
            self.prefill_time += execution_time
            self.total_prefill_tokens += q_len
        else:
            self.decoding_time += execution_time
            self.total_decoding_tokens += q_len

        # if is_prefilling:
        #     print(f"Prefilling {q_len} tokens took {execution_time:.2f} seconds")
        # else:
        #     print(f"Decoding {q_len} tokens took {execution_time:.2f} seconds")

        if isinstance(cache, QuantizedCache):
            cache._quantized_key_cache[module.layer_idx] = cache._quantize(keys, axis=cache.axis_key)
            cache._quantized_value_cache[module.layer_idx] = cache._quantize(values, axis=cache.axis_value)
            cache.key_cache[module.layer_idx] = torch.zeros(0, dtype=keys.dtype, device=keys.device)
            cache.value_cache[module.layer_idx] = torch.zeros(0, dtype=keys.dtype, device=keys.device)
            cache._seen_tokens = keys.shape[2]
        else:
            cache.key_cache[module.layer_idx] = keys
            cache.value_cache[module.layer_idx] = values

        return output

    @contextmanager
    def __call__(self, model: PreTrainedModel) -> Generator:
        """
        Context manager to apply a compression method to a model.
        Apply this context manager during the pre-filling phase to compress the context.

        Parameters
        ----------
        model : PreTrainedModel
            Model to apply the compression method to
        """

        if not isinstance(model, (LlamaForCausalLM, MistralForCausalLM, Phi3ForCausalLM, Qwen2ForCausalLM)):
            logger.warning(f"Model {type(model)} not tested")

        hooks = []
        try:
            for layer in model.model.layers:
                layer.self_attn.rotary_emb = model.model.rotary_emb
                hooks.append(layer.self_attn.register_forward_hook(self.forward_hook, with_kwargs=True))
            yield
        finally:
            for forward_hook in hooks:
                forward_hook.remove()
