# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
import os
import csv
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator
from time import time
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

logger = logging.getLogger(__name__)


@dataclass
class BasePress:
    """
    Base class for all KV cache compression methods.
    The `forward_hook` method is called after the forward pass of an attention layer to update the cache.
    """

    def __post_init__(self):
        """Initialize timing tracking attributes"""
        self.prefill_time: float = 0.0
        self.decoding_time: float = 0.0
        self.total_prefill_tokens: int = 0
        self.total_decoding_tokens: int = 0
        self.latency: bool = False
        self.debug: bool = False
        self.csv_path: str = ""
        
        # Token tracking attributes
        self.tokenizer = None
        self.input_tokens = None
        self.retained_token_indices: list = []  # List of sets of retained indices per compression step
        self.all_token_indices: list = []  # List of all token indices before compression
        
        # Per-step token tracking during generation
        self.generation_steps: list = []  # List of dicts with step info
        self.current_generation_step: int = 0
        self.previous_cache_tokens: set = set()  # Track tokens in KV cache at end of previous step
        
        # Track cache position → sequence position mapping
        # This persists across compressions to know which seq position each cache pos corresponds to
        self.cache_to_seq_pos: list = []  # cache_to_seq_pos[cache_idx] = sequence_position
        self.next_seq_pos: int = 0  # Next sequence position for newly generated tokens

    def reset_timing(self):
        """Reset timing counters and token tracking state"""
        self.prefill_time = 0.0
        self.decoding_time = 0.0
        self.total_prefill_tokens = 0
        self.total_decoding_tokens = 0
        # Reset token tracking
        self.retained_token_indices = []
        self.all_token_indices = []
        self.generation_steps = []
        self.current_generation_step = 0
        self.previous_cache_tokens = set()
        # Track original input positions through compressions
        # Maps: original_position -> current_position_in_cache
        self.original_to_current_pos = {}
        self.tracking_initialized = False  # Flag to track first compression
        # Track cache position → sequence position mapping
        self.cache_to_seq_pos = []
        self.next_seq_pos = 0

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

    def set_tokenizer_and_tokens(self, tokenizer, input_tokens):
        """Set tokenizer and input tokens for text decoding and tracking."""
        self.tokenizer = tokenizer
        self.input_tokens = input_tokens

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
    
    def track_generation_step(self, all_token_ids: list, retained_token_ids: list, tokenizer=None, 
                               scores: list = None, retained_positions: list = None):
        """
        Track token retention/eviction at each generation step during decoding.
        Early return if tokenizer is None (tracking disabled).
        
        Parameters
        ----------
        all_token_ids : list
            All token IDs in the KV cache at this step (before compression)
        retained_token_ids : list
            Token IDs that were retained in the KV cache (after compression)
        tokenizer : optional
            Tokenizer to decode tokens to text
        scores : list, optional
            Importance scores for each token position (used for heatmap visualization)
        retained_positions : list, optional
            Position indices of retained tokens (maps to scores)
        """
        # Early return if tokenizer is None (tracking disabled)
        if tokenizer is None:
            return
        
        # Get input token count for tracking original positions
        input_len = len(self.input_tokens) if self.input_tokens is not None else 0
        kv_len = len(all_token_ids)
        
        # Initialize cache_to_seq_pos on FIRST compression
        if not self.tracking_initialized:
            # Initially, cache positions 0 to kv_len-1 map to sequence positions 0 to kv_len-1
            self.cache_to_seq_pos = list(range(kv_len))
            self.next_seq_pos = kv_len  # Next generated token will be at this seq position
            self.original_to_current_pos = {i: i for i in range(input_len)}
            self.tracking_initialized = True
        else:
            # Before compression: cache has grown by some tokens since last compression
            # Extend cache_to_seq_pos for new tokens added since last compression
            current_cache_len = len(self.cache_to_seq_pos)
            if kv_len > current_cache_len:
                # Add new sequence positions for newly generated tokens
                for _ in range(kv_len - current_cache_len):
                    self.cache_to_seq_pos.append(self.next_seq_pos)
                    self.next_seq_pos += 1
        
        # Get sequence positions for all current cache positions
        seq_positions = self.cache_to_seq_pos[:kv_len]  # Sequence position for each cache position
        
        # Track which original INPUT positions survive this compression
        retained_set = set(retained_positions) if retained_positions else set()
        
        # Create mapping from current position to new position after compression
        sorted_retained = sorted(retained_positions) if retained_positions else []
        current_to_new_pos = {pos: idx for idx, pos in enumerate(sorted_retained)}
        
        # Update original_to_current_pos: which original input positions survive?
        surviving_originals = {}
        for orig_pos, curr_pos in self.original_to_current_pos.items():
            if curr_pos in current_to_new_pos:
                surviving_originals[orig_pos] = current_to_new_pos[curr_pos]
        
        evicted_original_positions = [
            orig_pos for orig_pos in self.original_to_current_pos.keys() 
            if orig_pos not in surviving_originals
        ]
        original_input_retained = sorted(surviving_originals.keys())
        self.original_to_current_pos = surviving_originals
        
        # Count evicted vs retained
        num_evicted = kv_len - len(retained_positions) if retained_positions else 0
        
        # Build step info
        step_info = {
            'step': self.current_generation_step,
            'phase': 'decoding',
            'note': f'Decoding step: {kv_len} tokens before compression, {len(retained_positions) if retained_positions else 0} retained',
            'kv_len_before_compression': kv_len,
            'num_retained': len(retained_positions) if retained_positions else 0,
            'num_evicted': num_evicted,
            'input_length': input_len,
            'original_input_positions_retained': original_input_retained,
            'original_input_positions_evicted_this_step': evicted_original_positions,
            # Store sequence positions for each cache position (for post-hoc token ID resolution)
            'cache_to_seq_positions': seq_positions.copy(),
            'retained_positions': list(retained_positions) if retained_positions else [],
        }
        
        # Add importance scores if provided (for heatmap visualization)
        if scores is not None:
            step_info['importance_scores'] = scores.copy()
        
        # Add original input tokens info
        if self.input_tokens is not None:
            step_info['original_input_tokens'] = self.input_tokens[:input_len].tolist()
            step_info['original_input_tokens_text'] = [
                tokenizer.decode([tid], skip_special_tokens=True) if isinstance(tid, int) else str(tid) 
                for tid in self.input_tokens[:input_len].tolist()
            ]
        
        self.generation_steps.append(step_info)
        self.current_generation_step += 1
        
        # Update cache_to_seq_pos for AFTER compression
        # Only retained positions survive, in sorted order → new positions 0, 1, 2, ...
        new_cache_to_seq = [seq_positions[pos] for pos in sorted_retained]
        self.cache_to_seq_pos = new_cache_to_seq
    
    def _track_prefilling_step(self, module: nn.Module, keys: torch.Tensor):
        """Track a prefilling step."""
        # Early return if tokenizer or input_tokens is None (tracking disabled)
        if self.tokenizer is None or self.input_tokens is None:
            return
        kv_len = keys.shape[2]
        
        # For prefilling, track that we're in prefilling phase
        if kv_len <= len(self.input_tokens):
            all_token_ids = self.input_tokens[:kv_len].cpu().tolist()
        else:
            all_token_ids = self.input_tokens.cpu().tolist() + list(range(len(self.input_tokens), kv_len))
        
        step_info = {
            'step': self.current_generation_step,
            'phase': 'prefilling',
            'note': 'Prefilling phase - processing input tokens',
            'all_tokens': all_token_ids.copy(),
            'retained_tokens': all_token_ids.copy(),  # All tokens retained during prefilling
            'evicted_tokens': [],
            'newly_added_tokens': all_token_ids.copy(),  # All tokens are newly added during prefilling
            'previous_cache_tokens': [],
            'num_evicted': 0,
            'num_newly_added': len(all_token_ids)
        }
        
        step_info['all_tokens_text'] = [self.tokenizer.decode([tid], skip_special_tokens=True) if isinstance(tid, int) else str(tid) for tid in all_token_ids]
        step_info['retained_tokens_text'] = step_info['all_tokens_text'].copy()
        step_info['evicted_tokens_text'] = []
        step_info['newly_added_tokens_text'] = step_info['all_tokens_text'].copy()
        step_info['previous_cache_tokens_text'] = []
        
        self.generation_steps.append(step_info)
        self.current_generation_step += 1
        
        # Update previous cache tokens for next step
        self.previous_cache_tokens = set(all_token_ids)
    
    def _track_decoding_step(self, module: nn.Module, keys: torch.Tensor):
        """Track a decoding step - called once per token generation at layer 0."""
        # Early return if tokenizer or input_tokens is None (tracking disabled)
        if self.tokenizer is None or self.input_tokens is None:
            return
        kv_len = keys.shape[2]
        
        # Get all tokens in cache before compression
        if kv_len <= len(self.input_tokens):
            all_token_ids = self.input_tokens[:kv_len].cpu().tolist()
        else:
            # Input tokens + generated tokens
            all_token_ids = self.input_tokens.cpu().tolist() + list(range(len(self.input_tokens), kv_len))
        
        step_info = {
            'step': self.current_generation_step,
            'phase': 'decoding',
            'all_tokens_before_compression': all_token_ids.copy(),
            'previous_cache_tokens': list(self.previous_cache_tokens),
            'retained_tokens': all_token_ids.copy(),  # Default: all retained if no compression
            'evicted_tokens': [],
            'newly_added_tokens': list(set(all_token_ids) - self.previous_cache_tokens),
            'num_evicted': 0,
            'num_newly_added': len(set(all_token_ids) - self.previous_cache_tokens),
            'note': f'Decoding step: {len(set(all_token_ids) - self.previous_cache_tokens)} tokens newly retained, 0 tokens evicted'
        }
        
        step_info['all_tokens_before_compression_text'] = [self.tokenizer.decode([tid], skip_special_tokens=True) if isinstance(tid, int) else str(tid) for tid in all_token_ids]
        step_info['previous_cache_tokens_text'] = [self.tokenizer.decode([tid], skip_special_tokens=True) if isinstance(tid, int) else str(tid) for tid in self.previous_cache_tokens]
        step_info['retained_tokens_text'] = step_info['all_tokens_before_compression_text'].copy()
        step_info['evicted_tokens_text'] = []
        step_info['newly_added_tokens_text'] = [self.tokenizer.decode([tid], skip_special_tokens=True) if isinstance(tid, int) else str(tid) for tid in step_info['newly_added_tokens']]
        
        self.generation_steps.append(step_info)
        self.current_generation_step += 1
        
        # Update previous cache tokens for next step (will be updated again after compression if it happens)
        self.previous_cache_tokens = set(all_token_ids)
    
    def get_generation_steps(self) -> list:
        """Get all tracked generation steps"""
        # If tokenizer is not set, return empty list (tracking disabled)
        if self.tokenizer is None:
            return []
        return self.generation_steps.copy()
    
    def update_generated_tokens(self, all_token_ids: list, tokenizer):
        """
        Update generation_steps with actual generated token IDs and text.
        Call this after generation completes with the full sequence of token IDs.
        
        Uses cache_to_seq_positions to correctly map cache positions to sequence positions,
        accounting for position shifts due to compression.
        
        Parameters
        ----------
        all_token_ids : list
            Complete list of token IDs (input + generated)
        tokenizer : 
            Tokenizer to decode tokens
        """
        if not self.generation_steps or tokenizer is None:
            return
        
        for step_info in self.generation_steps:
            # Use cache_to_seq_positions to get correct sequence positions
            cache_to_seq = step_info.get('cache_to_seq_positions', [])
            if not cache_to_seq:
                continue
            
            # Map cache positions to actual token IDs using sequence positions
            all_token_ids_this_step = []
            for cache_pos, seq_pos in enumerate(cache_to_seq):
                if seq_pos < len(all_token_ids):
                    all_token_ids_this_step.append(all_token_ids[seq_pos])
                else:
                    # Fallback for tokens beyond what we have
                    all_token_ids_this_step.append(seq_pos)  # Use seq_pos as placeholder
            
            step_info['all_tokens_before_compression'] = all_token_ids_this_step
            
            # Update text representations
            step_info['all_tokens_text'] = [
                tokenizer.decode([tid], skip_special_tokens=True) if isinstance(tid, int) else str(tid) 
                for tid in all_token_ids_this_step
            ]
            
            # Update retained tokens based on retained_positions
            retained_positions = step_info.get('retained_positions', [])
            if retained_positions:
                retained_tokens = [all_token_ids_this_step[pos] for pos in retained_positions if pos < len(all_token_ids_this_step)]
                step_info['retained_tokens'] = retained_tokens
                step_info['retained_tokens_text'] = [
                    tokenizer.decode([tid], skip_special_tokens=True) if isinstance(tid, int) else str(tid) 
                    for tid in retained_tokens
                ]
            
            # Update evicted tokens
            retained_set = set(retained_positions) if retained_positions else set()
            evicted_positions = [i for i in range(len(all_token_ids_this_step)) if i not in retained_set]
            evicted_tokens = [all_token_ids_this_step[pos] for pos in evicted_positions]
            step_info['evicted_tokens'] = evicted_tokens
            step_info['evicted_tokens_text'] = [
                tokenizer.decode([tid], skip_special_tokens=True) if isinstance(tid, int) else str(tid) 
                for tid in evicted_tokens
            ]

    
    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Legacy compression method (supports both prefilling and decoding).
        If a subclass implements this, it will be used as a fallback for
        compress_prefilling and compress_decoding if they are not implemented.
        """
        raise NotImplementedError("Subclass must implement compress(), compress_prefilling(), or compress_decoding()")

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
        # Fallback to generic compress if specific prefill method is not implemented
        try:
            return self.compress(module, hidden_states, keys, values, attentions, kwargs)
        except NotImplementedError:
            raise NotImplementedError("compress_prefilling method (or compress) must be implemented in subclass")


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
        # Fallback to generic compress if specific decoding method is not implemented
        try:
            return self.compress(module, hidden_states, keys, values, attentions, kwargs)
        except NotImplementedError:
            raise NotImplementedError("compress_decoding method (or compress) must be implemented in subclass")
    
    def write_data(
        self,
        csv_path: str,
        prune_step: int,
        layer_idx: int,
        head_idx: int,
        kv_len_pre: int,
        attn_len: int,
        diff_indices: int,
        attn_pre: float | None = None,
        attn_post: float | None = None,
        evicted_positions: str | None = None,
    ) -> None:
        """Append one row to the attention-loss CSV, writing header if needed.

        Header columns:
        prune_step, layer_idx, head_idx, kv_len_pre, attn_len, diff_indices, attn_pre, attn_post
        """
        if not csv_path:
            csv_path = self.csv_path
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        file_exists = os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "prune_step",
                    "layer_idx",
                    "head_idx",
                    "kv_len_pre",
                    "attn_len",
                    "diff_indices",
                    "attn_pre",
                    "attn_post",
                    "evicted_positions",
                ])
            ap = f"{attn_pre:.6f}" if attn_pre is not None else ""
            po = f"{attn_post:.6f}" if attn_post is not None else ""
            row = [
                prune_step,
                layer_idx,
                head_idx,
                kv_len_pre,
                attn_len,
                diff_indices,
                ap,
                po,
                evicted_positions if evicted_positions is not None else "",
            ]
            if self.debug:
                debug_row = row[:-1]  # Exclude evicted_positions
                print(f"[CSV] {debug_row}")
            writer.writerow(row)


    
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
        hidden_states = kwargs["hidden_states"]
        # Handle both old and new key names for backward compatibility
        cache = kwargs.get("past_key_value") or kwargs.get("past_key_values")
        q_len = hidden_states.shape[1]

        is_prefilling = kwargs["cache_position"][-1] <= q_len

        if isinstance(cache, QuantizedCache):
            keys = cache._dequantize(cache._quantized_key_cache[module.layer_idx])
            values = cache._dequantize(cache._quantized_value_cache[module.layer_idx])
        else:
            keys = cache.key_cache[module.layer_idx]
            values = cache.value_cache[module.layer_idx]

        if self.latency:
            torch.cuda.synchronize()
            start = time()

        if self.debug:
            print(f"[ATTN] layer_idx={getattr(module, 'layer_idx', -1)} kv_len={keys.shape[2]} prefill={is_prefilling}")


        if is_prefilling:
            keys, values = self.compress_prefilling(module, hidden_states, keys, values, output[1], kwargs)
        else:
            keys, values = self.compress_decoding(module, hidden_states, keys, values, output[1], kwargs)

        # Always count tokens for progress (only once per layer 0)
        if getattr(module, "layer_idx", -1) == 0:
            if is_prefilling:
                self.total_prefill_tokens += q_len
            else:
                self.total_decoding_tokens += q_len

        if self.latency:
            torch.cuda.synchronize()
            execution_time = time() - start

            if is_prefilling:
                self.prefill_time += execution_time
            else:
                self.decoding_time += execution_time

        # Human-readable progress: report current prefill and decoding tokens (layer 0 only)
        if getattr(module, "layer_idx", -1) == 0:
            should_report = (
                self.debug
                or getattr(self, "progress_enabled", False)
                or self.latency
            )
            if should_report:
                label = getattr(self, "progress_label", "PROGRESS")
                callback = getattr(self, "progress_update", None)
                if is_prefilling:
                    if callable(callback):
                        callback(self.total_prefill_tokens, self.total_decoding_tokens, phase="prefill", label=label)
                    else:
                        print(f"[{label}] prefill_tokens={self.total_prefill_tokens} decoding_tokens={self.total_decoding_tokens}")
                else:
                    if callable(callback):
                        callback(self.total_prefill_tokens, self.total_decoding_tokens, phase="decoding", label=label)
                    else:
                        print(f"[{label}] prefill_tokens={self.total_prefill_tokens} decoding_tokens={self.total_decoding_tokens}")
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
        for layer in model.model.layers:
            layer.self_attn.rotary_emb = model.model.rotary_emb
            hooks.append(layer.self_attn.register_forward_hook(self.forward_hook, with_kwargs=True))
        yield
        for forward_hook in hooks:
            forward_hook.remove()
