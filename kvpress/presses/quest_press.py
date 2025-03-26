"""
Operating systems use paging to manage virtual memory to get around storing everything contiguously,
and PagedAttention does that same for the KV cache. Instead of storing KV vectors in one contiguous memory
block, divides into small blocks and maintains a block manager.
"""
import math
import torch
from contextlib import contextmanager
from kvpress.presses.base_press import BasePress

class QuestPress(BasePress):
    def __init__(self, block_size: int = 256, device: str = 'cuda'):
        """
        Initialize the QuestPress compressor.
        """
        self.block_size = block_size # stores 256 tokens per block
        self.device = device


    
    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        """
        Reorganize the KV cache into fixed-size pages and update the block table.
        """
        if self.compression_ratio == 0:
            return keys, values
        
        batch_size, num_heads, seq_len, head_dim = keys.shape  # 8 attn heads, 128 dimension
        n_kept = int(seq_len * (1 - self.compression_ratio))
        blocks_kept = math.ceil(n_kept / self.block_size)


        # STEP 1: Divide the keys and values into blocks and keep track of the filled tokens
        # and the min and max key vector per block

        # the number of blocks in the sequence
        num_blocks = math.ceil(seq_len / self.block_size)
        last_block_size = seq_len % self.block_size

        # pad the last block if necessary
        if last_block_size < self.block_size:
            pad_size = self.block_size - last_block_size
            pad_k = torch.zeros((batch_size, num_heads, pad_size, head_dim), dtype=keys.dtype, device=keys.device)
            pad_v = torch.zeros((batch_size, num_heads, pad_size, head_dim), dtype=values.dtype, device=values.device)
            keys = torch.cat([keys, pad_k], dim=2)
            values = torch.cat([values, pad_v], dim=2)
        
        # divide the keys and values into blocks
        paged_keys = keys.view(batch_size, num_heads, num_blocks, self.block_size, head_dim)
        paged_values = values.view(batch_size, num_heads, num_blocks, self.block_size, head_dim)
        # store the number of filled tokens in each block
        filled_tokens = torch.ones((batch_size, num_heads, num_blocks), dtype=torch.int32, device=keys.device) * self.block_size
        # correct the last block size
        filled_tokens[:, :, -1] = last_block_size

        # calculate the channel wise min and max vector per page
        min_keys = paged_keys.min(dim=3).values # (batch_size, num_heads, num_blocks, head_dim)
        max_keys = paged_keys.max(dim=3).values # (batch_size, num_heads, num_blocks, head_dim)

        # TODO: do I need to correct the last block here since it is zero padded?

        # store the pages on cpu
        if not self.paged_keys_on_cpu:
            self.paged_keys_on_cpu = paged_keys.cpu()
            self.paged_values_on_cpu = paged_values.cpu()
        else:
            self.paged_keys_on_cpu = torch.cat([self.paged_keys_on_cpu, paged_keys.cpu()], dim=2)
            self.paged_values_on_cpu = torch.cat([self.paged_values_on_cpu, paged_values.cpu()], dim=2)
        
        # STEP 2: Decide which blocks to fetch from the cpu

        # calculate the query vector
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, head_dim)
        query_states = module.q_proj(hidden_states).view(hidden_shape).transpose(1, 2) # (batch_size, num_heads, seq_len, head_dim) 
        

        torch.mul(query_states, min_keys)



        

