"""
Operating systems use paging to manage virtual memory to get around storing everything contiguously,
and PagedAttention does that same for the KV cache. Instead of storing KV vectors in one contiguous memory
block, divides into small blocks and maintains a block manager.
"""
import math
import torch
from contextlib import contextmanager
from kvpress.presses.base_press import BasePress

class PagedAttentionPress(BasePress):
    def __init__(self, block_size: int = 256, max_blocks: int = 40960, offload_threshold: int = 10, device: str = 'cuda'):
        """
        Initialize the PagedAttentionPress compressor.
        """
        self.block_size = block_size # stores 256 tokens per block
        self.max_blocks = max_blocks
        self.offload_threshold = offload_threshold
        self.device = device

        # internal storage for kv pages. each page is indexed and stored separately
        self.k_pages = [None] * max_blocks
        self.v_pages = [None] * max_blocks

        # maps each batch to a list of (page index, filled length)
        self.block_table = {}  # e.g., {batch_idx: [(page_idx, filled), ...], ...}
        # list of available pages
        self.free_pages = list(range(max_blocks))
        # keeps offloaded pages in cpu memory
        self.offloaded_pages = {}
    
    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        """
        Reorganize the KV cache into fixed-size pages and update the block table.
        """
        batch_size, num_heads, seq_len, head_dim = keys.shape  # 8 attn heads, 128 dimension
        num_blocks_per_seq = math.ceil(seq_len / self.block_size)
        new_block_table = {}

        paged_keys_list = []
        paged_values_list = []

        # Process each batch separately
        for b in range(batch_size):
            new_block_table[b] = []
            batch_k_blocks = []
            batch_v_blocks = []

            for i in range(num_blocks_per_seq):
                start_idx = i * self.block_size
                end_idx = min((i + 1) * self.block_size, seq_len)

                key_block = keys[b, :, start_idx:end_idx, :].clone()  # Now keeps heads
                value_block = values[b, :, start_idx:end_idx, :].clone()
                filled = end_idx - start_idx

                # **Ensure fixed-size blocks**
                if filled < self.block_size:
                    pad_size = self.block_size - filled
                    pad_k = torch.zeros((num_heads, pad_size, head_dim), dtype=key_block.dtype, device=key_block.device)
                    pad_v = torch.zeros((num_heads, pad_size, head_dim), dtype=value_block.dtype, device=value_block.device)
                    
                    key_block = torch.cat([key_block, pad_k], dim=1)  # Pad sequence dim
                    value_block = torch.cat([value_block, pad_v], dim=1)

                assert key_block.shape[1] == self.block_size, f"Key block has shape {key_block.shape} instead of (num_heads, {self.block_size}, {head_dim})"
                assert value_block.shape[1] == self.block_size, f"Value block has shape {value_block.shape} instead of (num_heads, {self.block_size}, {head_dim})"

                if self.free_pages:
                    page_idx = self.free_pages.pop(0)  # Allocate a free page
                else:
                    page_idx = self.offload_oldest_page()  # Offload if necessary

                # Move KV data to the correct device
                key_block = key_block.to(self.device)
                value_block = value_block.to(self.device)

                # Store in paged cache
                self.k_pages[page_idx] = key_block
                self.v_pages[page_idx] = value_block
                new_block_table[b].append((page_idx, filled))

                batch_k_blocks.append(key_block)
                batch_v_blocks.append(value_block)

            # Concatenate across all blocks for this batch
            paged_keys_list.append(torch.cat(batch_k_blocks, dim=1))
            paged_values_list.append(torch.cat(batch_v_blocks, dim=1))

        self.block_table = new_block_table
        self.offload_pages()

        # Convert list back to tensor
        paged_keys = torch.stack(paged_keys_list, dim=0)
        paged_values = torch.stack(paged_values_list, dim=0)

        return paged_keys, paged_values


    def offload_pages(self):
        """
        Offload pages to CPU memory if their page index exceeds the offload threshold.
        """
        for page_idx, page in enumerate(self.k_pages):
            if page is not None and page.device.type == 'cuda' and page_idx >= self.offload_threshold:
                #  move data from the GPU to the CPU
                self.offloaded_pages[page_idx] = page.cpu()
                self.k_pages[page_idx] = page.cpu()
                self.v_pages[page_idx] = self.v_pages[page_idx].cpu()

    def fetch_page(self, page_idx: int):
        """
        Fetch a page from CPU offloaded storage back to GPU.
        """
        if page_idx in self.offloaded_pages:
            page = self.offloaded_pages.pop(page_idx)
            page = page.to(self.device)
            self.k_pages[page_idx] = page
            return page
        else:
            return self.k_pages[page_idx]

    def offload_oldest_page(self):
        """
        Offload the oldest (or least-recently used) page to free up a free page slot.
        """
        for idx in range(self.offload_threshold, self.max_blocks):
            if idx not in self.offloaded_pages and self.k_pages[idx] is not None:
                self.k_pages[idx] = self.k_pages[idx].cpu()
                self.v_pages[idx] = self.v_pages[idx].cpu()

                self.offloaded_pages[idx] = self.k_pages[idx]
                self.free_pages.append(idx) # set page back to available
                return idx

        raise Exception("No page available for offloading to cpu, try increasing max_blocks.")