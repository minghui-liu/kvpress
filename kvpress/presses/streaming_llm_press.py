# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass
import os
import csv

import torch
from torch import nn

from kvpress.presses.scorer_press import ScorerPress


@dataclass
class StreamingLLMPress(ScorerPress):
    """
    Prune a fixed number of KV pairs at the beginning and end of the sequence (https://arxiv.org/abs/2309.17453)
    We keep the first n_sink tokens and the last n_local tokens.
    n_local is computed using the compression ratio.

    Note that the original implementation https://github.com/mit-han-lab/streaming-llm additionally rerotates keys.
    This can be achieved by using
    press = KeyRerotationPress(press=StreamingLLMPress(compression_ratio, n_sink))
    """

    compression_ratio: float = 0.0
    cache_budget: int = 0
    n_sink: int = 4
    attn_csv_path: str = "attn_loss.csv"
    prune_step: int = 0
    output_attentions: bool = True

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
        
        if is_prefill:
            q_len = hidden_states.shape[1]
            assert q_len > self.n_sink, f"Input should contain more tokens than n_sink={self.n_sink}"
            n_pruned = q_len - self.cache_budget
            scores = torch.ones_like(keys[..., 0])
            scores[:, :, self.n_sink : self.n_sink + n_pruned] = 0
        else:
            # during generation, we keep the first n_sink tokens and the last n_local tokens
            n_local = self.cache_budget - self.n_sink
            scores = torch.zeros_like(keys[..., 0])
            scores[:, :, : self.n_sink] = 1
            scores[:, :, -n_local:] = 1

        # Debug prints similar to RKV/H2O
        full_len = scores.shape[-1]
        kept_len = scores.sum(dim=-1).int().min().item()  # minimum kept across batch/heads
        print("---" * 10)
        print(f"[DEBUG] (PRE) keys shape: {keys.shape}, values shape: {values.shape}")
        print(f"[DEBUG] diff indices: {full_len - kept_len}")


        return scores

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
        if self.cache_budget >= kv_len:
            return keys, values

        # Compute scores and select kept indices
        scores = self.score(module, hidden_states, keys, values, attentions, False, kwargs)
        indices = scores.topk(self.cache_budget, dim=-1).indices  # [B, Hkv, K]

        # Debug counts
        # try:
        #     print("---" * 10)
        #     print(f"[DEBUG] (PRE) keys shape: {keys.shape}, values shape: {values.shape}")
        #     full_len = scores.shape[-1]
        #     kept_len = indices.shape[2]
        #     print(f"[DEBUG] diff indices: {full_len - kept_len}")
        # except Exception:
        #     pass

        # If attentions are provided, compute attention mass removed using accumulated attentions (more stable)
        try:
            if attentions is not None:
                bsz, n_heads, _, q_len = attentions.shape
                n_kv_groups = module.num_key_value_groups
                n_kv_heads = n_heads // n_kv_groups

                # Initialize accumulators lazily
                if not hasattr(self, "acc_attn") or self.acc_attn is None:
                    # Start with zeros of length q_len for the first call
                    self.acc_attn = torch.zeros(bsz, n_kv_heads, q_len, device=attentions.device, dtype=attentions.dtype)
                    self.n_tokens_in_sum = torch.zeros_like(self.acc_attn)

                # Build per-key mass for this step and accumulate (sum over queries, average kv groups)
                step_sum = attentions.sum(2)  # [B, H, L]
                step_kv = step_sum.view(bsz, n_kv_heads, n_kv_groups, q_len).mean(2)  # [B, Hkv, L]

                # Align accumulator length if sequence grew
                if step_kv.shape[-1] > self.acc_attn.shape[-1]:
                    grow = step_kv.shape[-1] - self.acc_attn.shape[-1]
                    pad = torch.zeros(bsz, n_kv_heads, grow, device=attentions.device, dtype=attentions.dtype)
                    self.acc_attn = torch.cat([self.acc_attn, pad], dim=-1)
                    self.n_tokens_in_sum = torch.cat([self.n_tokens_in_sum, pad], dim=-1)

                self.acc_attn[:, :, : q_len] = self.acc_attn[:, :, : q_len] + step_kv
                self.n_tokens_in_sum[:, :, : q_len] = self.n_tokens_in_sum[:, :, : q_len] + 1

                # Average accumulated attention
                attn_kv = self.acc_attn[:, :, : q_len] / torch.clamp_min(self.n_tokens_in_sum[:, :, : q_len], 1)

                csv_path = self.attn_csv_path
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
                            "attn_loss",
                        ])
                    for head_idx in range(attn_kv.shape[1]):
                        head_attn = attn_kv[:, head_idx, :]  # [B, L]
                        kept_pos = indices[:, head_idx, :]    # [B, K]
                        pre_total = head_attn.sum(-1)         # [B]
                        kept_total = head_attn.gather(-1, kept_pos).sum(-1)  # [B]
                        loss_h = (pre_total - kept_total).sum()  # scalar over batch
                        row = [
                            self.prune_step,
                            getattr(module, "layer_idx", -1),
                            head_idx,
                            keys.shape[2],
                            q_len,
                            int(full_len - kept_len),
                            float(loss_h.item()),
                        ]
                        writer.writerow(row)
                        # print(f"[CSV] {row}")
                self.prune_step += 1
        except Exception as e:
            print(f"[WARN] StreamingLLM attn loss logging failed: {e}")

        # Gather pruned keys/values
        kv_indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)
        keys = keys.gather(2, kv_indices).contiguous()
        values = values.gather(2, kv_indices).contiguous()
        return keys, values
