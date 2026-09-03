#!/usr/bin/env python3
"""
Benchmark script to compare the performance of score() methods between RKVPress and RKVLSHPress.
"""

import torch
import time
import statistics
from torch import nn
from dataclasses import dataclass

from kvpress.presses.rkv_press import RKVPress
from kvpress.presses.rkv_press_lsh import RKVLSHPress


@dataclass
class MockConfig:
    """Mock model config for testing."""
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    hidden_size: int = 4096
    model_type: str = "llama"
    name_or_path: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"


class MockAttentionModule(nn.Module):
    """Mock attention module for testing."""
    def __init__(self, hidden_size: int = 4096, head_dim: int = 128, num_heads: int = 32):
        super().__init__()
        self.config = MockConfig()
        self.config.hidden_size = hidden_size
        self.head_dim = head_dim
        self.config.head_dim = head_dim
        self.layer_idx = 0
        # Create q_proj layer for compute_window_attention
        # q_proj projects hidden_size -> num_heads * head_dim
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)


def create_dummy_inputs(
    bsz: int = 1,
    num_key_value_heads: int = 8,
    q_len: int = 512,
    head_dim: int = 128,
    hidden_size: int = 4096,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    dtype: torch.dtype = torch.bfloat16,
):
    """Create dummy input tensors for benchmarking."""
    # Create keys and values: [bsz, num_key_value_heads, q_len, head_dim]
    keys = torch.randn(bsz, num_key_value_heads, q_len, head_dim, device=device, dtype=dtype)
    values = torch.randn(bsz, num_key_value_heads, q_len, head_dim, device=device, dtype=dtype)
    
    # Create hidden_states: [bsz, q_len, hidden_size]
    hidden_states = torch.randn(bsz, q_len, hidden_size, device=device, dtype=dtype)
    
    # Create position embeddings (cos, sin) for RoPE
    # Shape: [bsz, q_len, head_dim] for cos and sin (they need to match head_dim for broadcasting)
    # Note: In real RoPE, cos/sin are typically [bsz, q_len, head_dim//2], but they get
    # repeated/interleaved to match head_dim. For simplicity in benchmarking, we use full head_dim.
    cos = torch.randn(bsz, q_len, head_dim, device=device, dtype=dtype)
    sin = torch.randn(bsz, q_len, head_dim, device=device, dtype=dtype)
    position_embeddings = (cos, sin)
    
    # Create kwargs dict
    kwargs = {"position_embeddings": position_embeddings}
    
    # Attentions can be None (not used in these presses)
    attentions = None
    
    return keys, values, hidden_states, attentions, kwargs


def benchmark_score_method(
    press,
    module,
    keys,
    values,
    hidden_states,
    attentions,
    kwargs,
    num_warmup: int = 10,
    num_iterations: int = 100,
    num_runs: int = 3,
    device: str = "cuda",
):
    """Benchmark a single score() method with multiple runs for better statistics."""
    all_times = []
    
    for _ in range(num_runs):
        # Warmup runs for each run
        for _ in range(num_warmup):
            with torch.no_grad():
                _ = press.score(module, hidden_states, keys, values, attentions, False, kwargs)
        
        # Synchronize if using CUDA
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        
        # Actual timing for this run
        run_times = []
        for _ in range(num_iterations):
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            with torch.no_grad():
                _ = press.score(module, hidden_states, keys, values, attentions, False, kwargs)
            
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            run_times.append((end_time - start_time) * 1000)  # Convert to milliseconds
        
        all_times.extend(run_times)
    
    # Aggregate statistics across all runs
    return {
        "mean_ms": statistics.mean(all_times),
        "median_ms": statistics.median(all_times),
        "std_ms": statistics.stdev(all_times) if len(all_times) > 1 else 0.0,
        "min_ms": min(all_times),
        "max_ms": max(all_times),
        "times": all_times,
        "num_samples": len(all_times),
    }


def main():
    """Main benchmarking function."""
    print("=" * 80)
    print("Benchmarking score() methods: RKVPress vs RKVLSHPress")
    print("=" * 80)
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    
    print("\nDevice:", device)
    print("Dtype:", dtype)
    
    # Test configurations - focus on long sequences for reasoning models
    # Reasoning models typically generate very long sequences (thousands of tokens)
    test_configs = [
        {"bsz": 1, "q_len": 2048, "name": "Long (2048 tokens)"},
        {"bsz": 1, "q_len": 4096, "name": "Very Long (4096 tokens)"},
        {"bsz": 1, "q_len": 8192, "name": "Extremely Long (8192 tokens)"},
        {"bsz": 1, "q_len": 16384, "name": "Ultra Long (16384 tokens)"},
        {"bsz": 2, "q_len": 4096, "name": "Batch=2 (4096 tokens)"},
        {"bsz": 2, "q_len": 8192, "name": "Batch=2 (8192 tokens)"},
    ]
    
    # Create press instances
    rkv_press = RKVPress(
        cache_budget=128,
        window_size=8,
        kernel_size=5,
    )
    
    rkv_lsh_press = RKVLSHPress(
        cache_budget=128,
        window_size=8,
        kernel_size=5,
        n_hash_buckets=8,
        lam=0.1,
    )
    
    # Initialize buckets for RKVLSHPress ONCE before all benchmarks
    # This is the "fixed bucket counting" initialization - done outside the loop
    rkv_lsh_press.initialize_buckets(device=device)
    
    results = []
    
    for config in test_configs:
        print("\n" + "=" * 80)
        print("Testing:", config['name'])
        print("=" * 80)
        
        # Create inputs
        keys, values, hidden_states, attentions, kwargs = create_dummy_inputs(
            bsz=config["bsz"],
            q_len=config["q_len"],
            device=device,
            dtype=dtype,
        )
        
        # Create mock module with proper dimensions for this configuration
        module = MockAttentionModule(
            hidden_size=hidden_states.shape[-1],
            head_dim=keys.shape[-1],
            num_heads=32
        )
        module = module.to(device).to(dtype)
        module.eval()  # Set to eval mode
        
        print("Input shapes:")
        print("  keys:", keys.shape)
        print("  values:", values.shape)
        print("  hidden_states:", hidden_states.shape)
        
        # Determine number of iterations based on sequence length (more for larger sequences)
        # For very long sequences, use more iterations for better statistics
        if config["q_len"] >= 16384:
            num_iterations = 50  # Fewer iterations for ultra-long sequences (they take longer)
            num_runs = 5
        elif config["q_len"] >= 8192:
            num_iterations = 100
            num_runs = 5
        elif config["q_len"] >= 4096:
            num_iterations = 150
            num_runs = 4
        else:
            num_iterations = 200
            num_runs = 3
        
        # Benchmark RKVPress
        print(f"\nBenchmarking RKVPress ({num_runs} runs, {num_iterations} iterations each)...")
        rkv_results = benchmark_score_method(
            rkv_press, module, keys, values, hidden_states, attentions, kwargs,
            num_warmup=10, num_iterations=num_iterations, num_runs=num_runs, device=device
        )
        
        # Benchmark RKVLSHPress
        print(f"Benchmarking RKVLSHPress ({num_runs} runs, {num_iterations} iterations each)...")
        rkv_lsh_results = benchmark_score_method(
            rkv_lsh_press, module, keys, values, hidden_states, attentions, kwargs,
            num_warmup=10, num_iterations=num_iterations, num_runs=num_runs, device=device
        )
        
        # Print results
        print("\nResults:")
        print("  RKVPress:")
        print(f"    Mean:   {rkv_results['mean_ms']:.3f} ms")
        print(f"    Median: {rkv_results['median_ms']:.3f} ms")
        print(f"    Std:    {rkv_results['std_ms']:.3f} ms")
        print(f"    Min:    {rkv_results['min_ms']:.3f} ms")
        print(f"    Max:    {rkv_results['max_ms']:.3f} ms")
        
        print("  RKVLSHPress:")
        print(f"    Mean:   {rkv_lsh_results['mean_ms']:.3f} ms")
        print(f"    Median: {rkv_lsh_results['median_ms']:.3f} ms")
        print(f"    Std:    {rkv_lsh_results['std_ms']:.3f} ms")
        print(f"    Min:    {rkv_lsh_results['min_ms']:.3f} ms")
        print(f"    Max:    {rkv_lsh_results['max_ms']:.3f} ms")
        
        # Calculate speedup (always show as "X times faster/slower")
        # speedup > 1 means RKVLSHPress is faster
        speedup = rkv_results['mean_ms'] / rkv_lsh_results['mean_ms']
        if speedup > 1:
            print(f"\n  ✓ RKVLSHPress is {speedup:.2f}x FASTER than RKVPress")
            speedup_display = f"{speedup:.2f}x faster"
        else:
            slowdown = 1 / speedup
            print(f"\n  ✗ RKVLSHPress is {slowdown:.2f}x SLOWER than RKVPress")
            speedup_display = f"{slowdown:.2f}x slower"
        
        results.append({
            "config": config["name"],
            "rkv": rkv_results,
            "rkv_lsh": rkv_lsh_results,
            "speedup": speedup,
            "speedup_display": speedup_display,
        })
    
    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Configuration':<30} {'RKVPress (ms)':<15} {'RKVLSHPress (ms)':<18} {'Speedup':<20}")
    print("-" * 85)
    for result in results:
        speedup_val = result['speedup']
        if speedup_val > 1:
            speedup_str = f"{speedup_val:.2f}x faster"
        else:
            slowdown = 1 / speedup_val
            speedup_str = f"{slowdown:.2f}x slower"
        
        print(f"{result['config']:<30} "
              f"{result['rkv']['mean_ms']:<15.3f} "
              f"{result['rkv_lsh']['mean_ms']:<18.3f} "
              f"{speedup_str:<20}")
    
    print("\n" + "=" * 80)
    print("Benchmark complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

