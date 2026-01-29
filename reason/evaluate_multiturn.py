# Multi-turn conversation evaluation for KV cache compression
# Tests how well compression methods preserve context across conversation turns

import contextlib
import json
import os
from pathlib import Path
from typing import Optional, List, Dict, Any

import torch
from datasets import load_dataset
from fire import Fire
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from kvpress import BasePress
from kvpress import (
    KnormPress,
    RandomPress,
    StreamingLLMPress,
    FullPress,
    RKVPress,
    RKVLSHPress,
    H2OPress,
)
from kvpress.presses.snapkv_press import SnapKVPress
from kvpress.presses.pyramidkv_press import PyramidKVPress


# Multi-turn dataset configurations
MULTITURN_DATASETS = {
    "gsm8k_multiturn": ("euclaise/gsm8k_multiturn", None, "train"),
}


def get_press(press_name: str, cache_budget: int, **kwargs) -> BasePress:
    """Create a press instance by name."""
    press_map = {
        "full": FullPress,
        "knorm": KnormPress,
        "random": RandomPress,
        "streaming_llm": StreamingLLMPress,
        "rkv": RKVPress,
        "rkv_lsh": RKVLSHPress,
        "h2o": H2OPress,
        "snapkv": SnapKVPress,
        "pyramidkv": PyramidKVPress,
    }
    
    if press_name not in press_map:
        raise ValueError(f"Unknown press: {press_name}. Available: {list(press_map.keys())}")
    
    press_class = press_map[press_name]
    
    # Different presses have different parameter names
    if press_name in ["full"]:
        return press_class()
    else:
        # ALL presses need cache_budget to actually compress!
        press = press_class(cache_budget=cache_budget, **kwargs)
        # Disable CSV logging for speed (HUGE speedup, especially for H2O)
        press.latency = True
        # Suppress progress prints by providing a no-op callback
        press.progress_update = lambda *args, **kwargs: None
        return press


def extract_final_answer(text: str) -> str:
    """Extract the numerical answer from GSM8K response."""
    import re
    # Look for the last number in the response
    numbers = re.findall(r'[-+]?\d*\.?\d+', text.replace(',', ''))
    if numbers:
        return numbers[-1]
    return text.strip()


def gsm8k_scorer(pred: str, gt: str) -> bool:
    """Score GSM8K answer - compare final numbers."""
    import re
    
    def normalize(s):
        # Extract number, handle commas, etc.
        s = str(s).replace(',', '').replace('$', '').strip()
        numbers = re.findall(r'[-+]?\d*\.?\d+', s)
        if numbers:
            try:
                return float(numbers[-1])
            except:
                return s
        return s
    
    pred_norm = normalize(pred)
    gt_norm = normalize(gt)
    
    if isinstance(pred_norm, float) and isinstance(gt_norm, float):
        return abs(pred_norm - gt_norm) < 1e-5
    return str(pred_norm) == str(gt_norm)


def format_conversation_for_model(
    conversation: List[Dict[str, str]], 
    tokenizer, 
    up_to_turn: int = None
) -> str:
    """
    Format a multi-turn conversation into the model's chat template.
    
    Args:
        conversation: List of {"role": "user"/"assistant", "content": "..."}
        tokenizer: The model's tokenizer with chat template
        up_to_turn: If specified, only include turns up to this index
    """
    if up_to_turn is not None:
        conversation = conversation[:up_to_turn]
    
    # Use the tokenizer's chat template
    try:
        return tokenizer.apply_chat_template(
            conversation, 
            tokenize=False, 
            add_generation_prompt=True
        )
    except:
        # Fallback for models without chat template
        text = ""
        for turn in conversation:
            role = turn["role"]
            content = turn["content"]
            if role == "user":
                text += f"User: {content}\n"
            else:
                text += f"Assistant: {content}\n"
        text += "Assistant:"
        return text


def evaluate_multiturn(
    dataset: str = "gsm8k_multiturn",
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    press_name: str = "rkv",
    cache_budget: int = 512,
    num_samples: Optional[int] = None,
    num_turns: Optional[int] = None,
    random_seed: int = 42,
    max_new_tokens: int = 2056,
    track_tokens: bool = False,
    do_sampling: bool = False,
    debug: bool = False,
    output_dir: str = "/fs/nexus-scratch/apalnitk/fsx/kvpress/reason/results_multiturn",
):
    """
    Evaluate KV cache compression on multi-turn conversations.
    
    This tests whether compression methods can preserve context from earlier
    turns when answering questions in later turns.
    
    Args:
        dataset: Multi-turn dataset name (e.g., "gsm8k_multiturn")
        model_name: HuggingFace model name
        press_name: Compression method (rkv, rkv_lsh, h2o, snapkv, etc.)
        cache_budget: Maximum KV cache size
        num_samples: Number of samples to evaluate (None = all)
        num_turns: Exact number of turns per conversation (filter)
        random_seed: Random seed for sampling
        max_new_tokens: Max tokens per turn response
        track_tokens: Enable token tracking for visualization
        do_sampling: Use sampling vs greedy decoding
        debug: Enable debug output
        output_dir: Directory for results
    """
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",  # Need for attention tracking
    )
    model.eval()
    
    # Create press
    print(f"Creating press: {press_name} with budget={cache_budget}")
    press = get_press(press_name, cache_budget)
    if hasattr(press, 'tokenizer'):
        press.tokenizer = tokenizer
    
    # Load dataset
    if dataset not in MULTITURN_DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}. Available: {list(MULTITURN_DATASETS.keys())}")
    
    ds_name, ds_subset, ds_split = MULTITURN_DATASETS[dataset]
    print(f"Loading dataset: {ds_name}")
    ds = load_dataset(ds_name, ds_subset, split=ds_split)
    
    # Filter by exact number of turns if specified
    if num_turns is not None:
        print(f"Filtering for conversations with exactly {num_turns} turns...")
        original_size = len(ds)
        # Count user turns (each user message = 1 turn)
        ds = ds.filter(lambda x: sum(1 for t in x["conversations"] if t["role"] == "user") == num_turns)
        print(f"  Filtered: {original_size} -> {len(ds)} samples")
    
    if num_samples is not None:
        ds = ds.shuffle(seed=random_seed).select(range(min(num_samples, len(ds))))
    
    print(f"Evaluating {len(ds)} samples")
    
    # Setup output
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_short = model_name.replace("/", "--")
    turns_suffix = f"__turns{num_turns}" if num_turns else ""
    save_filename = f"{dataset}__{model_short}__{press_name}__budget{cache_budget}{turns_suffix}__n{len(ds)}.jsonl"
    save_path = output_dir / save_filename
    partial_path = output_dir / (save_filename.replace(".jsonl", ".partial.jsonl"))
    
    # Skip if results already exist
    if save_path.exists():
        print(f"⏭️  Results already exist, skipping: {save_path}")
        # Load and return existing summary
        with open(save_path) as f:
            summary = json.loads(f.readline())
        return summary
    
    results = []
    correct_final = 0
    correct_all_turns = 0
    total_turns = 0
    start_idx = 0
    
    # Check for partial results to resume
    if partial_path.exists():
        print(f"📂 Found partial results, resuming...")
        with open(partial_path) as f:
            for line in f:
                r = json.loads(line)
                results.append(r)
                total_turns += r["num_turns"]
                if r["turns"] and r["turns"][-1]["correct"]:
                    correct_final += 1
                if r["correct_turns"] == r["num_turns"]:
                    correct_all_turns += 1
        start_idx = len(results)
        print(f"   Resuming from sample {start_idx}/{len(ds)}")
    
    for idx, example in enumerate(tqdm(ds, desc="Evaluating", initial=start_idx, total=len(ds))):
        if idx < start_idx:
            continue
        # Reset press state for each sample
        if press is not None:
            press.reset_timing()
            # Set csv_path to avoid file errors (empty string causes issues)
            csv_dir = output_dir / "csv_logs"
            csv_dir.mkdir(exist_ok=True)
            press.csv_path = str(csv_dir / f"sample{idx}.csv")
        
        conversation = example["conversations"]
        
        # Separate into turns (user questions and expected assistant answers)
        user_turns = [t for t in conversation if t["role"] == "user"]
        assistant_turns = [t for t in conversation if t["role"] == "assistant"]
        
        num_turns = len(user_turns)
        total_turns += num_turns
        
        sample_result = {
            "idx": idx,
            "num_turns": num_turns,
            "turns": [],
            "correct_turns": 0,
        }
        
        # Build conversation incrementally
        conversation_so_far = []
        past_key_values = None  # KV cache persists across turns!
        
        for turn_idx in range(num_turns):
            # Add user question
            user_msg = user_turns[turn_idx]
            conversation_so_far.append(user_msg)
            
            # Expected answer
            expected_answer = assistant_turns[turn_idx]["content"] if turn_idx < len(assistant_turns) else ""
            
            # Format input
            input_text = format_conversation_for_model(conversation_so_far, tokenizer)
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(device)
            
            # Show progress for each turn
            print(f"\r  Sample {idx+1}/{len(ds)}, Turn {turn_idx+1}/{num_turns}, {inputs['input_ids'].shape[1]} tokens...", end="", flush=True)
            
            # Generate with KV cache compression
            # H2O and some presses require output_attentions
            needs_attn = press_name in ["h2o", "snapkv", "pyramidkv"]
            
            try:
                with press(model) if press is not None else contextlib.nullcontext():
                    if do_sampling:
                        outputs = model.generate(
                            inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            max_new_tokens=max_new_tokens,
                            do_sample=True,
                            top_p=0.9,
                            temperature=0.7,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            use_cache=True,
                            output_attentions=needs_attn,
                        )
                    else:
                        outputs = model.generate(
                            inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            use_cache=True,
                            output_attentions=needs_attn,
                        )
            except Exception as e:
                print(f"Error in sample {idx}, turn {turn_idx}: {e}")
                continue
            
            # Decode response
            pred_start = inputs["input_ids"].shape[1]
            response = tokenizer.decode(outputs[0][pred_start:], skip_special_tokens=True)
            
            # Extract and score answer
            pred_answer = extract_final_answer(response)
            gt_answer = extract_final_answer(expected_answer)
            is_correct = gsm8k_scorer(pred_answer, gt_answer)
            
            if is_correct:
                sample_result["correct_turns"] += 1
            
            turn_result = {
                "turn_idx": turn_idx,
                "user_question": user_msg["content"],
                "expected_answer": expected_answer,
                "model_response": response,
                "pred_answer": pred_answer,
                "gt_answer": gt_answer,
                "correct": is_correct,
                "input_tokens": inputs["input_ids"].shape[1],
                "output_tokens": outputs[0].shape[0] - inputs["input_ids"].shape[1],
            }
            sample_result["turns"].append(turn_result)
            
            # Add model response to conversation for next turn
            conversation_so_far.append({
                "role": "assistant",
                "content": response
            })
            
            if debug:
                print(f"\n--- Sample {idx}, Turn {turn_idx} ---")
                print(f"Q: {user_msg['content'][:100]}...")
                print(f"Expected: {expected_answer[:100]}...")
                print(f"Got: {response[:100]}...")
                print(f"Correct: {is_correct}")
        
        # Check if final turn is correct (most important)
        if sample_result["turns"] and sample_result["turns"][-1]["correct"]:
            correct_final += 1
        
        # Check if all turns are correct
        if sample_result["correct_turns"] == num_turns:
            correct_all_turns += 1
        
        results.append(sample_result)
        
        # Show completion and save partial results
        print(f"\r  ✓ Sample {idx+1}/{len(ds)} done: {sample_result['correct_turns']}/{num_turns} turns correct")
        
        # Save partial results after each sample (for resume)
        partial_path = output_dir / (save_filename.replace(".jsonl", ".partial.jsonl"))
        with open(partial_path, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
    
    # Calculate metrics
    final_accuracy = correct_final / len(ds) if ds else 0
    all_turns_accuracy = correct_all_turns / len(ds) if ds else 0
    per_turn_accuracy = sum(r["correct_turns"] for r in results) / total_turns if total_turns else 0
    
    # Summary
    summary = {
        "dataset": dataset,
        "model_name": model_name,
        "press_name": press_name,
        "cache_budget": cache_budget,
        "num_samples": len(ds),
        "total_turns": total_turns,
        "avg_turns_per_sample": total_turns / len(ds) if ds else 0,
        "final_turn_accuracy": final_accuracy,
        "all_turns_correct_accuracy": all_turns_accuracy,
        "per_turn_accuracy": per_turn_accuracy,
    }
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Dataset: {dataset}")
    print(f"Model: {model_name}")
    print(f"Press: {press_name}, Budget: {cache_budget}")
    print(f"Samples: {len(ds)}, Total turns: {total_turns}")
    print(f"Final turn accuracy: {final_accuracy:.2%}")
    print(f"All turns correct: {all_turns_accuracy:.2%}")
    print(f"Per-turn accuracy: {per_turn_accuracy:.2%}")
    print("="*60)
    
    # Save results
    with open(save_path, "w") as f:
        f.write(json.dumps(summary) + "\n")
        for r in results:
            f.write(json.dumps(r) + "\n")
    
    # Clean up partial file
    if partial_path.exists():
        partial_path.unlink()
    
    print(f"\nResults saved to: {save_path}")
    
    return summary


if __name__ == "__main__":
    Fire(evaluate_multiturn)

