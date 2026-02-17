# Qualitative Analysis Setup - Quick Start

## Issues Fixed

1. ✅ **Added `--enable_qualitative_analysis` flag** to `batch_script.py`
2. ✅ **Fixed tokenizer setup** in `evaluate.py` - now sets tokenizer when either `track_tokens=True` OR `enable_qualitative_analysis=True`
3. ✅ **Configured incremental saving** - data saves after each sample during execution

## How to Run

### Option 1: Using batch_script.py (Recommended)

```bash
python batch_script.py
```

This will run with qualitative analysis enabled automatically.

### Option 2: Manual run for testing

```bash
python reason/evaluate.py \
    --dataset=math500 \
    --model_name=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --press_name=rkvlsh \
    --cache_budget=1024 \
    --lam=0.1 \
    --n_hash_buckets=16 \
    --num_samples=5 \
    --enable_qualitative_analysis=True \
    --measure_memory=false \
    --measure_latency=true
```

## Where to Find Output

After running, you should see:

### 1. Directory Structure
```
kvpress/
├── ranking_analysis/           # Created automatically when first sample completes
│   ├── token_decisions_rkvlsh_budget1024_buckets16.jsonl      # Incremental data (JSONL)
│   └── token_decisions_rkvlsh_budget1024_buckets16_summary.txt # Human-readable summary
```

### 2. Monitor Progress During Execution

```bash
# Watch the file being updated in real-time
tail -f ranking_analysis/token_decisions_rkvlsh_budget1024_buckets16.jsonl

# Count how many samples have been processed
wc -l ranking_analysis/token_decisions_rkvlsh_budget1024_buckets16.jsonl

# Check file size
ls -lh ranking_analysis/
```

## Expected Output Messages

During execution, you'll see:

```
[RKV-LSH] Qualitative analysis mode enabled
[RKV-LSH] Incremental output will be saved to: ranking_analysis/token_decisions_rkvlsh_budget1024_buckets16.jsonl
...
[RKV-LSH] Sample 0 qualitative data saved (5 eviction steps)
✅ [1/5] Saved result for question 1 to ...
[RKV-LSH] Sample 1 qualitative data saved (6 eviction steps)
✅ [2/5] Saved result for question 2 to ...
...
[RKV-LSH] Qualitative analysis complete:
  - Total samples: 5
  - Total eviction steps: 28
  - Incremental data saved to: ranking_analysis/token_decisions_rkvlsh_budget1024_buckets16.jsonl
  - Summary saved to: ranking_analysis/token_decisions_rkvlsh_budget1024_buckets16_summary.txt
✅ Qualitative analysis complete - check ranking_analysis/token_decisions_rkvlsh_budget1024_buckets16.jsonl
```

## Output File Format

### JSONL File (Incremental Data)
Each line is a JSON object representing one sample:

```json
{"sample_id": 0, "num_eviction_steps": 5, "eviction_steps": [...]}
{"sample_id": 1, "num_eviction_steps": 6, "eviction_steps": [...]}
{"sample_id": 2, "num_eviction_steps": 4, "eviction_steps": [...]}
```

### Summary File (Human-Readable)
Shows:
- Top retained tokens with scores
- Top evicted tokens with scores
- Repetitive keyword statistics (wait, so, but)
- Token texts decoded for easy reading

## Troubleshooting

### "Directory not found"
- The `ranking_analysis/` directory is created automatically when the first sample completes
- If you don't see it, check for error messages in the output

### "No qualitative data collected"
- Make sure `--enable_qualitative_analysis=True` is set
- Check that the press is RKV-LSH (`--press_name=rkvlsh`)
- Verify at least one sample ran successfully

### "Cannot generate summary without tokenizer"
- This should be fixed now with the tokenizer setup changes
- If you still see this, make sure you're using the updated `evaluate.py`

## Comparing RKV vs RKV-LSH

To compare two methods, run twice with different lambda values:

```bash
# RKV (attention-only)
python reason/evaluate.py --press_name=rkvlsh --lam=1.0 --n_hash_buckets=6 --enable_qualitative_analysis=True ...

# RKV-LSH (redundancy-only)
python reason/evaluate.py --press_name=rkvlsh --lam=0.0 --n_hash_buckets=32 --enable_qualitative_analysis=True ...
```

Then use the comparison script:
```bash
python compare_rkv_rkvlsh.py \
    --rkv_file ranking_analysis/token_decisions_rkv_budget1024_buckets6.jsonl \
    --rkvlsh_file ranking_analysis/token_decisions_rkvlsh_budget1024_buckets32.jsonl \
    --output comparison_report.txt
```

Note: You may need to update `compare_rkv_rkvlsh.py` to read JSONL format instead of JSON.
