#!/bin/bash
python3 token_count.py output/mt_aggregate.jsonl
python3 token_count.py output/mt_retrieval.jsonl
python3 token_count.py output/mt_val_tracking.jsonl
python3 token_count.py output/mt_vars_tracking.jsonl