#!/usr/bin/env python3
"""
Submit a SLURM array job in sequential chunks.

Example:
    python submit_slurm_chunks.py
    python submit_slurm_chunks.py --script batch_slurm.sh --total 2520 --chunk-size 250
"""

import argparse
import math
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit a SLURM array script in chunks.",
    )
    parser.add_argument(
        "--script",
        type=Path,
        default=Path("batch_slurm.sh"),
        help="Path to the SLURM batch script.",
    )
    parser.add_argument(
        "--total",
        type=int,
        default=2520,
        help="Total number of array tasks to cover.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=250,
        help="Maximum number of array tasks per sbatch submission.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print sbatch commands without submitting them.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.script.is_file():
        print(f"Batch script not found: {args.script}", file=sys.stderr)
        return 1
    if args.total <= 0:
        print("--total must be > 0", file=sys.stderr)
        return 1
    if args.chunk_size <= 0:
        print("--chunk-size must be > 0", file=sys.stderr)
        return 1

    num_submissions = math.ceil(args.total / args.chunk_size)
    print(
        f"Submitting {args.total} tasks from {args.script} "
        f"in {num_submissions} chunk(s) of up to {args.chunk_size}."
    )

    for start in range(0, args.total, args.chunk_size):
        end = min(start + args.chunk_size - 1, args.total - 1)
        cmd = ["sbatch", f"--array={start}-{end}", str(args.script)]
        print(f"[{start}-{end}] {' '.join(cmd)}")

        if args.dry_run:
            continue

        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        stdout = result.stdout.strip()
        if stdout:
            print(stdout)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
