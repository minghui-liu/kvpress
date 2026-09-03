#!/usr/bin/env python3
"""
Submit a SLURM array job while respecting a per-user active-job cap.

The script polls `squeue` every 30 minutes by default. If the user's current
active job count is below the cap, it submits the next array range that fits in
the remaining capacity. If the cap is reached, it waits and checks again.

Example:
    python submit_slurm_chunks.py
    python submit_slurm_chunks.py --script batch_slurm.sh --total 2520 --max-active 250
"""

import argparse
import getpass
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit a SLURM array script with polling and active-job throttling.",
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
        help="Total number of array tasks to cover. For 0-2519, use 2520.",
    )
    parser.add_argument(
        "--max-active",
        type=int,
        default=250,
        help="Maximum number of active jobs allowed for the user at one time.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=1800,
        help="Polling interval in seconds. Default is 1800 (30 minutes).",
    )
    parser.add_argument(
        "--user",
        type=str,
        default=getpass.getuser(),
        help="SLURM username to check with squeue.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be submitted without calling sbatch.",
    )
    return parser.parse_args()


def get_active_job_count(user: str) -> int:
    result = subprocess.run(
        ["squeue", "-h", "-u", user],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    return len(lines)


def submit_range(script: Path, start: int, end: int, dry_run: bool) -> None:
    cmd = ["sbatch", f"--array={start}-{end}", str(script)]
    print(f"Submitting range {start}-{end}: {' '.join(cmd)}")
    if dry_run:
        return

    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    stdout = result.stdout.strip()
    if stdout:
        print(stdout)


def main() -> int:
    args = parse_args()

    if not args.script.is_file():
        print(f"Batch script not found: {args.script}", file=sys.stderr)
        return 1
    if args.total <= 0:
        print("--total must be > 0", file=sys.stderr)
        return 1
    if args.max_active <= 0:
        print("--max-active must be > 0", file=sys.stderr)
        return 1
    if args.poll_seconds <= 0:
        print("--poll-seconds must be > 0", file=sys.stderr)
        return 1

    next_start = 0
    final_index = args.total - 1

    print(
        f"Managing submissions for tasks 0-{final_index} using {args.script}. "
        f"User={args.user}, max_active={args.max_active}, poll_seconds={args.poll_seconds}."
    )

    while next_start < args.total:
        try:
            active_jobs = get_active_job_count(args.user)
        except subprocess.CalledProcessError as exc:
            print(f"Failed to query squeue for user {args.user}: {exc}", file=sys.stderr)
            return 1

        print(f"Current active jobs for {args.user}: {active_jobs}")

        if active_jobs >= args.max_active:
            print(f"Active jobs reached cap {args.max_active}. Sleeping for {args.poll_seconds} seconds.")
            time.sleep(args.poll_seconds)
            continue

        available_slots = args.max_active - active_jobs
        submit_count = min(available_slots, args.total - next_start)
        submit_end = next_start + submit_count - 1

        submit_range(args.script, next_start, submit_end, args.dry_run)
        next_start = submit_end + 1

        if next_start < args.total:
            print(f"Submitted through {submit_end}. Next task to submit: {next_start}.")
            time.sleep(args.poll_seconds)

    print("All array tasks have been submitted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
