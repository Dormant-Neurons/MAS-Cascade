#!/usr/bin/env python3
"""
Summarize belief logs into one CSV per scenario/seed.

Output path: {seed_dir}/beliefs_summary.csv
Output format (one row per item_id x round):
  item_id, round, agent_0_A, agent_0_B, ..., agent_N_A, agent_N_B, ...

Usage:
  python cascade/analysis/summarize_beliefs.py [base_dir]

  base_dir defaults to "output"
"""

import re
import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd


def parse_round_file(path: Path):
    """
    Parse belief_after_item_{id}_round{N}.csv.

    Returns (item_id, round_num, options, beliefs_2d) or (None,...) on failure.
    beliefs_2d: list of lists, shape (num_agents, num_options)
    """
    match = re.search(r'belief_after_item_([a-f0-9]+)_round(\d+)\.csv', path.name)
    if not match:
        return None, None, None, None

    item_id = match.group(1)
    round_num = int(match.group(2))

    df = pd.read_csv(path, header=None)
    options = [str(o) for o in df.iloc[0].tolist()]
    beliefs = df.iloc[1:].values.astype(float).tolist()

    return item_id, round_num, options, beliefs


def summarize_seed_dir(seed_dir: Path) -> Path | None:
    """
    Collect all belief files under seed_dir/belief_logs/ and write
    beliefs_summary.csv next to belief_logs/.

    Returns the output path, or None if no files were found.
    """
    belief_log_dir = seed_dir / "belief_logs"
    round_files = list(belief_log_dir.glob("belief_after_item_*_round*.csv"))

    if not round_files:
        return None

    # Gather all records: (item_id, round_num, options, beliefs)
    records = []
    for path in round_files:
        item_id, round_num, options, beliefs = parse_round_file(path)
        if item_id is None:
            continue
        records.append((item_id, round_num, options, beliefs))

    if not records:
        return None

    # Determine num_agents and options from first valid record
    # (assume consistent across files)
    _, _, options, first_beliefs = records[0]
    num_agents = len(first_beliefs)
    num_options = len(options)

    # Build column names: agent_0_A, agent_0_B, ..., agent_N_X
    belief_cols = [
        f"agent_{a}_{opt}"
        for a in range(num_agents)
        for opt in options
    ]

    rows = []
    for item_id, round_num, _, beliefs in records:
        flat = [b for agent_row in beliefs for b in agent_row]
        rows.append([item_id, round_num] + flat)

    df = pd.DataFrame(rows, columns=["item_id", "round"] + belief_cols)
    df = df.sort_values(["item_id", "round"]).reset_index(drop=True)

    out_path = seed_dir / "beliefs_summary.csv"
    df.to_csv(out_path, index=False)
    return out_path


def process_all(base_dir: Path):
    """
    Recursively find all belief_logs dirs, summarize their parent (seed) dir.
    """
    belief_log_dirs = sorted(base_dir.rglob("belief_logs"))

    if not belief_log_dirs:
        print(f"No belief_logs directories found in {base_dir}")
        return

    print(f"Found {len(belief_log_dirs)} belief_logs directories\n")

    for belief_log_dir in belief_log_dirs:
        seed_dir = belief_log_dir.parent
        # Pretty label: model/dataset/scenario/seed
        try:
            label = "/".join(seed_dir.parts[-4:])
        except Exception:
            label = str(seed_dir)

        out = summarize_seed_dir(seed_dir)
        if out:
            print(f"  {label}  ->  {out.name}")
        else:
            print(f"  {label}  ->  (no files)")

    print("\nDone.")


def main():
    base_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output")

    if not base_dir.exists():
        print(f"ERROR: Directory not found: {base_dir}")
        sys.exit(1)

    print(f"Summarizing belief logs under: {base_dir}\n")
    process_all(base_dir)


if __name__ == "__main__":
    main()