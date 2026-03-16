#!/usr/bin/env python3
"""
Consolidate individual round belief CSV files into per-question consolidated CSVs.

Converts from:
  belief_logs/belief_after_item_{item_id}_round{N}.csv (one file per round)

To:
  belief_logs/{item_id}_beliefs.csv (all rounds for one question in one file)

Output format expected by fit_fj_star_full.py:
  round,A,B,C,D,E,A,B,C,D,E,...  (header)
  A,B,C,D,E,A,B,C,D,E,...         (options row)
  round0,0.2,0.2,0.2,0.2,0.2,... (data rows)
  round1,0.3,0.1,0.3,0.2,0.1,...
  ...
"""

import re
import sys
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np

def parse_round_file(path: Path) -> tuple:
    """
    Parse belief_after_item_{id}_round{N}.csv file.

    Returns:
        (item_id, round_num, beliefs_matrix)
        beliefs_matrix: (num_agents, num_options) array
    """
    match = re.search(r'belief_after_item_([a-f0-9]+)_round(\d+)\.csv', path.name)
    if not match:
        return None, None, None, None

    item_id = match.group(1)
    round_num = int(match.group(2))

    df = pd.read_csv(path, header=None)
    options = df.iloc[0].tolist()
    beliefs = df.iloc[1:].values.astype(float)

    return item_id, round_num, beliefs, options


def consolidate_belief_logs(belief_log_dir: Path):
    """
    Find all belief_after_item_*_round*.csv files in directory,
    group by item_id, and create consolidated CSV for each item.
    """
    round_files = list(belief_log_dir.glob("belief_after_item_*_round*.csv"))

    if not round_files:
        print(f"  No belief round files found in {belief_log_dir}")
        return 0

    items = defaultdict(dict)  # item_id -> {round_num: (beliefs, options)}

    for path in round_files:
        item_id, round_num, beliefs, options = parse_round_file(path)
        if item_id is None:
            continue
        items[item_id][round_num] = (beliefs, options)

    num_created = 0
    for item_id, rounds_data in items.items():
        round_nums = sorted(rounds_data.keys())

        if not round_nums:
            continue

        first_beliefs, options = rounds_data[round_nums[0]]
        num_agents, num_options = first_beliefs.shape

        rows = []
        for rnd in round_nums:
            beliefs, _ = rounds_data[rnd]
            # Flatten: agent0_optionA, agent0_optionB, ..., agent1_optionA, ...
            row_data = beliefs.flatten()  # Row-major: all options for agent0, then agent1, etc.
            rows.append([f"round{rnd}"] + row_data.tolist())

        cols = ["round"]
        for opt in options:
            cols.extend([opt] * num_agents)

        df = pd.DataFrame(rows, columns=cols)

        options_row = [""] + options * num_agents
        df_with_header = pd.DataFrame([options_row] + df.values.tolist(), columns=df.columns)

        out_path = belief_log_dir / f"{item_id}_beliefs.csv"
        df_with_header.to_csv(out_path, index=False, header=True)
        num_created += 1

    return num_created


def process_all_experiments(base_dir: Path):
    """
    Recursively find all belief_logs directories and consolidate them.
    """
    belief_log_dirs = list(base_dir.rglob("belief_logs"))

    if not belief_log_dirs:
        print(f"No belief_logs directories found in {base_dir}")
        return

    print(f"Found {len(belief_log_dirs)} belief_logs directories")
    print()

    total_created = 0
    for belief_log_dir in belief_log_dirs:
        # Get experiment name (parent's parent's name)
        experiment_name = belief_log_dir.parent.parent.name
        print(f"Processing {experiment_name}...")

        num_created = consolidate_belief_logs(belief_log_dir)
        total_created += num_created

        if num_created > 0:
            print(f"  Created {num_created} consolidated belief CSVs")
        print()

    print(f"✓ Total consolidated CSVs created: {total_created}")


def main():
    if len(sys.argv) > 1:
        base_dir = Path(sys.argv[1])
    else:
        base_dir = Path("output/gemini-3-flash-preview/csqa_test30")

    if not base_dir.exists():
        print(f"ERROR: Directory not found: {base_dir}")
        return

    print(f"Consolidating belief logs in: {base_dir}")
    print()

    process_all_experiments(base_dir)


if __name__ == "__main__":
    main()