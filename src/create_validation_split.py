"""
Create deterministic stratified train/validation split (80/20) from ground truth.
Outputs Parquet files: data/train_gt_split.parquet and data/val_gt_split.parquet
"""

import csv
import random
from collections import defaultdict
from pathlib import Path
import polars as pl

# Use absolute dataset path directly to prevent symlink errors
DATA_DIR = Path("/home/ubuntu/dataset/train")
GT_FILE = DATA_DIR / "train_ground_truth.tsv"
S1_FILE = DATA_DIR / "train_source1.tsv"

TRAIN_OUT = Path("data/train_gt_split.parquet")
VAL_OUT = Path("data/val_gt_split.parquet")

VALIDATION_FRACTION = 0.20
RANDOM_SEED = 20260926


def count_matches(raw: str) -> int:
    """Count comma-separated target IDs; null/blank means 0 matches (singletons/unmatched)."""
    if raw is None or not str(raw).strip():
        return 0
    return len([item for item in str(raw).split(",") if item.strip()])


def match_bin(n: int) -> str:
    """Bin the match count into distinct operational buckets."""
    if n == 0:
        return "0"
    if n == 1:
        return "1"
    if n <= 3:
        return "2-3"
    return "4+"


def main():
    print(f"Loading Source 1 country mappings from {S1_FILE}...")
    s1_df = pl.read_csv(S1_FILE, separator="\t")
    s1_country = dict(
        zip(
            s1_df["entity_id"].to_list(),
            s1_df["country"].fill_null("UNKNOWN").to_list()
        )
    )

    print(f"Loading ground truth from {GT_FILE}...")
    gt_rows = []
    with GT_FILE.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            gt_rows.append(row)

    print(f"Total ground-truth records: {len(gt_rows):,}")

    # Stratify by (country, match_bin)
    strata = defaultdict(list)
    for row in gt_rows:
        s1_id = row["source1_entity_id"]
        num_m = count_matches(row.get("matched_entity_ids", ""))
        m_bin = match_bin(num_m)
        country = s1_country.get(s1_id, "UNKNOWN") or "UNKNOWN"
        
        stratum_key = (country, m_bin)
        strata[stratum_key].append(row)

    train_rows = []
    val_rows = []

    rng = random.Random(RANDOM_SEED)

    print("Stratifying 80/20 train/val split...")
    for stratum_key, items in strata.items():
        rng.shuffle(items)
        val_count = int(round(len(items) * VALIDATION_FRACTION))
        val_rows.extend(items[:val_count])
        train_rows.extend(items[val_count:])

    TRAIN_OUT.parent.mkdir(parents=True, exist_ok=True)

    # Convert to Polars DataFrames and save as Parquet
    train_df = pl.DataFrame(train_rows)
    val_df = pl.DataFrame(val_rows)

    train_df.write_parquet(TRAIN_OUT, compression="snappy")
    val_df.write_parquet(VAL_OUT, compression="snappy")

    print("\n--- Split Summary ---")
    print(f"Train split saved: {train_df.height:,} rows -> {TRAIN_OUT}")
    print(f"Val split saved:   {val_df.height:,} rows -> {VAL_OUT}")


if __name__ == "__main__":
    main()
