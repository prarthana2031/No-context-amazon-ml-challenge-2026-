"""
Phase 2 Task Execution for Member D:
1. Quality checks on candidate set.
2. Blocking recall evaluation using validation split.
3. Mining hard negatives for Phase 3 model training.
"""

from pathlib import Path
import polars as pl
from metrics import calculate_blocking_recall

# File Paths
VAL_SPLIT_PATH = Path("data/val_gt_split.parquet")
CANDIDATES_PATH = Path("data/candidate_pairs.parquet")
CANDIDATES_TSV_PATH = Path("data/candidate_pairs.tsv")
OUTPUT_MINED_NEGATIVES = Path("data/hard_negatives_val.parquet")


def load_candidates() -> pl.DataFrame:
    """Load candidate pairs supporting both Parquet and TSV formats."""
    if CANDIDATES_PATH.exists():
        print(f"Loading candidate pairs from Parquet: {CANDIDATES_PATH}")
        return pl.read_parquet(CANDIDATES_PATH)
    elif CANDIDATES_TSV_PATH.exists():
        print(f"Loading candidate pairs from TSV: {CANDIDATES_TSV_PATH}")
        return pl.read_csv(CANDIDATES_TSV_PATH, separator="\t")
    else:
        raise FileNotFoundError(
            f"Candidate file not found at {CANDIDATES_PATH} or {CANDIDATES_TSV_PATH}"
        )


def run_quality_checks(cand_df: pl.DataFrame):
    """Task 1: Check schema, row counts, and nulls."""
    print("\n--- 1. Quality Checks ---")
    print(f"Total candidate pairs generated: {cand_df.height:,}")
    print(f"Columns present: {cand_df.columns}")

    required_cols = {"source1_entity_id", "matched_entity_id"}
    missing_cols = required_cols - set(cand_df.columns)
    if missing_cols:
        raise ValueError(f"CRITICAL: Candidate set missing required columns: {missing_cols}")

    null_count = cand_df.select([
        pl.col("source1_entity_id").null_count().alias("s1_nulls"),
        pl.col("matched_entity_id").null_count().alias("s2_nulls")
    ])
    print(f"Null value check: {null_count.to_dicts()[0]}")
    assert null_count["s1_nulls"][0] == 0, "Nulls found in source1_entity_id!"
    assert null_count["s2_nulls"][0] == 0, "Nulls found in matched_entity_id!"
    print("✓ Quality checks passed successfully!")


def evaluate_recall_and_mine_negatives(val_gt_path: Path, cand_df: pl.DataFrame):
    """Task 2 & 3: Measure recall and extract hard negatives."""
    print("\n--- 2. Measuring Blocking Recall ---")
    val_gt_df = pl.read_parquet(val_gt_path)

    # Explode comma-separated ground truth matched_entity_ids into distinct pairs
    gt_pairs = (
        val_gt_df
        .filter(pl.col("matched_entity_ids").is_not_null() & (pl.col("matched_entity_ids") != ""))
        .with_columns(pl.col("matched_entity_ids").str.split(","))
        .explode("matched_entity_ids")
        .with_columns(pl.col("matched_entity_ids").str.strip_chars())
        .rename({"matched_entity_ids": "matched_entity_id"})
        .select(["source1_entity_id", "matched_entity_id"])
        .unique()
    )

    print(f"Total true positive pairs in validation ground truth: {gt_pairs.height:,}")

    # Calculate Blocking Recall
    recall = calculate_blocking_recall(gt_pairs_df=gt_pairs, candidate_pairs_df=cand_df)
    print("\n==========================================")
    print(f"  BLOCKING RECALL (Validation): {recall * 100:.2f}%")
    print("==========================================")

    print("\n--- 3. Mining Hard Negatives ---")
    # Hard Negatives = Candidate pairs generated that are NOT true ground truth matches
    hard_negatives = cand_df.join(
        gt_pairs,
        on=["source1_entity_id", "matched_entity_id"],
        how="anti"
    )

    print(f"Total Hard Negative pairs mined: {hard_negatives.height:,}")

    OUTPUT_MINED_NEGATIVES.parent.mkdir(parents=True, exist_ok=True)
    hard_negatives.write_parquet(OUTPUT_MINED_NEGATIVES, compression="snappy")
    print(f"Saved hard negatives to: {OUTPUT_MINED_NEGATIVES}")


def main():
    if not VAL_SPLIT_PATH.exists():
        print(f"Error: Validation split missing at {VAL_SPLIT_PATH}.")
        return

    try:
        cand_df = load_candidates()
    except FileNotFoundError as e:
        print(f"\n[Awaiting Candidate File] {e}")
        print("Run this script once Member A/B/C produces 'data/candidate_pairs.parquet' or 'data/candidate_pairs.tsv'.")
        return

    run_quality_checks(cand_df)
    evaluate_recall_and_mine_negatives(VAL_SPLIT_PATH, cand_df)


if __name__ == "__main__":
    main()
