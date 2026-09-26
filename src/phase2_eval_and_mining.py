"""
Phase 2 Evaluation & Hard Negative Mining
Member D - Quality Control, Metrics & Mining Lead
"""

import sys
from pathlib import Path
import polars as pl

# Project Root Setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.metrics import calculate_blocking_recall

# Paths
VAL_GT_PATH = PROJECT_ROOT / "data" / "val_gt_split.parquet"
CANDIDATES_TSV_PATH = PROJECT_ROOT / "data" / "candidate_pairs.tsv"
CANDIDATES_PARQUET_PATH = PROJECT_ROOT / "data" / "candidate_pairs.parquet"
HARD_NEGATIVES_OUTPUT_PATH = PROJECT_ROOT / "data" / "hard_negatives_val.parquet"


def load_candidates() -> pl.DataFrame:
    """
    Loads candidate pairs from TSV or Parquet format.
    Automatically handles comma-separated lists (Member A's format) and explodes them into single pairs.
    """
    if CANDIDATES_PARQUET_PATH.exists():
        print(f"Loading candidate pairs from Parquet: {CANDIDATES_PARQUET_PATH.name}")
        df = pl.read_parquet(CANDIDATES_PARQUET_PATH)
    elif CANDIDATES_TSV_PATH.exists():
        print(f"Loading candidate pairs from TSV: {CANDIDATES_TSV_PATH.name}")
        df = pl.read_csv(CANDIDATES_TSV_PATH, separator="\t")
    else:
        raise FileNotFoundError(
            f"Candidate file missing! Expecting {CANDIDATES_TSV_PATH} or {CANDIDATES_PARQUET_PATH}"
        )

    # Handle Member A's comma-separated format ('candidate_entity_ids' column)
    if "candidate_entity_ids" in df.columns:
        df = (
            df
            .filter(
                pl.col("candidate_entity_ids").is_not_null() 
                & (pl.col("candidate_entity_ids") != "")
            )
            .with_columns(pl.col("candidate_entity_ids").str.split(","))
            .explode("candidate_entity_ids")
            .with_columns(pl.col("candidate_entity_ids").str.strip_chars())
            .rename({"candidate_entity_ids": "matched_entity_id"})
        )
    elif "candidate_id" in df.columns:
        df = df.rename({"candidate_id": "matched_entity_id"})

    # Clean & standardize column selection
    df = df.select(["source1_entity_id", "matched_entity_id"]).distinct()
    return df


def run_quality_checks(candidates_df: pl.DataFrame) -> None:
    """Performs schema verification and quality control checks on candidates."""
    print("\n--- Running Candidate Quality Checks ---")
    
    # 1. Schema check
    expected_cols = {"source1_entity_id", "matched_entity_id"}
    actual_cols = set(candidates_df.columns)
    if not expected_cols.issubset(actual_cols):
        raise ValueError(f"Schema mismatch! Expected {expected_cols}, got {actual_cols}")
    print("[PASS] Schema validation passed.")

    # 2. Total candidate pairs count
    total_pairs = candidates_df.height
    print(f"[INFO] Total candidate pairs loaded: {total_pairs:,}")

    # 3. Null values check
    null_count = candidates_df.null_count().sum(axis=1)[0]
    if null_count > 0:
        raise ValueError(f"Quality Check Failed: Found {null_count} null values in candidates dataframe!")
    print("[PASS] Null check passed (0 nulls found).")

    # 4. Candidate distribution per source entity
    cand_stats = (
        candidates_df
        .group_by("source1_entity_id")
        .agg(pl.len().alias("cand_count"))
    )
    avg_cands = cand_stats["cand_count"].mean()
    max_cands = cand_stats["cand_count"].max()
    print(f"[INFO] Candidate Statistics -> Avg per S1: {avg_cands:.2f} | Max per S1: {max_cands:,}")


def mine_hard_negatives(candidates_df: pl.DataFrame, val_gt_df: pl.DataFrame) -> pl.DataFrame:
    """
    Extracts hard negative pairs (pairs present in candidates but NOT in ground truth).
    """
    print("\n--- Phase 2: Mining Hard Negatives ---")

    # Format GT positive matches into exploded single pairs
    gt_positives = (
        val_gt_df
        .filter(
            pl.col("matched_entity_ids").is_not_null() 
            & (pl.col("matched_entity_ids") != "")
        )
        .with_columns(pl.col("matched_entity_ids").str.split(","))
        .explode("matched_entity_ids")
        .with_columns(pl.col("matched_entity_ids").str.strip_chars())
        .rename({"matched_entity_ids": "matched_entity_id"})
        .select(["source1_entity_id", "matched_entity_id"])
        .distinct()
    )

    # Perform Anti-Join to extract non-matching candidate pairs
    hard_negatives = candidates_df.join(
        gt_positives,
        on=["source1_entity_id", "matched_entity_id"],
        how="anti"
    )

    print(f"[SUCCESS] Extracted {hard_negatives.height:,} hard negative samples.")
    return hard_negatives


def main():
    print("==================================================")
    print("   PHASE 2: EVALUATION & HARD NEGATIVE MINING    ")
    print("==================================================")

    # 1. Load Validation Split
    if not VAL_GT_PATH.exists():
        print(f"Error: Validation split missing at {VAL_GT_PATH}")
        print("Please run 'python3 src/create_validation_split.py' first.")
        sys.exit(1)

    print(f"Loading validation ground truth from: {VAL_GT_PATH.name}")
    val_gt = pl.read_parquet(VAL_GT_PATH)

    # 2. Check and Load Candidates
    try:
        candidates = load_candidates()
    except FileNotFoundError as e:
        print(f"\n[Awaiting Candidate File] {e}")
        print("Run this script once Member A produces 'data/candidate_pairs.tsv'.")
        sys.exit(0)

    # 3. Quality Control Checks
    run_quality_checks(candidates)

    # 4. Calculate Blocking Recall
    print("\n--- Phase 2: Calculating Blocking Recall ---")
    recall_score = calculate_blocking_recall(candidates, val_gt)
    print(f"--> BLOCKING RECALL (Validation): {recall_score * 100:.2f}% <--")
    
    if recall_score >= 0.985:
        print("[TARGET MET] Recall meets the > 98.5% threshold target!")
    else:
        print("[WARNING] Recall is below 98.5%. Consider expanding blocking keys.")

    # 5. Extract and Save Hard Negatives
    hard_negs = mine_hard_negatives(candidates, val_gt)
    
    # Save output
    HARD_NEGATIVES_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    hard_negs.write_parquet(HARD_NEGATIVES_OUTPUT_PATH)
    print(f"[SAVED] Mined hard negatives written to: {HARD_NEGATIVES_OUTPUT_PATH}")
    print("\nPhase 2 execution complete. Ready for Phase 3 Model Training!")


if __name__ == "__main__":
    main()
How to Update It in Your Terminal
Run this command directly in your terminal to replace the old file:

Bash
cat > src/phase2_eval_and_mining.py <<'PY'
"""
Phase 2 Evaluation & Hard Negative Mining
Member D - Quality Control, Metrics & Mining Lead
"""

import sys
from pathlib import Path
import polars as pl

# Project Root Setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.metrics import calculate_blocking_recall

# Paths
VAL_GT_PATH = PROJECT_ROOT / "data" / "val_gt_split.parquet"
CANDIDATES_TSV_PATH = PROJECT_ROOT / "data" / "candidate_pairs.tsv"
CANDIDATES_PARQUET_PATH = PROJECT_ROOT / "data" / "candidate_pairs.parquet"
HARD_NEGATIVES_OUTPUT_PATH = PROJECT_ROOT / "data" / "hard_negatives_val.parquet"


def load_candidates() -> pl.DataFrame:
    """
    Loads candidate pairs from TSV or Parquet format.
    Automatically handles comma-separated lists (Member A's format) and explodes them into single pairs.
    """
    if CANDIDATES_PARQUET_PATH.exists():
        print(f"Loading candidate pairs from Parquet: {CANDIDATES_PARQUET_PATH.name}")
        df = pl.read_parquet(CANDIDATES_PARQUET_PATH)
    elif CANDIDATES_TSV_PATH.exists():
        print(f"Loading candidate pairs from TSV: {CANDIDATES_TSV_PATH.name}")
        df = pl.read_csv(CANDIDATES_TSV_PATH, separator="\t")
    else:
        raise FileNotFoundError(
            f"Candidate file missing! Expecting {CANDIDATES_TSV_PATH} or {CANDIDATES_PARQUET_PATH}"
        )

    # Handle Member A's comma-separated format ('candidate_entity_ids' column)
    if "candidate_entity_ids" in df.columns:
        df = (
            df
            .filter(
                pl.col("candidate_entity_ids").is_not_null() 
                & (pl.col("candidate_entity_ids") != "")
            )
            .with_columns(pl.col("candidate_entity_ids").str.split(","))
            .explode("candidate_entity_ids")
            .with_columns(pl.col("candidate_entity_ids").str.strip_chars())
            .rename({"candidate_entity_ids": "matched_entity_id"})
        )
    elif "candidate_id" in df.columns:
        df = df.rename({"candidate_id": "matched_entity_id"})

    # Clean & standardize column selection
    df = df.select(["source1_entity_id", "matched_entity_id"]).distinct()
    return df


def run_quality_checks(candidates_df: pl.DataFrame) -> None:
    """Performs schema verification and quality control checks on candidates."""
    print("\n--- Running Candidate Quality Checks ---")
    
    # 1. Schema check
    expected_cols = {"source1_entity_id", "matched_entity_id"}
    actual_cols = set(candidates_df.columns)
    if not expected_cols.issubset(actual_cols):
        raise ValueError(f"Schema mismatch! Expected {expected_cols}, got {actual_cols}")
    print("[PASS] Schema validation passed.")

    # 2. Total candidate pairs count
    total_pairs = candidates_df.height
    print(f"[INFO] Total candidate pairs loaded: {total_pairs:,}")

    # 3. Null values check
    null_count = candidates_df.null_count().sum(axis=1)[0]
    if null_count > 0:
        raise ValueError(f"Quality Check Failed: Found {null_count} null values in candidates dataframe!")
    print("[PASS] Null check passed (0 nulls found).")

    # 4. Candidate distribution per source entity
    cand_stats = (
        candidates_df
        .group_by("source1_entity_id")
        .agg(pl.len().alias("cand_count"))
    )
    avg_cands = cand_stats["cand_count"].mean()
    max_cands = cand_stats["cand_count"].max()
    print(f"[INFO] Candidate Statistics -> Avg per S1: {avg_cands:.2f} | Max per S1: {max_cands:,}")


def mine_hard_negatives(candidates_df: pl.DataFrame, val_gt_df: pl.DataFrame) -> pl.DataFrame:
    """
    Extracts hard negative pairs (pairs present in candidates but NOT in ground truth).
    """
    print("\n--- Phase 2: Mining Hard Negatives ---")

    # Format GT positive matches into exploded single pairs
    gt_positives = (
        val_gt_df
        .filter(
            pl.col("matched_entity_ids").is_not_null() 
            & (pl.col("matched_entity_ids") != "")
        )
        .with_columns(pl.col("matched_entity_ids").str.split(","))
        .explode("matched_entity_ids")
        .with_columns(pl.col("matched_entity_ids").str.strip_chars())
        .rename({"matched_entity_ids": "matched_entity_id"})
        .select(["source1_entity_id", "matched_entity_id"])
        .distinct()
    )

    # Perform Anti-Join to extract non-matching candidate pairs
    hard_negatives = candidates_df.join(
        gt_positives,
        on=["source1_entity_id", "matched_entity_id"],
        how="anti"
    )

    print(f"[SUCCESS] Extracted {hard_negatives.height:,} hard negative samples.")
    return hard_negatives


def main():
    print("==================================================")
    print("   PHASE 2: EVALUATION & HARD NEGATIVE MINING    ")
    print("==================================================")

    # 1. Load Validation Split
    if not VAL_GT_PATH.exists():
        print(f"Error: Validation split missing at {VAL_GT_PATH}")
        print("Please run 'python3 src/create_validation_split.py' first.")
        sys.exit(1)

    print(f"Loading validation ground truth from: {VAL_GT_PATH.name}")
    val_gt = pl.read_parquet(VAL_GT_PATH)

    # 2. Check and Load Candidates
    try:
        candidates = load_candidates()
    except FileNotFoundError as e:
        print(f"\n[Awaiting Candidate File] {e}")
        print("Run this script once Member A produces 'data/candidate_pairs.tsv'.")
        sys.exit(0)

    # 3. Quality Control Checks
    run_quality_checks(candidates)

    # 4. Calculate Blocking Recall
    print("\n--- Phase 2: Calculating Blocking Recall ---")
    recall_score = calculate_blocking_recall(candidates, val_gt)
    print(f"--> BLOCKING RECALL (Validation): {recall_score * 100:.2f}% <--")
    
    if recall_score >= 0.985:
        print("[TARGET MET] Recall meets the > 98.5% threshold target!")
    else:
        print("[WARNING] Recall is below 98.5%. Consider expanding blocking keys.")

    # 5. Extract and Save Hard Negatives
    hard_negs = mine_hard_negatives(candidates, val_gt)
    
    # Save output
    HARD_NEGATIVES_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    hard_negs.write_parquet(HARD_NEGATIVES_OUTPUT_PATH)
    print(f"[SAVED] Mined hard negatives written to: {HARD_NEGATIVES_OUTPUT_PATH}")
    print("\nPhase 2 execution complete. Ready for Phase 3 Model Training!")


if __name__ == "__main__":
    main()
PY
