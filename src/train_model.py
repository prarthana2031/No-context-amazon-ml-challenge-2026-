from pathlib import Path
import polars as pl
import lightgbm as lgb
from metrics import calculate_macro_f05

# File Paths
TRAIN_GT_PATH = Path("data/train_gt_split.parquet")
VAL_GT_PATH = Path("data/val_gt_split.parquet")
HARD_NEGATIVES_PATH = Path("data/hard_negatives_val.parquet")


def prepare_labeled_pairs() -> pl.DataFrame:
    """Combine GT true positives (y=1) and mined hard negatives (y=0)."""
    if not TRAIN_GT_PATH.exists():
        raise FileNotFoundError(f"Missing {TRAIN_GT_PATH}")
    if not HARD_NEGATIVES_PATH.exists():
        raise FileNotFoundError(
            f"Missing {HARD_NEGATIVES_PATH}. Complete Phase 2 evaluation/mining first!"
        )

    # 1. Load Positive Pairs (y=1)
    train_gt = pl.read_parquet(TRAIN_GT_PATH)
    positives = (
        train_gt
        .filter(pl.col("matched_entity_ids").is_not_null() & (pl.col("matched_entity_ids") != ""))
        .with_columns(pl.col("matched_entity_ids").str.split(","))
        .explode("matched_entity_ids")
        .with_columns(pl.col("matched_entity_ids").str.strip_chars())
        .rename({"matched_entity_ids": "matched_entity_id"})
        .select(["source1_entity_id", "matched_entity_id"])
        .with_columns(pl.lit(1).alias("label"))
    )

    # 2. Load Mined Hard Negatives (y=0)
    hard_negs = (
        pl.read_parquet(HARD_NEGATIVES_PATH)
        .select(["source1_entity_id", "matched_entity_id"])
        .with_columns(pl.lit(0).alias("label"))
    )

    labeled_dataset = pl.concat([positives, hard_negs])
    print(f"Dataset Ready: {positives.height:,} Positives (1s) | {hard_negs.height:,} Negatives (0s)")
    return labeled_dataset


def main():
    print("--- Phase 3: Model Training Setup ---")
    try:
        df = prepare_labeled_pairs()
        print("Dataset successfully assembled! Ready for feature join and LightGBM training.")
    except FileNotFoundError as e:
        print(f"\n[Standing By] {e}")


if __name__ == "__main__":
    main()
PY
