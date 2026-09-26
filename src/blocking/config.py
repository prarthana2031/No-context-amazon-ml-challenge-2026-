from pathlib import Path

DATA_DIR = Path("data/dataset")
OUTPUT_DIR = Path("output/candidates")
NOTES_DIR = Path("notes/blocking")

TRAIN_S1 = DATA_DIR / "train/train_source1.tsv"
TRAIN_S2 = DATA_DIR / "train/train_source2.tsv"
TRAIN_S3 = DATA_DIR / "train/train_source3.tsv"
GT_PATH  = DATA_DIR / "train/train_ground_truth.tsv"

VAL_GT   = Path("data/val_gt_split.csv")
