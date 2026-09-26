import polars as pl
from pathlib import Path

DATA = Path("data/dataset/train")
OUT = Path("data/parquet")
OUT.mkdir(parents=True, exist_ok=True)

files = {
    "source1": DATA / "train_source1.tsv",
    "source2": DATA / "train_source2.tsv",
    "source3": DATA / "train_source3.tsv",
    "ground_truth": DATA / "train_ground_truth.tsv",
}

for name, path in files.items():
    print(f"Converting {name}...")
    df = pl.read_csv(path, separator="\t")
    out_path = OUT / f"{name}.parquet"
    df.write_parquet(out_path, compression="zstd")
    print(f"  → Saved {out_path} ({df.height:,} rows)")

print("\nDone! All files converted to Parquet.")
