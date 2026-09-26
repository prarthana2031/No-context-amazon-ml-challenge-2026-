import polars as pl
from pathlib import Path


def save_parquet(df: pl.DataFrame, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    print(f"Saved {path} | shape = {df.shape}")


def load_parquet(path: str) -> pl.DataFrame:
    return pl.read_parquet(path)
