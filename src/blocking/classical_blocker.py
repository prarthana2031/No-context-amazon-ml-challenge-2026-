import polars as pl
from pathlib import Path
from typing import Dict, List
import gc

def normalize_expr(col: str) -> pl.Expr:
    return (
        pl.col(col)
        .str.to_lowercase()
        .str.replace_all(r"[^\w\s]", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )

def add_keys(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns([
        normalize_expr("business_name").alias("name_norm"),
        normalize_expr("business_address").alias("addr_norm"),
        pl.col("country").fill_null("").str.to_uppercase().alias("cty"),
    ])

    # Stricter name prefix (8 chars + must have some content)
    df = df.with_columns(
        pl.col("name_norm")
        .str.replace_all(r"\b(inc|corp|ltd|pvt|llc|limited|private|company|incorporated|corporation)\b", "")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
        .str.slice(0, 8)
        .alias("np")
    )

    # Better sorted tokens (require at least 2 meaningful tokens)
    df = df.with_columns(
        pl.col("name_norm")
        .str.replace_all(r"\b(inc|corp|ltd|pvt|llc|limited|private|company|incorporated|corporation)\b", "")
        .str.split(" ")
        .list.eval(pl.element().filter(pl.element().str.len_chars() > 2))
        .list.sort()
        .list.head(3)
        .list.join(" ")
        .alias("st")
    )

    # Street key - keep only if number exists
    df = df.with_columns(
        pl.when(pl.col("addr_norm").str.contains(r"^\d+"))
        .then(
            pl.col("addr_norm").str.extract(r"^(\d+)\s+([a-z0-9]+)", 1)
            .add("_")
            .add(pl.col("addr_norm").str.extract(r"^(\d+)\s+([a-z0-9]+)", 2))
        )
        .otherwise(None)
        .alias("sk")
    )

    # Postal key
    df = df.with_columns(
        pl.col("addr_norm").str.extract(r"\b(\d{5,6})\b", 1).alias("pk")
    )

    return df.select(["entity_id", "cty", "np", "st", "sk", "pk"])


def process_one_key(s1: pl.DataFrame, others: pl.DataFrame, key: str, output_dir: Path, 
                    chunk_size: int = 250_000, max_per_s1: int = 40):
    print(f"\n=== Processing key: {key} (max {max_per_s1} cands/S1) ===")
    output_file = output_dir / f"pairs_{key}.parquet"
    
    if output_file.exists():
        print(f"  Already exists → skipping")
        return

    all_parts = []
    total = s1.height

    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        print(f"  Chunk {start:,} → {end:,}")

        chunk = s1.slice(start, end - start)
        left = chunk.filter(pl.col(key).is_not_null() & (pl.col(key) != "")).select(["entity_id", "cty", key])
        right = others.filter(pl.col(key).is_not_null() & (pl.col(key) != "")).select(["entity_id", "cty", key])

        if left.height == 0 or right.height == 0:
            continue

        joined = (
            left.join(right, on=["cty", key], how="inner")
            .select([
                pl.col("entity_id").alias("s1_id"),
                pl.col("entity_id_right").alias("cand_id")
            ])
        )

        # CRITICAL: Cap candidates per S1 for this key
        joined = (
            joined
            .group_by("s1_id")
            .agg(pl.col("cand_id").unique().head(max_per_s1))
            .explode("cand_id")
        )

        all_parts.append(joined)
        del left, right, joined, chunk
        gc.collect()

    if all_parts:
        final = pl.concat(all_parts).unique()
        final.write_parquet(output_file)
        print(f"  Saved {final.height:,} pairs")
        del final
    gc.collect()


def get_classical_candidates(s1_df, s2_df, s3_df) -> Dict[str, List[str]]:
    output_dir = Path("output/tmp_keys")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Adding keys...")
    s1 = add_keys(s1_df)
    others = pl.concat([add_keys(s2_df), add_keys(s3_df)])
    print(f"Others size: {others.height:,}")

    # Order: most selective first
    # pk = postal (very good)
    # sk = street (now stricter)
    # np = name prefix 8 chars
    # st = sorted tokens
    for key, max_cands in [("pk", 30), ("sk", 25), ("np", 100), ("st", 30)]:
        process_one_key(s1, others, key, output_dir, chunk_size=250_000, max_per_s1=max_cands)

    print("\nCombining all keys...")
    files = list(output_dir.glob("pairs_*.parquet"))
    if not files:
        return {sid: [] for sid in s1["entity_id"].to_list()}

    all_pairs = pl.concat([pl.read_parquet(f) for f in files]).unique()
    
    grouped = all_pairs.group_by("s1_id").agg(pl.col("cand_id").unique().alias("cands"))
    result = {r["s1_id"]: sorted(r["cands"]) for r in grouped.iter_rows(named=True)}

    for sid in s1["entity_id"].to_list():
        if sid not in result:
            result[sid] = []

    print(f"Final S1 covered: {len(result):,}")
    return result
