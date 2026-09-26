import re
import polars as pl

S1 = "/home/ubuntu/dataset/train/train_source1.tsv"
S2 = "/home/ubuntu/dataset/train/train_source2.tsv"
S3 = "/home/ubuntu/dataset/train/train_source3.tsv"


def normalize_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def sorted_token_name(s: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", str(s).lower())
    return " ".join(sorted(tokens))


def phonetic_name(s: str) -> str:
    # Lightweight Soundex-style encoding.
    tokens = re.findall(r"[a-z]+", str(s).lower())
    result = []

    for token in tokens:
        if not token:
            continue

        first = token[0]
        mapping = {
            "bfpv": "1",
            "cgjkqsxz": "2",
            "dt": "3",
            "l": "4",
            "mn": "5",
            "r": "6",
        }

        encoded = []
        previous = ""
        for ch in token[1:]:
            code = ""
            for letters, value in mapping.items():
                if ch in letters:
                    code = value
                    break

            if code and code != previous:
                encoded.append(code)

            previous = code

        result.append(first.upper() + "".join(encoded)[:3].ljust(3, "0"))

    return " ".join(result)


def add_keys(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([
        pl.col("business_name")
        .map_elements(normalize_name, return_dtype=pl.String)
        .alias("name_norm"),
    ]).with_columns([
        pl.col("name_norm").alias("exact_name_key"),
        pl.col("business_name")
        .map_elements(sorted_token_name, return_dtype=pl.String)
        .alias("sorted_token_name_key"),
        pl.col("business_name")
        .map_elements(phonetic_name, return_dtype=pl.String)
        .alias("phonetic_name_key"),
        pl.col("business_address")
        .map_elements(street_number, return_dtype=pl.String)
        .alias("street_number_key"),
        pl.col("business_address")
        .map_elements(postal_code, return_dtype=pl.String)
        .alias("postal_code_key"),
    ])


print("Loading S1...")
s1 = pl.read_csv(S1, separator="\t")
print("Loading S2...")
s2 = pl.read_csv(S2, separator="\t")
print("Loading S3...")
s3 = pl.read_csv(S3, separator="\t")

s1 = add_keys(s1)
s2 = add_keys(s2)
s3 = add_keys(s3)

print("\nS1 rows:", s1.height)
print("S2 rows:", s2.height)
print("S3 rows:", s3.height)

print("\nCountry counts:")
print(s1.group_by("country").len().sort("len", descending=True))

print("\nExact-name key examples:")
print(
    s1.select(["entity_id", "business_name", "country", "exact_name_key"])
    .head(10)
)

print("\nLoading ground truth...")
gt = pl.read_csv(
    "/home/ubuntu/dataset/train/train_ground_truth.tsv",
    separator="\t",
)

gt = gt.with_columns(
    pl.col("matched_entity_ids")
    .fill_null("")
    .str.split(",")
    .alias("matched_ids")
)

# Positive S1 -> S2/S3 pairs
pairs = (
    gt.select(["source1_entity_id", "matched_ids"])
    .explode("matched_ids")
    .filter(pl.col("matched_ids").str.len_chars() > 0)
    .rename({
        "source1_entity_id": "s1_id",
        "matched_ids": "candidate_id",
    })
)

print("Positive pairs:", pairs.height)

# S1 blocking keys
s1_keys = s1.select([
    "entity_id",
    "country",
    "exact_name_key",
]).rename({"entity_id": "s1_id"})

# S2 + S3 blocking keys
s23_keys = pl.concat([
    s2.select(["entity_id", "country", "exact_name_key"]),
    s3.select(["entity_id", "country", "exact_name_key"]),
]).rename({"entity_id": "candidate_id"})

# Join positive pairs to their S1 and true candidate records
positive_with_keys = (
    pairs
    .join(s1_keys, on="s1_id", how="left")
    .join(s23_keys, on="candidate_id", how="left")
)

# Country + exact-name blocking recall
hits = positive_with_keys.filter(
    (pl.col("country") == pl.col("country_right"))
    & (pl.col("exact_name_key") == pl.col("exact_name_key_right"))
)

recall = hits.height / positive_with_keys.height

print(f"\nCountry + exact-name recall: {recall:.4%}")
print(f"Hits: {hits.height:,} / {positive_with_keys.height:,}")

# Sorted-token name blocking recall
s1_sorted = s1.select([
    "entity_id",
    "country",
    "sorted_token_name_key",
]).rename({"entity_id": "s1_id"})

s23_sorted = pl.concat([
    s2.select(["entity_id", "country", "sorted_token_name_key"]),
    s3.select(["entity_id", "country", "sorted_token_name_key"]),
]).rename({"entity_id": "candidate_id"})

positive_sorted = (
    pairs
    .join(s1_sorted, on="s1_id", how="left")
    .join(s23_sorted, on="candidate_id", how="left")
)

sorted_hits = positive_sorted.filter(
    (pl.col("country") == pl.col("country_right"))
    & (
        pl.col("sorted_token_name_key")
        == pl.col("sorted_token_name_key_right")
    )
)

sorted_recall = sorted_hits.height / positive_sorted.height

print(f"\nCountry + sorted-token-name recall: {sorted_recall:.4%}")
print(f"Hits: {sorted_hits.height:,} / {positive_sorted.height:,}")

# Phonetic name blocking recall
s1_phonetic = s1.select([
    "entity_id",
    "country",
    "phonetic_name_key",
]).rename({"entity_id": "s1_id"})

s23_phonetic = pl.concat([
    s2.select(["entity_id", "country", "phonetic_name_key"]),
    s3.select(["entity_id", "country", "phonetic_name_key"]),
]).rename({"entity_id": "candidate_id"})

positive_phonetic = (
    pairs
    .join(s1_phonetic, on="s1_id", how="left")
    .join(s23_phonetic, on="candidate_id", how="left")
)

phonetic_hits = positive_phonetic.filter(
    (pl.col("country") == pl.col("country_right"))
    & (
        pl.col("phonetic_name_key")
        == pl.col("phonetic_name_key_right")
    )
)

phonetic_recall = phonetic_hits.height / positive_phonetic.height

print(f"\nCountry + phonetic-name recall: {phonetic_recall:.4%}")
print(f"Hits: {phonetic_hits.height:,} / {positive_phonetic.height:,}")

# Address blocking keys
def street_number(s: str) -> str:
    match = re.match(r"\s*(\d+)", str(s))
    return match.group(1) if match else ""

def postal_code(s: str) -> str:
    match = re.search(r"\b(\d{5}(?:-\d{4})?|\d{6})\b", str(s))
    return match.group(1) if match else ""

