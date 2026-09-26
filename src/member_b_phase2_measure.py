import re
import polars as pl

S1 = "/home/ubuntu/dataset/train/train_source1.tsv"
S2 = "/home/ubuntu/dataset/train/train_source2.tsv"
S3 = "/home/ubuntu/dataset/train/train_source3.tsv"
GT = "/home/ubuntu/dataset/train/train_ground_truth.tsv"

def norm_name(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

def sorted_name(s):
    tokens = re.findall(r"[a-z0-9]+", str(s).lower())
    return " ".join(sorted(tokens))

def phonetic(s):
    tokens = re.findall(r"[a-z]+", str(s).lower())
    mapping = {
        "bfpv": "1", "cgjkqsxz": "2", "dt": "3",
        "l": "4", "mn": "5", "r": "6"
    }
    out = []
    for token in tokens:
        if not token:
            continue
        codes = []
        prev = ""
        for ch in token[1:]:
            code = next((v for letters, v in mapping.items() if ch in letters), "")
            if code and code != prev:
                codes.append(code)
            prev = code
        out.append(token[0].upper() + "".join(codes)[:3].ljust(3, "0"))
    return " ".join(out)

def street(s):
    m = re.match(r"\s*(\d+)", str(s))
    return m.group(1) if m else ""

def postal(s):
    m = re.search(r"\b(\d{5}(?:-\d{4})?|\d{6})\b", str(s))
    return m.group(1) if m else ""

def prepare(path):
    return (
        pl.read_csv(path, separator="\t")
        .with_columns([
            pl.col("business_name").map_elements(norm_name, return_dtype=pl.String).alias("name_norm"),
            pl.col("business_name").map_elements(sorted_name, return_dtype=pl.String).alias("sorted_key"),
            pl.col("business_name").map_elements(phonetic, return_dtype=pl.String).alias("phonetic_key"),
            pl.col("business_address").map_elements(street, return_dtype=pl.String).alias("street_key"),
            pl.col("business_address").map_elements(postal, return_dtype=pl.String).alias("postal_key"),
        ])
        .with_columns([
            pl.col("name_norm").str.slice(0, 5).alias("name_prefix_key"),
        ])
    )

print("Loading S1...")
s1 = prepare(S1)
print("Loading S2...")
s2 = prepare(S2)
print("Loading S3...")
s3 = prepare(S3)

print("Loading ground truth...")
gt = pl.read_csv(GT, separator="\t").with_columns(
    pl.col("matched_entity_ids").fill_null("").str.split(",").alias("matched_ids")
)

pairs = (
    gt.select(["source1_entity_id", "matched_ids"])
    .explode("matched_ids")
    .filter(pl.col("matched_ids").str.len_chars() > 0)
    .rename({
        "source1_entity_id": "s1_id",
        "matched_ids": "candidate_id",
    })
)

print(f"Positive pairs: {pairs.height:,}")

# Keep only columns needed for the evaluation.
s1k = s1.select([
    "entity_id", "country",
    "name_prefix_key", "name_norm",
    "sorted_key", "phonetic_key",
    "street_key", "postal_key"
]).rename({"entity_id": "s1_id"})

s23k = pl.concat([
    s2.select([
        "entity_id", "country",
        "name_prefix_key", "name_norm",
        "sorted_key", "phonetic_key",
        "street_key", "postal_key"
    ]),
    s3.select([
        "entity_id", "country",
        "name_prefix_key", "name_norm",
        "sorted_key", "phonetic_key",
        "street_key", "postal_key"
    ])
]).rename({"entity_id": "candidate_id"})

print("Joining positive pairs to blocking keys once...")
eval_df = (
    pairs
    .join(s1k, on="s1_id", how="left")
    .join(s23k, on="candidate_id", how="left", suffix="_cand")
)

same_country = pl.col("country") == pl.col("country_cand")

conditions = {
    "name_prefix": same_country & (
        pl.col("name_prefix_key") == pl.col("name_prefix_key_cand")
    ),
    "exact_name": same_country & (
        pl.col("name_norm") == pl.col("name_norm_cand")
    ),
    "sorted_token": same_country & (
        pl.col("sorted_key") == pl.col("sorted_key_cand")
    ),
    "phonetic_name": same_country & (
        pl.col("phonetic_key") == pl.col("phonetic_key_cand")
    ),
    "street_number": same_country & (
        pl.col("street_key") != ""
    ) & (
        pl.col("street_key") == pl.col("street_key_cand")
    ),
    "postal_code": same_country & (
        pl.col("postal_key") != ""
    ) & (
        pl.col("postal_key") == pl.col("postal_key_cand")
    ),
}

print("\n=== INDIVIDUAL BLOCKING RECALL ===")
results = []

for name, condition in conditions.items():
    hits = eval_df.filter(condition).height
    recall = hits / eval_df.height
    results.append((name, hits, recall))
    print(f"{name:16s}: {recall:.4%}  ({hits:,} / {eval_df.height:,})")

combined_condition = None
for condition in conditions.values():
    combined_condition = condition if combined_condition is None else (
        combined_condition | condition
    )

combined_hits = eval_df.filter(combined_condition).height
combined_recall = combined_hits / eval_df.height

print("\n=== COMBINED CLASSICAL BLOCKING ===")
print(f"Recall: {combined_recall:.4%}")
print(f"Hits:   {combined_hits:,} / {eval_df.height:,}")

out = pl.DataFrame(
    results,
    schema=["blocking_key", "hits", "recall"],
    orient="row",
)

out = pl.concat([
    out,
    pl.DataFrame({
        "blocking_key": ["combined_classical"],
        "hits": [combined_hits],
        "recall": [combined_recall],
    })
])

out.write_csv("notes/phase2_blocking_recall.csv")

print("\nSaved: notes/phase2_blocking_recall.csv")
