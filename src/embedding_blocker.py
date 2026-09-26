import os
import random
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

BASE = "data/dataset/train"
OUT = "output"
TOP_K = 20

S1_LIMIT = 2000
RANDOM_SOURCE_LIMIT = 100000
BATCH_SIZE = 128

os.makedirs(OUT, exist_ok=True)

print("=== MEMBER C PHASE 2: POSITIVE-INCLUSIVE ANN BLOCKING ===")

# ---------------------------------------------------------
# 1. Load ground truth first, then fetch the exact S1 records
# ---------------------------------------------------------
print("Loading ground truth...")

gt = pd.read_csv(
    f"{BASE}/train_ground_truth.tsv",
    sep="\\t",
    dtype=str
).fillna("")

gt = gt.iloc[:S1_LIMIT].copy()

needed_s1_ids = set(gt["source1_entity_id"])

print(f"Ground-truth S1 queries: {len(needed_s1_ids)}")

print("Scanning Source 1 for exact S1 records...")

s1_parts = []

for chunk in pd.read_csv(
    f"{BASE}/train_source1.tsv",
    sep="\\t",
    dtype=str,
    chunksize=100000
):
    chunk = chunk.fillna("")
    found = chunk[chunk["entity_id"].isin(needed_s1_ids)]

    if len(found):
        s1_parts.append(found)

s1 = pd.concat(s1_parts, ignore_index=True)

# Keep exactly the same order as ground truth
s1 = (
    gt[["source1_entity_id"]]
    .merge(
        s1,
        left_on="source1_entity_id",
        right_on="entity_id",
        how="inner"
    )
    .rename(columns={"entity_id": "entity_id"})
)

print(f"Exact S1 records found: {len(s1)}")

if len(s1) != len(gt):
    missing = len(gt) - len(s1)
    raise RuntimeError(f"Missing {missing} S1 records")

# ---------------------------------------------------------
# 2. Collect all GT entity IDs needed by these S1 records
# ---------------------------------------------------------
positive_ids = set()

for value in gt["matched_entity_ids"]:
    if pd.isna(value):
        continue

    for x in str(value).replace(",", "|").split("|"):
        x = x.strip()
        if x:
            positive_ids.add(x)

print(f"Unique GT positive IDs: {len(positive_ids)}")

# ---------------------------------------------------------
# 3. Load source 2 and source 3 records containing positives
#    + random distractors
# ---------------------------------------------------------
def load_positive_and_random(path, source_name):
    print(f"Scanning {source_name}...")

    positive_rows = []
    random_rows = []

    # Keep a deterministic reservoir of random records.
    rng = random.Random(42)
    reservoir = []

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        chunksize=100000
    ):
        chunk = chunk.fillna("")

        # Positive records
        mask = chunk["entity_id"].isin(positive_ids)
        if mask.any():
            positive_rows.append(chunk[mask])

        # Reservoir sampling for random distractors
        for _, row in chunk.iterrows():
            if len(reservoir) < RANDOM_SOURCE_LIMIT:
                reservoir.append(row.to_dict())
            else:
                j = rng.randint(0, len(reservoir))
                if j < RANDOM_SOURCE_LIMIT:
                    reservoir[j] = row.to_dict()

    positive_df = (
        pd.concat(positive_rows, ignore_index=True)
        if positive_rows
        else pd.DataFrame(columns=["entity_id", "business_name",
                                   "business_address", "country"])
    )

    random_df = pd.DataFrame(reservoir)

    print(
        f"{source_name}: positives={len(positive_df)}, "
        f"random={len(random_df)}"
    )

    return positive_df, random_df


s2_pos, s2_random = load_positive_and_random(
    f"{BASE}/train_source2.tsv",
    "Source 2"
)

s3_pos, s3_random = load_positive_and_random(
    f"{BASE}/train_source3.tsv",
    "Source 3"
)

# ---------------------------------------------------------
# 4. Build ANN pool
# ---------------------------------------------------------
source = pd.concat(
    [s2_pos, s3_pos, s2_random, s3_random],
    ignore_index=True
)

source = source.drop_duplicates(
    subset=["entity_id"]
).reset_index(drop=True)

print(f"ANN source pool: {len(source)}")

# ---------------------------------------------------------
# 5. Text representation
# ---------------------------------------------------------
def make_text(df):
    return (
        "business: "
        + df["business_name"].astype(str).str.lower()
        + " address: "
        + df["business_address"].astype(str).str.lower()
        + " country: "
        + df["country"].astype(str).str.lower()
    ).str.replace(r"\s+", " ", regex=True).tolist()


source_text = make_text(source)
s1_text = make_text(s1)

# ---------------------------------------------------------
# 6. Load embedding model
# ---------------------------------------------------------
print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------------------------------------------------
# 7. Source embeddings
# ---------------------------------------------------------
print("Creating source embeddings...")

source_embeddings = model.encode(
    source_text,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    normalize_embeddings=True
)

source_embeddings = np.asarray(
    source_embeddings,
    dtype="float32"
)

# ---------------------------------------------------------
# 8. FAISS index
# ---------------------------------------------------------
print("Building FAISS index...")

index = faiss.IndexFlatIP(
    source_embeddings.shape[1]
)

index.add(source_embeddings)

print(f"FAISS vectors indexed: {index.ntotal}")

# ---------------------------------------------------------
# 9. S1 embeddings
# ---------------------------------------------------------
print("Creating S1 embeddings...")

s1_embeddings = model.encode(
    s1_text,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    normalize_embeddings=True
)

s1_embeddings = np.asarray(
    s1_embeddings,
    dtype="float32"
)

# ---------------------------------------------------------
# 10. ANN search
# ---------------------------------------------------------
print("Running ANN search...")

scores, indices = index.search(
    s1_embeddings,
    TOP_K
)

# ---------------------------------------------------------
# 11. Evaluate recall
# ---------------------------------------------------------
source_ids = source["entity_id"].astype(str).tolist()

hits = 0
candidate_rows = []

for i in range(len(s1)):
    candidates = [
        source_ids[j]
        for j in indices[i]
        if j >= 0
    ]

    gt_row = gt[gt["source1_entity_id"] == s1.iloc[i]["entity_id"]]

    if len(gt_row) == 0:
        true_ids = set()
    else:
        true_ids = {
            x.strip()
            for x in str(gt_row.iloc[0]["matched_entity_ids"]).split(",")
            if x.strip()
        }

    if true_ids.intersection(candidates):
        hits += 1

    for rank, entity_id in enumerate(candidates, start=1):
        candidate_rows.append({
            "source1_entity_id": s1.iloc[i]["entity_id"],
            "candidate_entity_id": entity_id,
            "ann_rank": rank
        })

recall = hits / len(s1)

# ---------------------------------------------------------
# 12. Save candidates
# ---------------------------------------------------------
candidate_file = (
    f"{OUT}/member_c_ann_candidate_pairs.tsv"
)

pd.DataFrame(candidate_rows).to_csv(
    candidate_file,
    sep="\t",
    index=False
)

print()
print("========== ANN RESULT ==========")
print(f"S1 evaluated:       {len(s1):,}")
print(f"ANN source pool:    {len(source):,}")
print(f"Top-K:              {TOP_K}")
print(f"Queries with hit:   {hits:,}")
print(f"Blocking recall:    {recall:.2%}")
print(f"Candidate pairs:    {len(candidate_rows):,}")
print(f"Candidate file:     {candidate_file}")
print("================================")
