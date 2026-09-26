import polars as pl
from pathlib import Path
from typing import Dict, List

def save_candidate_pairs(
    candidate_dict: Dict[str, List[str]], 
    output_path: Path
):
    rows = []
    for s1_id, cands in candidate_dict.items():
        # Clean candidates
        cleaned = [c for c in dict.fromkeys(cands) if c.startswith(("S2-", "S3-"))]
        rows.append({
            "source1_entity_id": s1_id,
            "candidate_entity_ids": ",".join(cleaned)
        })
    
    df = pl.DataFrame(rows)
    df.write_csv(output_path, separator="\t")
    print(f"Saved {len(df):,} rows → {output_path}")


def get_average_candidates(candidate_dict: Dict[str, List[str]]) -> float:
    if not candidate_dict:
        return 0.0
    return sum(len(v) for v in candidate_dict.values()) / len(candidate_dict)
