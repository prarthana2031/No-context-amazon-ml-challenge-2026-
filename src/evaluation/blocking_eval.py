import polars as pl
from pathlib import Path
from typing import Dict, List, Set

def load_ground_truth(gt_path: str) -> Dict[str, Set[str]]:
    gt = pl.read_csv(gt_path, separator="\t")
    gt = gt.with_columns(
        pl.col("matched_entity_ids")
        .fill_null("")
        .str.split(",")
        .list.eval(pl.element().str.strip_chars())
        .list.eval(pl.element().filter(pl.element() != ""))
        .alias("matched_list")
    )
    return {
        row["source1_entity_id"]: set(row["matched_list"])
        for row in gt.select(["source1_entity_id", "matched_list"]).iter_rows(named=True)
    }

def calculate_blocking_recall(candidates: Dict[str, List[str]], gt: Dict[str, Set[str]]) -> dict:
    total_true = 0
    found = 0
    singletons = 0
    covered_singletons = 0

    for s1_id, true_matches in gt.items():
        cands = set(candidates.get(s1_id, []))
        
        if len(true_matches) == 0:
            singletons += 1
            if len(cands) == 0:
                covered_singletons += 1
            continue

        total_true += len(true_matches)
        found += len(true_matches & cands)

    recall = found / total_true if total_true > 0 else 0.0

    return {
        "blocking_recall": round(recall * 100, 3),
        "true_matches_found": found,
        "total_true_matches": total_true,
        "singletons": singletons,
        "avg_candidates": sum(len(v) for v in candidates.values()) / len(candidates) if candidates else 0
    }

def load_candidate_file(path: str) -> Dict[str, List[str]]:
    df = pl.read_csv(path, separator="\t")
    grouped = df.group_by("source1_entity_id").agg(pl.col("candidate_entity_id").unique().alias("cands"))
    return {r["source1_entity_id"]: r["cands"] for r in grouped.iter_rows(named=True)}
