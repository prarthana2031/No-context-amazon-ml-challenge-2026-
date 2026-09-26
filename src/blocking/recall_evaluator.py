import polars as pl
from pathlib import Path
from typing import Dict, Set, List

def load_ground_truth(gt_path: Path) -> Dict[str, Set[str]]:
    """
    Loads ground truth and returns:
    { source1_entity_id : set of true matching entity_ids }
    """
    gt = pl.read_csv(gt_path, separator="\t")
    
    gt = gt.with_columns(
        pl.col("matched_entity_ids")
        .fill_null("")
        .str.split(",")
        .list.eval(pl.element().str.strip_chars())
        .list.eval(pl.element().filter(pl.element() != ""))
    )
    
    result = {}
    for row in gt.iter_rows(named=True):
        result[row["source1_entity_id"]] = set(row["matched_entity_ids"])
    
    return result


def calculate_blocking_recall(
    candidate_dict: Dict[str, List[str]], 
    gt_dict: Dict[str, Set[str]]
) -> dict:
    """
    Calculates blocking recall.
    
    Returns:
        - pair_recall: % of true matching pairs that were found in candidates
        - entity_recall: % of Source-1 entities whose ALL true matches were found
    """
    total_true = 0
    found_true = 0
    entities_with_matches = 0
    fully_recovered = 0

    for s1_id, true_matches in gt_dict.items():
        if len(true_matches) == 0:
            continue
            
        entities_with_matches += 1
        total_true += len(true_matches)
        
        cands = set(candidate_dict.get(s1_id, []))
        recovered = true_matches.intersection(cands)
        found_true += len(recovered)
        
        if recovered == true_matches:
            fully_recovered += 1

    pair_recall = found_true / total_true if total_true > 0 else 0.0
    entity_recall = fully_recovered / entities_with_matches if entities_with_matches > 0 else 0.0

    return {
        "pair_recall": round(pair_recall, 5),
        "entity_recall": round(entity_recall, 5),
        "total_true_pairs": total_true,
        "found_true_pairs": found_true,
        "entities_with_matches": entities_with_matches,
        "fully_recovered_entities": fully_recovered
    }


def print_recall_report(metrics: dict):
    print("\n===== BLOCKING RECALL REPORT =====")
    print(f"Pair Recall           : {metrics['pair_recall']*100:.2f}%")
    print(f"Entity Recall         : {metrics['entity_recall']*100:.2f}%")
    print(f"True pairs found      : {metrics['found_true_pairs']} / {metrics['total_true_pairs']}")
    print(f"Fully recovered S1    : {metrics['fully_recovered_entities']} / {metrics['entities_with_matches']}")
    print("==================================\n")
