"""
Evaluation Metrics for Business Entity Resolution (Amazon ML Challenge 2026).

Includes:
- calculate_macro_f05: Macro-averaged F_0.5 evaluation over entity groups (handling singletons).
- calculate_blocking_recall: Evaluates candidate generation stage recall.
"""

from typing import Dict, Iterable, Set, Union
import polars as pl


def compute_f05_pair(tp: int, fp: int, fn: int) -> float:
    """
    Computes F_0.5 score for a single query entity given counts of TP, FP, FN.
    F_0.5 = (1 + 0.25) * TP / ( (1 + 0.25) * TP + 0.25 * FN + FP )
    """
    beta_sq = 0.25  # (0.5)^2
    numerator = (1 + beta_sq) * tp
    denominator = (1 + beta_sq) * tp + beta_sq * fn + fp
    
    if denominator == 0:
        return 0.0
    return numerator / denominator


def calculate_macro_f05(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]]
) -> float:
    """
    Computes exact Macro-Averaged F_0.5 score across all Source 1 entities.

    Handling Singletons (Ground Truth set is Empty):
    - Ground Truth = ∅, Prediction = ∅  => Score = 1.0 (True Negative singleton)
    - Ground Truth = ∅, Prediction ≠ ∅  => Score = 0.0 (False Positive singleton)

    Parameters
    ----------
    ground_truth : Dict[str, Set[str]]
        Mapping of source1_entity_id -> set of true matched entity_ids.
    predictions : Dict[str, Set[str]]
        Mapping of source1_entity_id -> set of predicted matched entity_ids.

    Returns
    -------
    float
        Macro-averaged F_0.5 score over all entities in ground_truth.
    """
    total_score = 0.0
    num_entities = len(ground_truth)

    if num_entities == 0:
        return 0.0

    for s1_id, gt_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())

        # Singleton handling
        if len(gt_set) == 0:
            if len(pred_set) == 0:
                total_score += 1.0
            else:
                total_score += 0.0
            continue

        # Standard multi-match / positive ground-truth entity evaluation
        tp = len(gt_set.intersection(pred_set))
        fp = len(pred_set - gt_set)
        fn = len(gt_set - pred_set)

        f05 = compute_f05_pair(tp=tp, fp=fp, fn=fn)
        total_score += f05

    return total_score / num_entities


def calculate_blocking_recall(
    gt_pairs_df: pl.DataFrame,
    candidate_pairs_df: pl.DataFrame
) -> float:
    """
    Calculates candidate blocking recall.

    Parameters
    ----------
    gt_pairs_df : pl.DataFrame
        DataFrame with columns ['source1_entity_id', 'matched_entity_id'].
    candidate_pairs_df : pl.DataFrame
        DataFrame with columns ['source1_entity_id', 'matched_entity_id'].

    Returns
    -------
    float
        Blocking recall percentage in range [0.0, 1.0].
    """
    if gt_pairs_df.height == 0:
        return 0.0

    # Ensure distinct entity pairs
    gt_unique = gt_pairs_df.select(["source1_entity_id", "matched_entity_id"]).unique()
    cand_unique = candidate_pairs_df.select(["source1_entity_id", "matched_entity_id"]).unique()

    hits = (
        gt_unique
        .join(cand_unique, on=["source1_entity_id", "matched_entity_id"], how="inner")
        .height
    )

    return hits / gt_unique.height


def run_unit_tests():
    """Unit tests for verifying metric accuracy."""
    print("Running metrics unit tests...")

    # Test Case 1: Perfect Predictions & Singletons
    gt = {
        "S1-1": {"S2-100", "S3-101"},
        "S1-2": set(),  # Singleton
        "S1-3": {"S2-200"}
    }
    pred_perfect = {
        "S1-1": {"S2-100", "S3-101"},
        "S1-2": set(),  # Correctly empty
        "S1-3": {"S2-200"}
    }
    assert abs(calculate_macro_f05(gt, pred_perfect) - 1.0) < 1e-6, "Test 1 Failed"

    # Test Case 2: Incorrect Singleton Prediction
    pred_bad_singleton = {
        "S1-1": {"S2-100", "S3-101"},
        "S1-2": {"S2-999"},  # FP on singleton
        "S1-3": {"S2-200"}
    }
    expected_f05 = (1.0 + 0.0 + 1.0) / 3.0
    assert abs(calculate_macro_f05(gt, pred_bad_singleton) - expected_f05) < 1e-6, "Test 2 Failed"

    # Test Case 3: Blocking Recall
    gt_df = pl.DataFrame({
        "source1_entity_id": ["S1-1", "S1-1", "S1-2"],
        "matched_entity_id": ["S2-A", "S3-B", "S2-C"]
    })
    cand_df = pl.DataFrame({
        "source1_entity_id": ["S1-1", "S1-2", "S1-3"],
        "matched_entity_id": ["S2-A", "S2-C", "S2-X"]
    })
    recall = calculate_blocking_recall(gt_df, cand_df)
    assert abs(recall - (2.0 / 3.0)) < 1e-6, "Test 3 Failed"

    print("All metric unit tests passed successfully!")


if __name__ == "__main__":
    run_unit_tests()
