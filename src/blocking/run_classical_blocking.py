import sys
from pathlib import Path
import polars as pl

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.blocking.classical_blocker import get_classical_candidates
from src.blocking.candidate_generator import save_candidate_pairs, get_average_candidates
from src.blocking.recall_evaluator import load_ground_truth, calculate_blocking_recall, print_recall_report
from src.blocking.config import TRAIN_S1, TRAIN_S2, TRAIN_S3, GT_PATH, OUTPUT_DIR

def main():
    print("=== Classical Blocking - Generating Candidates ===\n")

    print("Loading Source 1...")
    s1 = pl.read_csv(TRAIN_S1, separator="\t")
    print("Loading Source 2...")
    s2 = pl.read_csv(TRAIN_S2, separator="\t")
    print("Loading Source 3...")
    s3 = pl.read_csv(TRAIN_S3, separator="\t")

    print(f"\nS1: {s1.height:,} | S2: {s2.height:,} | S3: {s3.height:,}")

    print("\nRunning classical blocker (this will take time)...")
    candidates = get_classical_candidates(s1, s2, s3)

    avg = get_average_candidates(candidates)
    print(f"\nAverage candidates per S1 entity: {avg:.2f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "candidate_pairs_train.tsv"
    save_candidate_pairs(candidates, output_path)

    print("\nCalculating blocking recall on full GT...")
    gt_dict = load_ground_truth(GT_PATH)
    metrics = calculate_blocking_recall(candidates, gt_dict)
    print_recall_report(metrics)

    print(f"\n✅ Candidate file ready for Member D:")
    print(f"   {output_path.resolve()}")

if __name__ == "__main__":
    main()
