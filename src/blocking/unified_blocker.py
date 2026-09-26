from collections import defaultdict
from typing import Dict, List
import polars as pl

def union_candidates(
    classical: Dict[str, List[str]], 
    embedding: Dict[str, List[str]]
) -> Dict[str, List[str]]:
    """Union of classical + embedding candidates"""
    final = defaultdict(set)
    
    for s1_id, cands in classical.items():
        final[s1_id].update(cands)
        
    for s1_id, cands in embedding.items():
        final[s1_id].update(cands)
        
    return {k: sorted(list(v)) for k, v in final.items()}


def generate_unified_candidates(
    s1_path, s2_path, s3_path,
    classical_func,
    embedding_func,
    top_k_embed=40
):
    print("Loading data...")
    s1 = pl.read_csv(s1_path, separator="\t")
    s2 = pl.read_csv(s2_path, separator="\t")
    s3 = pl.read_csv(s3_path, separator="\t")
    
    print("Running classical blocker...")
    classical_cands = classical_func(s1, s2, s3)
    
    print("Running embedding blocker...")
    embed_cands = embedding_func(s1, s2, s3, top_k=top_k_embed)
    
    print("Taking union...")
    unified = union_candidates(classical_cands, embed_cands)
    
    return unified
