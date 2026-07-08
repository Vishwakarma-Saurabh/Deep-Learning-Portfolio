"""
Hybrid Search - Combines dense (embeddings) and sparse (BM25) search.
Returns the best results from both methods.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List, Dict
from rag.retriever import retrieve_chunks
from retrieval.bm25_index import search_bm25, add_to_bm25


def hybrid_search(query: str, top_k: int = 5, dense_weight: float = 0.7) -> List[Dict]:

    # Get results from both methods
    dense_results = retrieve_chunks(query, top_k=top_k * 2)
    sparse_results = search_bm25(query, top_k=top_k * 2)
    
    # Normalize scores to 0-1 range
    if dense_results:
        max_dense = max(r["score"] for r in dense_results)
        for r in dense_results:
            r["normalized_score"] = r["score"] / max_dense if max_dense > 0 else 0
            r["source_type"] = "dense"
    
    if sparse_results:
        max_sparse = max(r["score"] for r in sparse_results) if sparse_results else 1
        for r in sparse_results:
            r["normalized_score"] = r["score"] / max_sparse if max_sparse > 0 else 0
            r["source_type"] = "sparse"
            # Normalize format
            if "filename" not in r:
                r["filename"] = r.get("doc_id", "unknown")
    
    # Merge and deduplicate by text similarity
    merged = []
    seen_texts = set()
    
    for r in dense_results + sparse_results:
        # Simple dedup: check first 50 chars
        text_key = r.get("text", "")[:50]
        if text_key in seen_texts:
            continue
        seen_texts.add(text_key)
        
        # Calculate weighted score
        if r.get("source_type") == "dense":
            r["final_score"] = r["normalized_score"] * dense_weight
        else:
            r["final_score"] = r["normalized_score"] * (1 - dense_weight)
        
        merged.append(r)
    
    # Sort by final score
    merged.sort(key=lambda x: x.get("final_score", 0), reverse=True)
    
    return merged[:top_k]


def index_document_for_hybrid(filename: str, chunks: List[Dict]):
    """Index document in BM25 for hybrid search."""
    for i, chunk in enumerate(chunks):
        doc_id = f"{filename}_chunk_{i}"
        add_to_bm25(doc_id, chunk["text"])


if __name__ == "__main__":
    # Test hybrid search
    results = hybrid_search("liability cap Section 4.2")
    for r in results:
        source = r.get("source_type", "unknown")
        score = r.get("final_score", r.get("score", 0))
        print(f"[{source}] Score: {score:.3f} | {r.get('text', '')[:80]}...")