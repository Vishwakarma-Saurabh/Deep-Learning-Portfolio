"""
Re-Ranker - Second-pass scoring for more accurate results.
Takes top 10 chunks from initial search, re-scores with cross-attention.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List, Dict
from sentence_transformers import CrossEncoder


class ReRanker:
    """Cross-encoder re-ranking for better precision."""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """Load cross-encoder model (runs on CPU, ~80MB)."""
        print(f"Loading re-ranker: {model_name}")
        self.model = CrossEncoder(model_name)
        print("Re-ranker loaded!")
    
    def rerank(self, query: str, candidates: List[Dict], top_k: int = 3) -> List[Dict]:

        if not candidates:
            return []
        
        # Prepare pairs for cross-encoder
        pairs = [(query, c.get("text", "")[:500]) for c in candidates]
        
        # Get relevance scores
        scores = self.model.predict(pairs)
        
        # Add scores and sort
        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = float(score)
        
        # Sort by rerank score descending
        candidates.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
        
        # Mark which ones are from re-ranking
        for i, c in enumerate(candidates[:top_k]):
            c["reranked_position"] = i + 1
            c["reranked"] = True
        
        return candidates[:top_k]


# Global instance
reranker = ReRanker()


def rerank_results(query: str, candidates: List[Dict], top_k: int = 3) -> List[Dict]:
    """Convenience function for re-ranking."""
    return reranker.rerank(query, candidates, top_k)


if __name__ == "__main__":
    # Test with sample data
    test_candidates = [
        {"text": "The liability cap is set at $500,000 for damages.", "score": 0.82},
        {"text": "Payment shall be made within 30 days of invoice.", "score": 0.78},
        {"text": "Section 4.2 covers liability limitations and damage caps.", "score": 0.75},
    ]
    
    results = rerank_results("What is the liability cap amount?", test_candidates)
    for i, r in enumerate(results):
        print(f"{i+1}. Score: {r['rerank_score']:.3f} | {r['text'][:80]}...")