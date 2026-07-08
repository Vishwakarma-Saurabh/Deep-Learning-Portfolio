"""
Semantic Cache - Caches LLM responses based on semantic similarity.
Similar questions get cached answers INSTANTLY.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import hashlib
import json
import time
from typing import Optional, Dict
from sentence_transformers import SentenceTransformer
import numpy as np


class SemanticCache:
    """Cache that matches by semantic similarity, not exact text."""
    
    def __init__(self, similarity_threshold: float = 0.80, ttl_seconds: int = 3600):
        self.threshold = similarity_threshold
        self.ttl = ttl_seconds
        self.cache = []  # List of {embedding, question, answer, timestamp}
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.hits = 0
        self.misses = 0
    
    def get(self, question: str) -> Optional[Dict]:
        """Check if a semantically similar question was cached."""
        question_embedding = self.embedder.encode(question)
        
        best_similarity = 0
        best_match = None
        
        for entry in self.cache:
            # Check TTL
            if time.time() - entry["timestamp"] > self.ttl:
                continue
            
            # Calculate cosine similarity
            similarity = np.dot(question_embedding, entry["embedding"]) / (
                np.linalg.norm(question_embedding) * np.linalg.norm(entry["embedding"])
            )
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = entry
        
        if best_similarity >= self.threshold:
            self.hits += 1
            return {
                "answer": best_match["answer"],
                "cached_question": best_match["question"],
                "similarity": best_similarity,
                "from_cache": True
            }
        
        self.misses += 1
        return None
    
    def set(self, question: str, answer: str):
        """Cache a question-answer pair."""
        embedding = self.embedder.encode(question)
        
        self.cache.append({
            "embedding": embedding,
            "question": question,
            "answer": answer,
            "timestamp": time.time()
        })
        
        # Limit cache size
        if len(self.cache) > 1000:
            self.cache = self.cache[-500:]
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            "cache_size": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.1f}%",
            "estimated_tokens_saved": self.hits * 450  # ~avg tokens per response
        }


# Global cache instance
semantic_cache = SemanticCache()


if __name__ == "__main__":
    # Test
    semantic_cache.set("What is the liability cap?", "The liability cap is $500,000.")
    
    # Exact match
    result = semantic_cache.get("What is the liability cap?")
    print(f"Exact: {result}")
    
    # Similar match
    result = semantic_cache.get("Tell me about the liability limit")
    print(f"Similar: {result}")
    
    # Different question
    result = semantic_cache.get("How to terminate?")
    print(f"Different: {result}")
    
    print(semantic_cache.get_stats())