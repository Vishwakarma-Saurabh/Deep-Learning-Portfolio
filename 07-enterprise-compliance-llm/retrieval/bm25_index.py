"""
BM25 Sparse Index - Keyword-based search for exact matches.
Handles queries like "Section 4.2" that embeddings might miss.
"""

import re
import math
from collections import defaultdict
from typing import List, Dict


class BM25Index:
    """Simple BM25 implementation for keyword search."""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents = []
        self.doc_lengths = []
        self.avg_doc_length = 0
        self.term_freqs = defaultdict(dict)
        self.doc_freqs = defaultdict(int)
        self.total_docs = 0
    
    def tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        return text.split()
    
    def add_document(self, doc_id: str, text: str):
        """Add a document to the index."""
        tokens = self.tokenize(text)
        doc_index = len(self.documents)
        
        self.documents.append({"id": doc_id, "text": text})
        self.doc_lengths.append(len(tokens))
        self.total_docs += 1
        self.avg_doc_length = sum(self.doc_lengths) / self.total_docs
        
        # Count term frequencies for this document
        term_counts = defaultdict(int)
        for token in tokens:
            term_counts[token] += 1
        
        self.term_freqs[doc_index] = dict(term_counts)
        
        # Count document frequencies
        for token in set(tokens):
            self.doc_freqs[token] += 1
    
    def score(self, query: str) -> List[Dict]:
        """Score all documents against a query."""
        query_tokens = self.tokenize(query)
        scores = []
        
        for doc_index in range(self.total_docs):
            score = 0
            doc_len = self.doc_lengths[doc_index]
            
            for token in query_tokens:
                if token not in self.doc_freqs:
                    continue
                
                # Term frequency in document
                tf = self.term_freqs[doc_index].get(token, 0)
                
                # Inverse document frequency
                df = self.doc_freqs[token]
                idf = math.log((self.total_docs - df + 0.5) / (df + 0.5) + 1)
                
                # BM25 formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length)
                score += idf * numerator / denominator
            
            if score > 0:
                scores.append({
                    "doc_id": self.documents[doc_index]["id"],
                    "text": self.documents[doc_index]["text"][:200],
                    "score": score,
                    "type": "keyword"
                })
        
        scores.sort(key=lambda x: x["score"], reverse=True)
        return scores


# Global instance
bm25_index = BM25Index()


def add_to_bm25(doc_id: str, text: str):
    """Add document to BM25 index."""
    bm25_index.add_document(doc_id, text)


def search_bm25(query: str, top_k: int = 5) -> List[Dict]:
    """Search BM25 index."""
    return bm25_index.score(query)[:top_k]


if __name__ == "__main__":
    # Test
    add_to_bm25("doc1", "The liability cap is $500,000 for damages under Section 4.2.")
    add_to_bm25("doc2", "Payment terms are Net 30 days from invoice date.")
    
    results = search_bm25("Section 4.2 liability")
    for r in results:
        print(f"Score: {r['score']:.3f} | {r['text'][:80]}...")