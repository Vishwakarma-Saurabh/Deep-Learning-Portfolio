"""
Retriever - Searches vector database for relevant chunks.
Uses scroll() for maximum compatibility with all qdrant-client versions.
"""

import os
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = SentenceTransformer(
    os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
)

QDRANT_CLIENT = QdrantClient(
    url=os.getenv("QDRANT_URL", "http://localhost:6333"),
    api_key=os.getenv("QDRANT_API_KEY", "")
)

COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "compliance_docs")


def retrieve_chunks(query: str, top_k: int = 3) -> List[Dict]:
    """
    Search for relevant chunks using cosine similarity.
    """
    # Step 1: Embed the query
    query_vector = EMBEDDING_MODEL.encode(query)
    
    # Step 2: Get all points from collection
    try:
        collection_info = QDRANT_CLIENT.get_collection(COLLECTION_NAME)
        total_points = collection_info.points_count
        
        if total_points == 0:
            print("No documents in collection")
            return []
        
        # Scroll gets all points with vectors and payload
        all_points, _ = QDRANT_CLIENT.scroll(
            collection_name=COLLECTION_NAME,
            limit=total_points,
            with_payload=True,
            with_vectors=True
        )
        
    except Exception as e:
        print(f"Error accessing Qdrant: {e}")
        return []
    
    # Step 3: Calculate cosine similarity for each point
    scored_points = []
    query_norm = np.linalg.norm(query_vector)
    
    for point in all_points:
        point_vector = np.array(point.vector)
        point_norm = np.linalg.norm(point_vector)
        
        if point_norm == 0 or query_norm == 0:
            similarity = 0.0
        else:
            similarity = float(np.dot(query_vector, point_vector) / (query_norm * point_norm))
        
        scored_points.append({
            "text": point.payload.get("text", ""),
            "filename": point.payload.get("filename", "unknown"),
            "chunk_index": point.payload.get("chunk_index", 0),
            "score": similarity,
            "char_count": point.payload.get("char_count", 0)
        })
    
    # Step 4: Sort by score (highest first) and return top_k
    scored_points.sort(key=lambda x: x["score"], reverse=True)
    
    return scored_points[:top_k]


def format_context(chunks: List[Dict]) -> str:
    """Format retrieved chunks into a context string."""
    if not chunks:
        return ""
    
    context_parts = []
    total_chars = 0
    max_chars = 3000
    
    for chunk in chunks:
        source = f"[Source: {chunk['filename']}, Chunk {chunk['chunk_index']}]"
        part = f"{source}\n{chunk['text']}\n"
        
        if total_chars + len(part) > max_chars:
            break
            
        context_parts.append(part)
        total_chars += len(part)
    
    return "\n".join(context_parts)


# Test independently
if __name__ == "__main__":
    query = "What is the liability cap?"
    chunks = retrieve_chunks(query)
    
    print(f"Found {len(chunks)} relevant chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i+1} (Score: {chunk['score']:.3f}) ---")
        print(f"Source: {chunk['filename']}")
        print(f"Text: {chunk['text'][:150]}...")