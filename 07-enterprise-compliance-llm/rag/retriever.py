"""
Retriever - Searches vector database for relevant chunks.
Concept: RAG retrieval - finding the right context is 80% of RAG success.
"""

import os
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()

# Reuse the same embedding model
EMBEDDING_MODEL = SentenceTransformer(
    os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
)

QDRANT_CLIENT = QdrantClient(
    host=os.getenv("QDRANT_HOST", "localhost"),
    port=int(os.getenv("QDRANT_PORT", 6333))
)

COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "compliance_docs")


def retrieve_chunks(query: str, top_k: int = 3) -> List[Dict]:

    # Embed the query
    query_vector = EMBEDDING_MODEL.encode(query).tolist()
    
    # Search Qdrant
    search_results = QDRANT_CLIENT.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k
    )
    
    # Format results
    chunks = []
    for result in search_results:
        chunks.append({
            "text": result.payload["text"],
            "filename": result.payload["filename"],
            "chunk_index": result.payload["chunk_index"],
            "score": result.score,  # Cosine similarity (higher = more relevant)
            "char_count": result.payload["char_count"]
        })
    
    return chunks


def format_context(chunks: List[Dict]) -> str:

    context_parts = []
    total_chars = 0
    max_chars = 3000  # Leave room for prompt + answer
    
    for i, chunk in enumerate(chunks):
        source = f"[Source: {chunk['filename']}, Chunk {chunk['chunk_index']}]"
        part = f"{source}\n{chunk['text']}\n"
        
        if total_chars + len(part) > max_chars:
            break
            
        context_parts.append(part)
        total_chars += len(part)
    
    return "\n".join(context_parts)
