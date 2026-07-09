"""
Embedding & Vector Storage - Converts text to vectors, stores in Qdrant.
Concept: Embeddings capture semantic meaning. Similar texts = close vectors.
"""

import os
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from dotenv import load_dotenv

load_dotenv()

# Load embedding model once (cached)
EMBEDDING_MODEL = SentenceTransformer(
    os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
)

# Qdrant client
QDRANT_CLIENT = QdrantClient(
    host=os.getenv("QDRANT_URL", "QDRANT_URL"),
    port=int(os.getenv("QDRANT_API_KEY", "QDRANT_API_KEY"))
)

COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "compliance_docs")
VECTOR_SIZE = 384  # all-MiniLM-L6-v2 outputs 384-dimensional vectors


def create_collection_if_not_exists():
    """Create Qdrant collection if it doesn't exist."""
    collections = QDRANT_CLIENT.get_collections().collections
    collection_names = [c.name for c in collections]
    
    if COLLECTION_NAME not in collection_names:
        QDRANT_CLIENT.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE  # Cosine similarity
            )
        )
        print(f"Created collection: {COLLECTION_NAME}")


def embed_and_store(chunks: List[Dict], filename: str) -> Dict:
    
    create_collection_if_not_exists()
    
    # Extract texts from chunks
    texts = [chunk["text"] for chunk in chunks]
    
    # Generate embeddings in batch (faster)
    print(f"Embedding {len(texts)} chunks...")
    embeddings = EMBEDDING_MODEL.encode(texts, show_progress_bar=True)
    
    # Prepare points for Qdrant
    points = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        point = PointStruct(
            id=i,  # Simple auto-increment ID
            vector=embedding.tolist(),
            payload={
                "text": chunk["text"],
                "filename": filename,
                "chunk_index": chunk["chunk_index"],
                "char_count": chunk["char_count"],
                "approx_tokens": chunk["approx_tokens"]
            }
        )
        points.append(point)
    
    # Upload to Qdrant
    operation_info = QDRANT_CLIENT.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    
    return {
        "status": "success",
        "chunks_stored": len(points),
        "vector_size": VECTOR_SIZE,
        "collection": COLLECTION_NAME
    }


def get_collection_info() -> Dict:
    """Get information about the current collection."""
    try:
        info = QDRANT_CLIENT.get_collection(COLLECTION_NAME)
        return {
            "name": COLLECTION_NAME,
            "vectors_count": info.vectors_count,
            "status": "active"
        }
    except Exception:
        return {"status": "not_found"}


# Test independently
if __name__ == "__main__":
    # Test with sample chunks
    test_chunks = [
        {
            "text": "The tools you must be fluent in: Python (non-negotiable), PyTorch (for anything model-related), HuggingFace (models, PEFT, datasets), LangChain or LlamaIndex (RAG/agent orchestration), a vector database (start with Chroma locally, learn Pinecone/Qdrant for production), Docker (to containerize your apps), and at least one cloud platform (AWS SageMaker, GCP Vertex, or Azure OpenAI).Want me to go deep on any specific layer — like how RAG actually works, or how to build your first LLM agent from scratch?.",
            "chunk_index": 0,
            "start_char": 0,
            "end_char": 55,
            "char_count": 55,
            "approx_tokens": 14
        }
    ]
    
    result = embed_and_store(test_chunks, "test_contract.pdf")
    print(result)
    print(get_collection_info())