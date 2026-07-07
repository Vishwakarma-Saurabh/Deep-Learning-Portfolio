"""
Search Tool - Agent can search through ingested documents.
Wraps the existing RAG retriever from Milestone 1.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rag.retriever import retrieve_chunks, format_context


def search_documents(query: str) -> dict:

    print(f"🔍 Agent searching for: {query}")
    
    try:
        chunks = retrieve_chunks(query, top_k=5)
        
        if not chunks:
            return {
                "success": False,
                "message": "No relevant documents found. Have documents been ingested?",
                "results": []
            }
        
        # Format results
        results = []
        for chunk in chunks:
            results.append({
                "source": chunk["filename"],
                "relevance_score": f"{chunk['score']:.2f}",
                "content_preview": chunk["text"][:200] + "..."
            })
        
        return {
            "success": True,
            "message": f"Found {len(chunks)} relevant chunks",
            "results": results,
            "context": format_context(chunks)  # Full text for the agent
        }
    
    except Exception as e:
        return {
            "success": False,
            "message": f"Search failed: {str(e)}",
            "results": []
        }


# Tool description for the LLM
TOOL_DESCRIPTION = """
Tool: search_documents
Description: Search through all ingested legal documents for specific information.
Input: query (string) - What to search for
Output: Matching document chunks with relevance scores
Use when: User asks about document contents, needs to find specific clauses, or mentions searching contracts
"""


# Test independently
if __name__ == "__main__":
    result = search_documents("liability cap")
    print(f"Success: {result['success']}")
    print(f"Message: {result['message']}")
    if result['results']:
        print(f"First result: {result['results'][0]['content_preview']}")