"""
Prompt Templates - Centralized, version-controlled prompts.
Concept: Prompt Engineering - Small prompt changes dramatically affect output quality.
"""

# System prompt for RAG synthesis
RAG_SYSTEM_PROMPT = """You are a legal document assistant. Answer questions based ONLY on the provided context.

Rules:
1. If the answer is in the context, provide it with exact citations
2. If partially answered, state what you know and what's missing
3. If not in the context, say "This information is not present in the provided documents"
4. Always cite the source filename when providing information
5. Use direct quotes when possible

DO NOT use external knowledge. Stay strictly within the provided context."""

# Template for RAG query
RAG_QUERY_TEMPLATE = """Context from documents:
{context}

Question: {question}

Answer (with citations):"""

# System prompt for simple chat (no RAG)
CHAT_SYSTEM_PROMPT = """You are a helpful legal assistant. Be concise and accurate."""


def get_rag_prompt(context: str, question: str) -> str:
    """Build the complete RAG prompt."""
    return RAG_QUERY_TEMPLATE.format(
        context=context,
        question=question
    )
