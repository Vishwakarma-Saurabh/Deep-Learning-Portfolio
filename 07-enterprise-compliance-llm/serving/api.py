import sys
import os
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import tempfile

from ingestion.document_parser import parse_document
from ingestion.chunker import chunk_text
from ingestion.embed_and_store import embed_and_store
from rag.retriever import retrieve_chunks, format_context
from rag.synthesizer import synthesize_answer
from compliance import check_compliance
from agents.orchestrator import execute_agent
from llmops.monitor import monitor
from datetime import datetime
from caching.semantic_cache import semantic_cache
from retrieval.hybrid_search import hybrid_search, index_document_for_hybrid
from memory.conversation_store import conversation_store
from resilience.rate_limiter import rate_limiter
from resilience.fallback_chain import fallback_chain
from streaming.stream_handler import StreamHandler, stream_handler
from fastapi.responses import StreamingResponse
from rag.prompts import RAG_SYSTEM_PROMPT, get_rag_prompt
from resilience.circuit_breaker import groq_breaker, qdrant_breaker

app = FastAPI(title="Enterprise Document Intelligence")


def check_rate_limit(user_id: str = "default"):
    """Check rate limit before processing request."""
    if not rate_limiter.is_allowed(user_id):
        retry_after = rate_limiter.get_retry_after(user_id)
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Try again in {retry_after:.0f} seconds."
        )
    

class QueryRequest(BaseModel):
    question: str
    top_k: int = 3


class AgentRequest(BaseModel):
    request: str
    max_steps: int = 10


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    check_rate_limit("default")
    start = time.time()
    with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        parsed = parse_document(tmp_path)
        if "error" in parsed:
            raise HTTPException(status_code=400, detail=parsed["error"])
        chunks = chunk_text(parsed["text"])
        result = embed_and_store(chunks, parsed["filename"])
        
        index_document_for_hybrid(parsed["filename"], chunks)
        
        latency = time.time() - start
        monitor.record_request("/ingest", latency, 0, True)
        return {"message": f"Ingested {file.filename}", "chunks_created": len(chunks), "storage": result}
    finally:
        os.unlink(tmp_path)

@app.post("/query")
async def query_document(request: QueryRequest):
    check_rate_limit("default")
    start = time.time()
    try:
        chunks = retrieve_chunks(request.question, top_k=request.top_k)
        if not chunks:
            latency = time.time() - start
            monitor.record_request("/query", latency, 0, True)
            return {"answer": "No relevant documents found.", "sources": [], "tokens_used": {}}
        
        context = format_context(chunks)
        result = synthesize_answer(request.question, context)
        latency = time.time() - start
        tokens = result["tokens_used"]["total"]
        monitor.record_request("/query", latency, tokens, True)
        
        sources = list(set(f"{c['filename']} (relevance: {c['score']:.2f})" for c in chunks))
        return {"answer": result["answer"], "sources": sources, "tokens_used": result["tokens_used"]}
    except Exception as e:
        latency = time.time() - start
        monitor.record_request("/query", latency, 0, False)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/audit")
async def audit_document(file: UploadFile = File(...)):
    start = time.time()
    with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        parsed = parse_document(tmp_path)
        chunks = chunk_text(parsed["text"], chunk_size=200, overlap=30)
        
        violations = []
        total = len(chunks)
        
        # Process in batches of 5 to avoid memory issues
        batch_size = 5
        for i in range(0, total, batch_size):
            batch = chunks[i:i+batch_size]
            for chunk in batch:
                result = check_compliance(chunk["text"])
                if result["violation"] != "SAFE":
                    violations.append(result)
        
        latency = time.time() - start
        monitor.record_request("/audit", latency, 0, True)
        
        high = len([v for v in violations if v["severity"] == "HIGH"])
        medium = len([v for v in violations if v["severity"] == "MEDIUM"])
        
        return {
            "filename": file.filename,
            "audit_summary": {
                "total_clauses": total,
                "violations_found": len(violations),
                "safe_clauses": total - len(violations),
                "risk_breakdown": {"high": high, "medium": medium}
            },
            "violations": violations
        }
    except Exception as e:
        latency = time.time() - start
        monitor.record_request("/audit", latency, 0, False)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)

@app.post("/agent")
async def agent_endpoint(req: AgentRequest):
    check_rate_limit("default")
    start = time.time()
    try:
        result = execute_agent(req.request, max_steps=req.max_steps)
        latency = time.time() - start
        monitor.record_request("/agent", latency, 0, result["success"])
        return result
    except Exception as e:
        latency = time.time() - start
        monitor.record_request("/agent", latency, 0, False)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics():
    check_rate_limit("default")
    stats = monitor.get_stats()
    alerts = monitor.should_alert()
    return {"stats": stats, "alerts": alerts, "timestamp": datetime.now().isoformat()}


@app.post("/query/hybrid")
async def hybrid_query(request: QueryRequest):
    check_rate_limit("default")
    """Query using hybrid search (dense + sparse)."""
    start = time.time()
    
    try:
        results = hybrid_search(request.question, top_k=request.top_k)
        
        if not results:
            return {"answer": "No results found.", "results": []}
        
        # Format context from hybrid results
        context = "\n\n".join([
            f"[{r.get('filename', 'doc')}] {r.get('text', '')}"
            for r in results
        ])
        
        user_prompt = get_rag_prompt(context, request.question)
        result = fallback_chain.generate(RAG_SYSTEM_PROMPT, user_prompt, request.question)
        
        monitor.record_request("/query/hybrid", time.time() - start, result.get("tokens", 0), True)
        
        return {
            "answer": result["answer"],
            "provider": result.get("provider", "unknown"),
            "results": [{"text": r.get("text", "")[:100], "score": r.get("final_score", r.get("score", 0)), "type": r.get("source_type", "unknown")} for r in results[:3]]
        }
    
    except Exception as e:
        monitor.record_request("/query/hybrid", time.time() - start, 0, False)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/cached")
async def cached_query(request: QueryRequest):
    check_rate_limit("default")
    """Query with semantic caching."""
    start = time.time()
    
    # Check cache first
    cached = semantic_cache.get(request.question)
    if cached:
        monitor.record_request("/query/cached", time.time() - start, 0, True)
        return {
            "answer": cached["answer"],
            "from_cache": True,
            "similar_to": cached.get("cached_question", ""),
            "cache_hit": True
        }
    
    # Cache miss - do normal query
    chunks = retrieve_chunks(request.question, top_k=request.top_k)
    
    if not chunks:
        return {"answer": "No documents found.", "cache_hit": False}
    
    context = format_context(chunks)
    user_prompt = get_rag_prompt(context, request.question)
    result = fallback_chain.generate(RAG_SYSTEM_PROMPT, user_prompt, request.question)
    
    # Store in cache
    semantic_cache.set(request.question, result["answer"])
    
    monitor.record_request("/query/cached", time.time() - start, result.get("tokens", 0), True)
    
    return {
        "answer": result["answer"],
        "from_cache": False,
        "cache_hit": False,
        "provider": result.get("provider", "unknown")
    }


@app.post("/query/stream")
async def stream_query(request: QueryRequest):
    check_rate_limit("default")
    """Query with streaming response."""
    
    chunks = retrieve_chunks(request.question, top_k=request.top_k)
    context = format_context(chunks) if chunks else ""
    
    user_prompt = get_rag_prompt(context, request.question)
    
    return StreamingResponse(
        stream_handler.stream_response(RAG_SYSTEM_PROMPT, user_prompt),
        media_type="text/event-stream"
    )


@app.post("/query/chat")
async def chat_query(request: QueryRequest, session_id: str = "default"):
    check_rate_limit("default")
    """Query with conversation memory."""
    start = time.time()
    
    # Get conversation context
    conv_context = conversation_store.get_context(session_id)
    
    # Get document context
    chunks = retrieve_chunks(request.question, top_k=request.top_k)
    doc_context = format_context(chunks) if chunks else ""
    
    # Combine contexts
    full_context = f"{conv_context}\n\nDocument context:\n{doc_context}" if conv_context else doc_context
    
    user_prompt = get_rag_prompt(full_context, request.question)
    result = fallback_chain.generate(RAG_SYSTEM_PROMPT, user_prompt, request.question)
    
    # Store in conversation memory
    conversation_store.add_exchange(session_id, request.question, result["answer"])
    
    monitor.record_request("/query/chat", time.time() - start, result.get("tokens", 0), True)
    
    return {
        "answer": result["answer"],
        "session_id": session_id,
        "exchanges": len(conversation_store.conversations.get(session_id, [])),
        "provider": result.get("provider", "unknown")
    }


@app.get("/cache/stats")
async def cache_stats():
    check_rate_limit("default")
    """Get semantic cache statistics."""
    return semantic_cache.get_stats()


@app.get("/circuit/status")
async def circuit_status():
    check_rate_limit("default")
    """Get circuit breaker status."""
    return {
        "groq": groq_breaker.get_state(),
        "qdrant": qdrant_breaker.get_state()
    }


@app.get("/rate/status")
async def rate_status(user_id: str = "default"):
    check_rate_limit("default")
    """Get rate limit status for a user."""
    return {
        "user_id": user_id,
        "remaining": rate_limiter.get_remaining(user_id),
        "retry_after_seconds": rate_limiter.get_retry_after(user_id)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)