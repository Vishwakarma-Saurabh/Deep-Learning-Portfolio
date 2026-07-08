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

app = FastAPI(title="Enterprise Document Intelligence")


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
        latency = time.time() - start
        monitor.record_request("/ingest", latency, 0, True)
        return {"message": f"Ingested {file.filename}", "chunks_created": len(chunks), "storage": result}
    finally:
        os.unlink(tmp_path)


@app.post("/query")
async def query_document(request: QueryRequest):
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
        for chunk in chunks:
            result = check_compliance(chunk["text"])
            if result["violation"] != "SAFE":
                violations.append(result)
        latency = time.time() - start
        monitor.record_request("/audit", latency, 0, True)
        return {"filename": file.filename, "total_clauses": len(chunks), "violations_found": len(violations), "violations": violations}
    except Exception as e:
        latency = time.time() - start
        monitor.record_request("/audit", latency, 0, False)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)


@app.post("/agent")
async def agent_endpoint(req: AgentRequest):
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
    stats = monitor.get_stats()
    alerts = monitor.should_alert()
    return {"stats": stats, "alerts": alerts, "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)