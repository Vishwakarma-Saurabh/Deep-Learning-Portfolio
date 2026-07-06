"""
FastAPI Server - Main application entry point.
Concept: Inference & Deployment - serving ML models via APIs.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
from pydantic import BaseModel
import tempfile
import os

from agents.tools.compliance_tool import check_compliance
from ingestion.document_parser import parse_document
from ingestion.chunker import chunk_text
from ingestion.embed_and_store import embed_and_store
from rag.retriever import retrieve_chunks, format_context
from rag.synthesizer import synthesize_answer

app = FastAPI(title="Enterprise Document RAG System")


class QueryRequest(BaseModel):
    question: str
    top_k: int = 3


class QueryResponse(BaseModel):
    answer: str
    sources: list
    tokens_used: dict


@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """
    Ingest a document: parse → chunk → embed → store
    """
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    
    try:
        # Parse
        parsed = parse_document(tmp_path)
        if "error" in parsed:
            raise HTTPException(status_code=400, detail=parsed["error"])
        
        # Chunk
        chunks = chunk_text(parsed["text"])
        if not chunks:
            raise HTTPException(status_code=400, detail="No text extracted")
        
        # Embed & Store
        result = embed_and_store(chunks, parsed["filename"])
        
        return {
            "message": f"Ingested {file.filename}",
            "chunks_created": len(chunks),
            "storage": result
        }
    
    finally:
        os.unlink(tmp_path)  # Clean up temp file


@app.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    """
    Query the document knowledge base.
    """
    # Retrieve relevant chunks
    chunks = retrieve_chunks(request.question, top_k=request.top_k)
    
    if not chunks:
        return QueryResponse(
            answer="No relevant documents found. Please ingest documents first.",
            sources=[],
            tokens_used={}
        )
    
    # Format context
    context = format_context(chunks)
    
    # Generate answer
    result = synthesize_answer(request.question, context)
    
    # Extract sources
    sources = list(set(
        f"{chunk['filename']} (relevance: {chunk['score']:.2f})"
        for chunk in chunks
    ))
    
    return QueryResponse(
        answer=result["answer"],
        sources=sources,
        tokens_used=result["tokens_used"]
    )


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/audit")
async def audit_document(file: UploadFile = File(...)):
    """
    Audit a document for compliance violations.
    Returns violation report with severity and explanations.
    """
    
    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    
    try:
        # Parse document
        parsed = parse_document(tmp_path)
        if "error" in parsed:
            raise HTTPException(status_code=400, detail=parsed["error"])
        
        # Chunk into clauses
        chunks = chunk_text(parsed["text"], chunk_size=300, overlap=50)
        
        # Check each chunk
        violations = []
        safe_count = 0
        
        for chunk in chunks:
            result = check_compliance(chunk["text"])
            
            if result["violation"] != "SAFE":
                violations.append(result)
            else:
                safe_count += 1
        
        # Generate summary
        high_risk = len([v for v in violations if v["severity"] == "HIGH"])
        medium_risk = len([v for v in violations if v["severity"] == "MEDIUM"])
        
        return {
            "filename": file.filename,
            "total_clauses": len(chunks),
            "violations_found": len(violations),
            "safe_clauses": safe_count,
            "risk_summary": {
                "high": high_risk,
                "medium": medium_risk,
                "low": len(violations) - high_risk - medium_risk
            },
            "violations": violations
        }
    
    finally:
        os.unlink(tmp_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)