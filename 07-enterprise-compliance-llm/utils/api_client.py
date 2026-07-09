"""
Centralized API client for Streamlit frontend.
All backend calls go through here for consistency.
"""

import requests
import streamlit as st
from typing import Dict, Optional, List

API_URL = "https://deep-learning-portfolio-production.up.railway.app"


@st.cache_data(ttl=30)
def check_health() -> bool:
    """Check if API is running."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def _make_request(method, endpoint, **kwargs):
    """Wrapper with error handling."""
    try:
        response = method(f"{API_URL}{endpoint}", timeout=30, **kwargs)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API server. Please try again later.")
        return {"error": "Connection failed"}
    except requests.exceptions.Timeout:
        st.error("Request timed out. The server may be overloaded.")
        return {"error": "Timeout"}
    except Exception as e:
        st.error(f"API error: {str(e)}")
        return {"error": str(e)}

def ingest_document(file_bytes, filename):
    files = {"file": (filename, file_bytes, "application/pdf")}
    return _make_request(requests.post, "/ingest", files=files)

def ask_question(question, use_cache=True):
    endpoint = "/query/cached" if use_cache else "/query"
    return _make_request(requests.post, endpoint, json={"question": question})


def ask_with_hybrid_search(question: str) -> Dict:
    """Ask question using hybrid search."""
    response = requests.post(f"{API_URL}/query/hybrid", json={"question": question})
    return response.json()


def chat_with_memory(question: str, session_id: str) -> Dict:
    """Ask question with conversation memory."""
    response = requests.post(
        f"{API_URL}/query/chat",
        json={"question": question},
        params={"session_id": session_id}
    )
    return response.json()


def audit_document(file_bytes: bytes, filename: str) -> Dict:
    """Run compliance audit on a document."""
    files = {"file": (filename, file_bytes, "application/pdf")}
    response = requests.post(f"{API_URL}/audit", files=files)
    return response.json()


def audit_by_filename(filename: str) -> Dict:
    """Audit an already ingested document by searching for its clauses."""
    response = requests.post(
        f"{API_URL}/agent",
        json={"request": f"Audit all clauses from {filename} and generate a report"}
    )
    return response.json()


def run_agent(request: str, max_steps: int = 10) -> Dict:
    """Execute an agent workflow."""
    response = requests.post(
        f"{API_URL}/agent",
        json={"request": request, "max_steps": max_steps}
    )
    return response.json()


@st.cache_data(ttl=10)
def get_metrics() -> Dict:
    """Get current system metrics."""
    try:
        response = requests.get(f"{API_URL}/metrics", timeout=2)
        return response.json()
    except:
        return {"stats": {"message": "Cannot connect to API"}}


@st.cache_data(ttl=30)
def get_cache_stats() -> Dict:
    """Get semantic cache statistics."""
    try:
        response = requests.get(f"{API_URL}/cache/stats", timeout=2)
        return response.json()
    except:
        return {"hit_rate": "0%", "estimated_tokens_saved": 0, "cache_size": 0}


@st.cache_data(ttl=30)
def get_circuit_status() -> Dict:
    """Get circuit breaker status."""
    try:
        response = requests.get(f"{API_URL}/circuit/status", timeout=2)
        return response.json()
    except:
        return {"groq": "unknown", "qdrant": "unknown"}


def get_rate_status(user_id: str = "default") -> Dict:
    """Get rate limit status."""
    response = requests.get(f"{API_URL}/rate/status", params={"user_id": user_id})
    return response.json()


def list_ingested_documents() -> List[str]:
    """Get list of ingested documents from session."""
    if "uploaded_documents" in st.session_state:
        return [d["filename"] for d in st.session_state.uploaded_documents]
    return []