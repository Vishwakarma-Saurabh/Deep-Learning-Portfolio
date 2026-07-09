"""
Session state management for Streamlit.
Handles chat history, uploaded documents, and user preferences.
"""

import streamlit as st
from typing import List, Dict, Optional
import uuid


def init_session():
    """Initialize all session state variables on first run."""
    
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        
        # Chat history
        st.session_state.messages = []
        
        # Uploaded documents tracking
        st.session_state.uploaded_documents = []
        
        # Currently selected document
        st.session_state.current_document = None
        
        # Unique session ID for conversation memory
        st.session_state.session_id = str(uuid.uuid4())[:8]
        
        # Audit results cache
        st.session_state.audit_results = None
        
        # Agent execution trace
        st.session_state.agent_steps = []
        
        # User preferences
        st.session_state.use_cache = True
        st.session_state.use_hybrid_search = False
        
        # API connection status
        st.session_state.api_connected = False


def add_message(role: str, content: str, metadata: Optional[Dict] = None):
    """Add a message to chat history."""
    if metadata is None:
        metadata = {}
    
    st.session_state.messages.append({
        "role": role,
        "content": content,
        "metadata": metadata
    })


def add_document(filename: str, chunks: int):
    """Record an uploaded document."""
    # Check if already exists
    existing = [d for d in st.session_state.uploaded_documents if d["filename"] == filename]
    if existing:
        existing[0]["chunks"] = chunks
    else:
        st.session_state.uploaded_documents.append({
            "filename": filename,
            "chunks": chunks
        })
    
    st.session_state.current_document = filename


def set_current_document(filename: str):
    """Set the currently active document."""
    st.session_state.current_document = filename


def get_current_document() -> Optional[str]:
    """Get the currently active document."""
    return st.session_state.current_document


def clear_chat():
    """Clear chat history but keep documents."""
    st.session_state.messages = []
    st.session_state.agent_steps = []


def clear_all():
    """Reset entire session."""
    st.session_state.messages = []
    st.session_state.uploaded_documents = []
    st.session_state.current_document = None
    st.session_state.audit_results = None
    st.session_state.agent_steps = []


def get_chat_history() -> List[Dict]:
    """Get all chat messages."""
    return st.session_state.messages


def get_session_id() -> str:
    """Get current session ID."""
    return st.session_state.session_id


def get_uploaded_documents() -> List[Dict]:
    """Get list of uploaded documents."""
    return st.session_state.uploaded_documents


def set_audit_results(results: Dict):
    """Store audit results in session."""
    st.session_state.audit_results = results


def get_audit_results() -> Optional[Dict]:
    """Get stored audit results."""
    return st.session_state.audit_results


def add_agent_step(step: Dict):
    """Add an agent execution step."""
    st.session_state.agent_steps.append(step)


def clear_agent_steps():
    """Clear agent execution trace."""
    st.session_state.agent_steps = []


def get_agent_steps() -> List[Dict]:
    """Get agent execution trace."""
    return st.session_state.agent_steps