"""
Chat message display component.
Renders user and AI messages with styling and metadata.
"""

import streamlit as st
from typing import Dict, Optional


def render_user_message(content: str):
    """Render a user message."""
    with st.chat_message("user", avatar="👤"):
        st.markdown(content)


def render_assistant_message(content: str, metadata: Optional[Dict] = None):
    """Render an AI assistant message with optional metadata."""
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(content)
        
        if metadata:
            # Show sources
            if metadata.get("sources"):
                with st.expander("📌 Sources", expanded=False):
                    for source in metadata["sources"]:
                        st.caption(f"• {source}")
            
            # Show cache indicator
            if metadata.get("cache_hit"):
                st.caption("⚡ Served from cache (instant, 0 tokens)")
            
            # Show provider info
            if metadata.get("provider"):
                st.caption(f"🔧 Provider: {metadata['provider']}")
            
            # Show token usage
            if metadata.get("tokens"):
                st.caption(f"🔢 {metadata['tokens']} tokens used")


def render_chat_history(messages: list):
    """Render entire chat history from session state."""
    for msg in messages:
        if msg["role"] == "user":
            render_user_message(msg["content"])
        else:
            render_assistant_message(msg["content"], msg.get("metadata"))


def render_thinking_indicator():
    """Show a thinking animation."""
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            placeholder = st.empty()
        return placeholder


def render_error_message(error: str):
    """Render an error message in chat."""
    with st.chat_message("assistant", avatar="❌"):
        st.error(f"Error: {error}")