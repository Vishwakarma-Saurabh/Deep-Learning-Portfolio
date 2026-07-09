"""
Compliance AI Assistant - Main Application.
Multi-page interface with chat, audit, agent, and dashboard.
"""

import streamlit as st
from pathlib import Path

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Compliance AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/your-repo",
        "Report a bug": "https://github.com/your-repo/issues",
        "About": "Enterprise Document Intelligence & Compliance Assistant"
    }
)

# Load custom CSS
css_path = Path(__file__).parent / "assets" / "style.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Import utilities
from utils.session import init_session, clear_chat, clear_all
from utils.api_client import check_health

# Initialize session state
init_session()

if "current_page" not in st.session_state:
    st.session_state.current_page = "💬 Chat"

# ============================================
# SIDEBAR
# ============================================

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h1 style="margin: 0;">🤖</h1>
        <h2 style="margin: 5px 0; color: #3B82F6;">Compliance AI</h2>
        <p style="color: #94A3B8; font-size: 0.9rem;">Enterprise Document Intelligence</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # API Status
    api_online = check_health()
    if api_online:
        st.success("🟢 API Connected")
        st.session_state.api_connected = True
    else:
        st.error("🔴 API Offline")
        st.info("Run: `python serving/api.py`")
        st.session_state.api_connected = False
    
    st.divider()
    
    # Navigation
    st.subheader("📱 Navigation")
    
    page = st.radio(
        "Select page",
        ["💬 Chat", "🔍 Audit", "🤖 Agent", "📊 Dashboard"],
        label_visibility="collapsed"
    )
    
    st.divider()
    
    # Session info
    st.subheader("📋 Session Info")
    st.caption(f"🆔 Session: `{st.session_state.session_id}`")
    st.caption(f"📄 Documents: {len(st.session_state.uploaded_documents)}")
    st.caption(f"💬 Messages: {len(st.session_state.messages)}")
    
    st.divider()
    
    # Actions
    st.subheader("⚙️ Actions")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            clear_chat()
            st.rerun()
    with col2:
        if st.button("🔄 Reset All", use_container_width=True):
            clear_all()
            st.rerun()
    
    st.divider()
    
    # Footer
    st.caption("Built with ❤️ using Streamlit")
    st.caption("© 2026 Compliance AI")

# ============================================
# MAIN CONTENT
# ============================================

if not st.session_state.api_connected:
    st.warning("⚠️ API server is not running. Please start it with `python serving/api.py`")
    st.code("python serving/api.py")
    st.stop()

# Route to selected page
if page == "💬 Chat":
    from pages.chat_page import render as chat_render
    chat_render()
elif page == "🔍 Audit":
    from pages.audit_page import render as audit_render
    audit_render()
elif page == "🤖 Agent":
    from pages.agent_page import render as agent_render
    agent_render()
elif page == "📊 Dashboard":
    from pages.dashboard_page import render as dashboard_render
    dashboard_render()