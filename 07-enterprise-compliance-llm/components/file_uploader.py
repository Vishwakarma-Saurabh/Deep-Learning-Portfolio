"""
File upload component with progress indication and document management.
"""

import streamlit as st
import time
from utils.api_client import ingest_document
from utils.session import add_document, get_uploaded_documents, set_current_document


def render_file_uploader():
    """Render document upload section with drag-and-drop."""
    
    st.subheader("📁 Document Upload")
    
    # Show existing documents
    documents = get_uploaded_documents()
    if documents:
        st.caption(f"{len(documents)} document(s) uploaded")
        
        # Document selector
        doc_names = ["Select a document..."] + [d["filename"] for d in documents]
        selected = st.selectbox(
            "Active document",
            doc_names,
            help="Select which document to query against"
        )
        
        if selected != "Select a document...":
            set_current_document(selected)
            doc_info = next(d for d in documents if d["filename"] == selected)
            st.success(f"📄 Active: {selected} ({doc_info['chunks']} clauses)")
    
    st.divider()
    
    # Upload new document
    uploaded_file = st.file_uploader(
        "Upload a new document",
        type=["pdf", "docx", "txt"],
        help="Supports PDF, DOCX, and TXT files. Max 200MB.",
        key="file_uploader_main"
    )
    
    if uploaded_file is not None:
        # Check if already uploaded
        existing = [d for d in documents if d["filename"] == uploaded_file.name]
        
        if existing:
            st.info(f"📄 {uploaded_file.name} is already uploaded ({existing[0]['chunks']} clauses)")
            
            if st.button("🔄 Re-process", key="reprocess"):
                _process_upload(uploaded_file)
        else:
            st.info(f"📄 New file: {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")
            
            if st.button("📤 Process Document", type="primary", key="process_new"):
                _process_upload(uploaded_file)


def _process_upload(uploaded_file):
    """Process an uploaded file with progress updates."""
    
    with st.status(f"Processing {uploaded_file.name}...", expanded=True) as status:
        
        # Step 1: Read file
        st.write("📖 Reading document...")
        file_bytes = uploaded_file.getvalue()
        time.sleep(0.3)
        
        # Step 2: Upload to API
        st.write("📤 Sending to server...")
        result = ingest_document(file_bytes, uploaded_file.name)
        
        if "error" in result:
            status.update(label=f"❌ Failed: {result['error']}", state="error")
            st.error(result["error"])
            return
        
        chunks = result.get("chunks_created", 0)
        st.write(f"✂️ Extracted {chunks} clauses")
        time.sleep(0.2)
        
        # Step 3: Indexing
        st.write("🔍 Indexing for search...")
        time.sleep(0.3)
        
        # Complete
        status.update(
            label=f"✅ {uploaded_file.name} ready! ({chunks} clauses)",
            state="complete"
        )
        
        add_document(uploaded_file.name, chunks)
        st.balloons()
        time.sleep(1)
        st.rerun()