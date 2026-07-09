"""
Audit Page - Compliance checking interface.
"""

import streamlit as st
import tempfile
import os
from components.file_uploader import render_file_uploader
from components.violation_cards import render_audit_summary, render_all_violations, render_email_report_button
from utils.api_client import audit_document
from utils.session import set_audit_results, get_audit_results, add_document


def render():
    """Render the audit page."""
    
    st.title("🔍 Compliance Audit")
    st.caption("Check documents for GDPR and SOX violations")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📁 Document Selection")
        
        # Upload new document
        uploaded_file = st.file_uploader(
            "Upload document for audit",
            type=["pdf", "docx", "txt"],
            key="audit_uploader"
        )
        
        if uploaded_file:
            st.info(f"Selected: {uploaded_file.name} ({uploaded_file.size / 1024:.0f} KB)")
            
            if st.button("🔍 Run Compliance Audit", type="primary", use_container_width=True):
                with st.status("Running audit...", expanded=True) as status:
                    st.write("📖 Parsing document...")
                    file_bytes = uploaded_file.getvalue()
                    
                    st.write("🔍 Checking compliance...")
                    with st.spinner("This may take 1-2 minutes for large documents..."):
                        result = audit_document(file_bytes, uploaded_file.name)
                    
                    if "detail" in result:
                        status.update(label="❌ Audit failed", state="error")
                        st.error(result["detail"])
                        return
                    
                    st.write("📊 Generating report...")
                    set_audit_results(result)
                    add_document(uploaded_file.name, result.get("total_clauses", 0))
                    
                    status.update(label="✅ Audit complete!", state="complete")
                    st.rerun()        
        # Audit options
        st.divider()
        st.subheader("⚙️ Options")
        st.checkbox("Include SAFE clauses in report", value=False, key="show_safe")
        st.checkbox("Show raw model output", value=False, key="show_raw")
        
        # Export options
        if get_audit_results():
            st.divider()
            st.subheader("📧 Export Report")
            render_email_report_button(get_audit_results())
    
    with col2:
        audit_results = get_audit_results()
        
        if audit_results:
            # Summary
            render_audit_summary(audit_results)
            
            st.divider()
            
            # Violation details
            violations = audit_results.get("violations", [])
            render_all_violations(violations)            
            # Raw output
            if st.session_state.get("show_raw"):
                st.divider()
                st.subheader("Raw Model Output")
                st.json(audit_results)
        else:
            # Empty state
            st.info("👈 Upload a document and click 'Run Compliance Audit' to get started")
            
            with st.expander("📋 What this does"):
                st.markdown("""
                **Compliance Audit** analyzes each clause in your document for:
                
                - 🔒 **GDPR Violations**: Data protection and privacy issues
                - 📊 **SOX Violations**: Financial compliance problems
                - ⚠️ **NEEDS_REVIEW**: Ambiguous or potentially problematic clauses
                - ✅ **SAFE**: Properly compliant clauses
                
                Each violation includes:
                - Severity level (HIGH/MEDIUM/LOW)
                - Explanation of the issue
                - Suggested fix to become compliant
                """)


if __name__ == "__main__":
    render()