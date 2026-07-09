"""
Violation card display component.
Shows compliance violations with severity colors.
"""

import streamlit as st
from typing import Dict, List


def render_violation_card(violation: Dict):
    """Render a single violation as a colored card."""
    
    severity = violation.get("severity", "LOW")
    violation_type = violation.get("violation", "Unknown")
    
    # Color mapping
    colors = {
        "HIGH": "#EF4444",
        "MEDIUM": "#F59E0B",
        "LOW": "#10B981"
    }
    
    emoji_map = {
        "GDPR_Violation": "🔒",
        "SOX_Violation": "📊",
        "SAFE": "✅",
        "NEEDS_REVIEW": "⚠️"
    }
    
    color = colors.get(severity, "#6B7280")
    emoji = emoji_map.get(violation_type, "❓")
    
    with st.container():
        st.markdown(f"""
        <div style="
            border-left: 4px solid {color};
            padding: 15px;
            margin: 10px 0;
            background-color: #1E293B;
            border-radius: 8px;
        ">
            <h4 style="margin: 0; color: {color};">{emoji} {violation_type}</h4>
            <p style="color: #94A3B8; margin: 5px 0;">
                <strong>Severity:</strong> 
                <span style="color: {color};">{severity}</span>
            </p>
            <p style="color: #CBD5E1;">{violation.get('explanation', 'No explanation')}</p>
            <p style="color: #10B981;"><strong>🔧 Fix:</strong> {violation.get('fix', 'No fix suggested')}</p>
        </div>
        """, unsafe_allow_html=True)


def render_audit_summary(audit_results: Dict):
    """Render audit summary with statistics."""
    
    if not audit_results:
        return
    
    summary = audit_results.get("audit_summary", audit_results)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Clauses", summary.get("total_clauses", 0))
    with col2:
        st.metric("Violations", summary.get("violations_found", 0), delta=None)
    with col3:
        st.metric("Safe Clauses", summary.get("safe_clauses", 0))
    with col4:
        risk = audit_results.get("risk_breakdown", {})
        st.metric("High Risk", risk.get("high", 0))


def render_all_violations(violations: List[Dict]):
    """Render all violations from audit results."""
    if not violations:
        st.success("✅ No violations found! All clauses are compliant.")
        return
    
    # Group by severity
    high = [v for v in violations if v.get("severity") == "HIGH"]
    medium = [v for v in violations if v.get("severity") == "MEDIUM"]
    low = [v for v in violations if v.get("severity") == "LOW"]
    
    if high:
        st.subheader(f"🔴 HIGH Risk ({len(high)})")
        for v in high:
            render_violation_card(v)
    
    if medium:
        st.subheader(f"🟡 MEDIUM Risk ({len(medium)})")
        for v in medium:
            render_violation_card(v)
    
    if low:
        st.subheader(f"🟢 LOW Risk ({len(low)})")
        for v in low:
            render_violation_card(v)

def render_email_report_button(audit_results: Dict):
    """Render a button to email the audit report."""
    if not audit_results or not audit_results.get("violations"):
        return
    
    st.divider()
    st.subheader("📧 Share Report")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        email = st.text_input("Email address", placeholder="legal@company.com", key="report_email")
    with col2:
        if st.button("📤 Send Report", type="primary", use_container_width=True):
            if email:
                with st.spinner("Sending report..."):
                    from utils.api_client import run_agent
                    result = run_agent(f"Email the audit report to {email} with subject 'Compliance Audit Report'")
                    if result.get("success"):
                        st.success(f"✅ Report sent to {email}")
                    else:
                        st.error("Failed to send report")
            else:
                st.warning("Please enter an email address")