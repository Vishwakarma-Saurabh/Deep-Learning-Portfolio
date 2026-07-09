"""
Dashboard Page - System monitoring and metrics.
"""

import streamlit as st
import time
from utils.api_client import get_metrics, get_cache_stats, get_circuit_status


def render():
    """Render the dashboard page."""
    
    st.title("📊 System Dashboard")
    st.caption("Real-time monitoring, metrics, and system health")
    
    # Auto-refresh
    auto_refresh = st.checkbox("Auto-refresh (10s)", value=True)
    
    if auto_refresh:
        time.sleep(10)
        st.rerun()
    
    # Manual refresh
    if st.button("🔄 Refresh Now"):
        st.rerun()
    
    # ============================================
    # API METRICS
    # ============================================
    
    st.subheader("📡 API Performance")
    
    metrics = get_metrics()
    stats = metrics.get("stats", {})
    
    if "message" in stats:
        st.info(stats["message"])
    else:
        # Key metrics row
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "Total Requests",
                stats.get("total_requests", 0),
                help="Total API requests since server start"
            )
        
        with col2:
            success_rate = stats.get("success_rate", 0)
            st.metric(
                "Success Rate",
                f"{success_rate}%",
                delta="Good" if success_rate > 95 else "Warning",
                delta_color="normal" if success_rate > 95 else "inverse"
            )
        
        with col3:
            st.metric(
                "Avg Latency",
                f"{stats.get('avg_latency', 0)}s",
                help="Average response time"
            )
        
        with col4:
            st.metric(
                "P95 Latency",
                f"{stats.get('latency_p95', 0)}s",
                help="95th percentile - worst 5% of requests"
            )
        
        with col5:
            st.metric(
                "Tokens Used",
                f"{stats.get('total_tokens', 0):,}",
                help="Total LLM tokens consumed"
            )
        
        # Endpoint breakdown
        if stats.get("endpoints"):
            st.divider()
            st.subheader("🔗 Endpoint Performance")
            
            endpoints = stats["endpoints"]
            
            # Create columns for each endpoint
            cols = st.columns(len(endpoints))
            
            for col, (ep, data) in zip(cols, endpoints.items()):
                with col:
                    error_rate = data.get("error_rate", 0)
                    status_color = "🟢" if error_rate == 0 else "🟡" if error_rate < 5 else "🔴"
                    
                    st.markdown(f"""
                    <div style="
                        background-color: #1E293B;
                        padding: 16px;
                        border-radius: 8px;
                        text-align: center;
                    ">
                        <h4 style="margin: 0; color: #94A3B8;">{status_color} {ep}</h4>
                        <p style="font-size: 2rem; margin: 8px 0; color: #3B82F6;">{data['requests']}</p>
                        <p style="color: #94A3B8; font-size: 0.8rem;">
                            {data['avg_latency']}s avg | {error_rate}% errors
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
    
    # ============================================
    # CACHE STATS
    # ============================================
    
    st.divider()
    st.subheader("💾 Cache Performance")
    
    cache = get_cache_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Hit Rate", cache.get("hit_rate", "0%"))
    with col2:
        st.metric("Cache Hits", cache.get("hits", 0))
    with col3:
        st.metric("Cache Misses", cache.get("misses", 0))
    with col4:
        st.metric("Tokens Saved", f"{cache.get('estimated_tokens_saved', 0):,}")
    
    # ============================================
    # SYSTEM HEALTH
    # ============================================
    
    st.divider()
    st.subheader("🏥 System Health")
    
    circuit = get_circuit_status()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        groq_status = circuit.get("groq", "unknown")
        if groq_status == "closed":
            st.success("🔌 Groq API: Connected")
        elif groq_status == "open":
            st.error("🔌 Groq API: Circuit OPEN")
        else:
            st.warning(f"🔌 Groq API: {groq_status}")
    
    with col2:
        qdrant_status = circuit.get("qdrant", "unknown")
        if qdrant_status == "closed":
            st.success("🗄️ Qdrant: Connected")
        elif qdrant_status == "open":
            st.error("🗄️ Qdrant: Circuit OPEN")
        else:
            st.warning(f"🗄️ Qdrant: {qdrant_status}")
    
    with col3:
        st.success("🖥️ Server: Running") if stats else st.error("🖥️ Server: Down")
    
    # ============================================
    # ALERTS
    # ============================================
    
    alerts = metrics.get("alerts", [])
    
    if alerts:
        st.divider()
        st.subheader("🚨 Active Alerts")
        for alert in alerts:
            st.warning(alert)
    else:
        st.divider()
        st.success("✅ No active alerts - All systems normal")
    
    # ============================================
    # UPTIME
    # ============================================
    
    uptime = stats.get("uptime_seconds", 0)
    if uptime > 0:
        hours = int(uptime // 3600)
        minutes = int((uptime % 3600) // 60)
        seconds = int(uptime % 60)
        
        st.divider()
        st.caption(f"⏱️ Server uptime: {hours}h {minutes}m {seconds}s")


if __name__ == "__main__":
    render()