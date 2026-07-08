"""
LLM Monitoring Dashboard - Reads metrics from API endpoint.
Run: streamlit run llmops/dashboard/app.py
"""

import streamlit as st
import pandas as pd
import requests
from datetime import datetime
import time

st.set_page_config(page_title="LLM Operations Dashboard", layout="wide")

st.title("📊 LLM Operations Dashboard")

# Read from API instead of local monitor
API_URL = "http://localhost:8001"

# Auto-refresh
if st.button("🔄 Refresh Now"):
    st.rerun()

try:
    response = requests.get(f"{API_URL}/metrics", timeout=2)
    
    if response.status_code == 200:
        data = response.json()
        stats = data.get("stats", {})
        alerts = data.get("alerts", [])
        
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if "message" in stats:
            st.info(stats["message"])
            st.info("📝 Make some API requests first:")
            st.code("curl -X POST http://localhost:8001/query -d '{\"question\":\"test\"}'")
        else:
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Requests", stats.get("total_requests", 0))
            with col2:
                st.metric("Success Rate", f"{stats.get('success_rate', 0)}%")
            with col3:
                st.metric("Avg Latency", f"{stats.get('avg_latency', 0)}s")
            with col4:
                st.metric("Tokens Used", f"{stats.get('total_tokens', 0):,}")
            
            # Endpoint breakdown
            if stats.get("endpoints"):
                st.subheader("Endpoint Performance")
                endpoint_df = pd.DataFrame(stats["endpoints"]).T
                st.dataframe(endpoint_df)
            
            # Alerts
            if alerts:
                st.subheader("🚨 Alerts")
                for alert in alerts:
                    st.warning(alert)
            else:
                st.success("✅ All systems normal")
    else:
        st.error(f"API returned {response.status_code}. Is the server running?")
        
except requests.exceptions.ConnectionError:
    st.error("❌ Cannot connect to API server.")
    st.info("Start the server: `python serving/api.py`")
    st.info("Then add the /metrics endpoint to api.py")

# Auto-refresh
time.sleep(5)
st.rerun()