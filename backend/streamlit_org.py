# streamlit_org.py
# streamlit run streamlit_org.py --server.port 8004

import streamlit as st
import requests

st.set_page_config(
    page_title="Organization Wellbeing Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Organization Wellbeing Dashboard")

ORG_ID = "ORG_001"
API_BASE = "http://127.0.0.1:8000"

HEADERS = {
    "X-Org-Id": ORG_ID
}

st.subheader("Overall Wellbeing")

try:
    response = requests.get(
        f"{API_BASE}/org/summary",
        headers=HEADERS,
        timeout=10
    )
    response.raise_for_status()
    summary = response.json()

    col1, col2, col3 = st.columns(3)

    col1.metric("Average Wellbeing Score", summary["avg_score"])
    col2.metric("Overall Status", summary["overall_status"].upper())
    col3.metric("Employees Tracked", summary["employee_count"])

except Exception as e:
    st.error("Unable to load organization summary")

st.divider()
st.info("👥 Employee-level insights will appear here once aggregation is finalized.")
