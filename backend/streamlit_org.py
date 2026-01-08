# streamlit_org.py

import streamlit as st
import requests

# ---------------------------------
# Page config
# ---------------------------------
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

# ---------------------------------
# Overall Wellbeing Summary
# ---------------------------------
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

    col1.metric(
        label="Average Wellbeing Score",
        value=summary.get("avg_score", 0)
    )

    col2.metric(
        label="Overall Status",
        value=summary.get("overall_status", "N/A").upper()
    )

    col3.metric(
        label="Employees Tracked",
        value=summary.get("employee_count", 0)
    )

except Exception as e:
    st.error("Unable to load organization summary")

# ---------------------------------
# Placeholder for future expansion
# ---------------------------------
st.divider()
st.info(
    "👥 Employee-level insights will appear here once aggregation is finalized."
)
