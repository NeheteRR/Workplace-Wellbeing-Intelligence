import streamlit as st
import requests
from datetime import date

# ---------------------------------
# Page config
# ---------------------------------
st.set_page_config(
    page_title="Daily Wellbeing Check-in",
    page_icon="💬",
    layout="centered"
)

st.title("💬 Daily Wellbeing Check-in")
st.caption("A private space to reflect on your day")

API_URL = "http://127.0.0.1:8000/analyze-day"

# ---------------------------------
# Session state
# ---------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_date" not in st.session_state:
    st.session_state.last_date = None

# ---------------------------------
# Daily streak indicator
# ---------------------------------
today = date.today().isoformat()

if st.session_state.last_date == today:
    st.success("✅ You’ve already checked in today — great consistency!")
else:
    st.info("🕊️ Take a moment to reflect on how your day felt.")

# ---------------------------------
# Display chat history
# ---------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------------
# Chat input
# ---------------------------------
user_input = st.chat_input("How was your workday today?")

if user_input:
    # Show user message
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    # ✅ FIXED payload (matches backend schema)
    payload = {
        "org_id": "ORG_001",          # same as Swagger
        "employee_id": "EMP_001",     # same as Swagger
        "text": user_input
    }

    try:
        with st.spinner("Thinking..."):
            response = requests.post(
                API_URL,
                json=payload,
                headers={
                    "X-Source": "user"   # critical
                },
                timeout=30
            )

            response.raise_for_status()
            data = response.json()

        assistant_message = data.get("assistant_message", "")
        suggestions = data.get("suggestions", [])

        # Show assistant response
        with st.chat_message("assistant"):
            st.markdown(assistant_message)

            if suggestions:
                st.markdown("### 🌱 You could gently try:")
                for s in suggestions:
                    st.markdown(
                        f"- **{s['title']}** — {s['description']}"
                    )

        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_message}
        )

        st.session_state.last_date = today

    except Exception as e:
        st.error("Something went wrong. Please try again later.")
