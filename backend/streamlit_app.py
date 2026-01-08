# streamlit_app.py
import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Burnout & Emotion Tracker", layout="centered")

st.title("🧠 Daily Emotion & Burnout Tracker")
st.markdown("_Not a medical tool — for reflection and gentle suggestions._")

API_URL = "http://127.0.0.1:8000/analyze-day"


# -------------------------
# User input
# -------------------------

user_id = st.text_input("User ID", value="user1")
text = st.text_area(
    "How was your day today?",
    height=200,
    placeholder="Share anything you'd like about your day..."
)

# -------------------------
# Submit
# -------------------------

if st.button("Analyze"):
    if not text or len(text.strip()) < 5:
        st.error("Please enter a short description of your day.")
    else:
        payload = {
            "user_id": user_id,
            "text": text
        }

        try:
            with st.spinner("Analyzing your day..."):
                resp = requests.post(API_URL, json=payload, timeout=30)
                resp.raise_for_status()
                data = resp.json()

            # -------------------------
            # Emotions
            # -------------------------

            emotions = data.get("emotions", {})
            if emotions:
                df = (
                    pd.DataFrame.from_dict(
                        emotions, orient="index", columns=["score"]
                    )
                    .sort_values("score", ascending=False)
                )

                st.subheader("Emotion probabilities")
                st.bar_chart(df)

            # -------------------------
            # Burnout summary
            # -------------------------

            st.subheader("Burnout summary")

            burnout_level = data.get("burnout_level", "unknown")
            burnout_score = data.get("burnout_score", 0.0)

            st.write(f"**Burnout level:** `{burnout_level}`")
            st.write(f"**Burnout score:** `{burnout_score:.2f}`")

            dominant_emotion = data.get("dominant_emotion")
            if dominant_emotion:
                st.write(f"**Dominant emotion:** `{dominant_emotion}`")

            dominant_signals = data.get("dominant_signals", [])
            if dominant_signals:
                st.write(
                    "**Dominant signals:** "
                    + ", ".join(f"`{s}`" for s in dominant_signals)
                )

            # -------------------------
            # Suggestions
            # -------------------------

            st.subheader("Suggestions")

            suggestions = data.get("suggestions", [])

            if not suggestions:
                st.info(
                    "No specific suggestions for now. "
                    "Your patterns look stable — keep maintaining balance."
                )
            else:
                for s in suggestions:
                    icon = {
                        "relaxation": "🧘",
                        "activity": "🚶",
                        "trend": "🌍"
                    }.get(s.get("type"), "💡")

                    st.markdown(
                        f"### {icon} {s.get('title', '')}"
                    )
                    st.write(s.get("description", ""))

        except requests.exceptions.RequestException as e:
            st.error(f"Failed to call API: {e}")
