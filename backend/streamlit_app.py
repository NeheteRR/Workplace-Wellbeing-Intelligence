# streamlit_app.py
import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Emotion Tracker", layout="centered")

st.title("Daily Emotion Tracker")
st.markdown("_Not a medical tool — for reflection and suggestions._")

API_URL = st.secrets.get("API_URL", "http://localhost:8000/analyze-day")

user_id = st.text_input("User ID", value="user1")
text = st.text_area("How was your day today?", height=200)

if st.button("Analyze"):
    if not text or len(text.strip()) < 3:
        st.error("Please enter a short description of your day.")
    else:
        payload = {"user_id": user_id, "text": text}
        try:
            with st.spinner("Analyzing..."):
                resp = requests.post(API_URL, json=payload, timeout=30)
                resp.raise_for_status()
                data = resp.json()

            emotions = data.get("emotions", {})
            df = pd.DataFrame.from_dict(emotions, orient="index", columns=["score"])
            df = df.sort_values("score", ascending=False)

            st.subheader("Emotion probabilities")
            st.bar_chart(df)

            st.subheader("Risk")
            st.write(f"**Risk level:** {data.get('risk_level')}")
            st.write(f"**Risk score:** {data.get('risk_score'):.2f}")

            st.subheader("Recommendations")
            st.info(data.get("message"))
            st.write("LLM-enhanced suggestion:")
            st.write(data.get("llm_suggestion"))

        except Exception as e:
            st.error("Failed to call API: " + str(e))
