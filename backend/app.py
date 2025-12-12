# backend/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from services.predictor import Predictor
from services.db import ChromaStore
from services.risk_engine import RiskEngine
from services.llm_recommender import LLMRecommender
from datetime import date
import os

app = FastAPI(title="Emotion Tracker API")

# Config - environment overrides possible
MODEL_PATH = os.environ.get("C:\Project\\backend\\models\\best_emotion_model.keras")
TOKENIZER_PATH = os.environ.get("C:\\Project\\backend\\models\\tokenizer.pkl", None)  # not required for this Keras example

# Initialize services
predictor = Predictor(model_path=MODEL_PATH)
store = ChromaStore(collection_name="emotion_logs")
risk_engine = RiskEngine(store=store)
llm_recommender = LLMRecommender(model_name=os.environ.get("LLM_MODEL", "google/flan-t5-small"))

class AnalyzeRequest(BaseModel):
    user_id: str
    text: str

@app.post("/analyze-day")
async def analyze_day(req: AnalyzeRequest):
    if not req.text or len(req.text.strip()) < 3:
        raise HTTPException(status_code=400, detail="Text must be at least 3 characters long")

    # Predict emotion probabilities
    probs = predictor.predict(req.text)

    # Build record
    today = date.today().isoformat()
    record = {
        "user_id": req.user_id,
        "date": today,
        "text": req.text,
        "emotions": probs
    }

    # Compute risk using last 2 previous days (so with today -> window of up to 3)
    previous = store.get_last_n(req.user_id, n=2)  # returns list newest->oldest
    risk_score, risk_level = risk_engine.compute_with_window(record, previous)
    record["risk_score"] = float(risk_score)
    record["risk_level"] = risk_level

    # Save
    store.add_record(record)

    # Recommendation: combine rule-based and LLM-enhanced suggestion
    base_message = risk_engine.make_recommendation(record)
    # Get enhanced suggestion from LLM (non-blocking style is fine here since small model)
    llm_suggestion = llm_recommender.enhance_recommendation(text=req.text, emotions=probs, base_message=base_message)

    return {
        "emotions": probs,
        "risk_score": float(risk_score),
        "risk_level": risk_level,
        "message": base_message,
        "llm_suggestion": llm_suggestion
    }
