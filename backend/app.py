from backend.services.predictor import Predictor
from backend.services.wellbeing_trend_engine import BurnoutEngine
from backend.services.llm_recommender import LLMRecommender
from backend.services.db import JsonStore

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import date
from pathlib import Path
from typing import Union, List, Optional


# --------------------------------
# App initialization
# --------------------------------

app = FastAPI(
    title="Daily Emotion Tracker API",
    description="Backend service for emotion analysis and wellbeing suggestions",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------
# Services
# --------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent  # SENTIMENT_ANALYSIS/

MODEL_PATH = BASE_DIR / "models" / "best_emotion_model.keras"
TOKENIZER_PATH = BASE_DIR / "models" / "tokenizer.pkl"
CONFIG_PATH = BASE_DIR / "models" / "config.json"

predictor = Predictor(
    model_path=MODEL_PATH,
    tokenizer_path=TOKENIZER_PATH,
    config_path=CONFIG_PATH
)

store = JsonStore()
wellbeing_trend_engine = BurnoutEngine()
llm = LLMRecommender()

# --------------------------------
# Schemas
# --------------------------------

class AnalyzeRequest(BaseModel):
    user_id: Union[str, int]
    text: str


class Suggestion(BaseModel):
    type: str
    title: str
    description: str


class UserResponse(BaseModel):
    assistant_message: str
    suggestions: List[Suggestion]


class OrgResponse(BaseModel):
    user_id: Union[str, int]
    wellbeing_status: str
    burnout_score: float
    dominant_signals: List[str]


# --------------------------------
# API Endpoints
# --------------------------------

@app.get("/")
def health_check():
    return {"status": "API running successfully"}


@app.post("/analyze-day", response_model=Union[UserResponse, OrgResponse])
def analyze_day(
    req: AnalyzeRequest,
    x_source: Optional[str] = Header(default="user")  # "user" or "org"
):
    # -------------------------
    # 1. Validation
    # -------------------------
    if not req.text or len(req.text.strip()) < 3:
        raise HTTPException(
            status_code=400,
            detail="Text must contain at least 3 characters"
        )

    # -------------------------
    # 2. Predict emotions
    # -------------------------
    emotions, dominant_emotion = predictor.predict(req.text)

    # -------------------------
    # 3. Prepare today's record
    # -------------------------
    today_record = {
        "user_id": req.user_id,
        "date": date.today().isoformat(),
        "text": req.text,
        "emotions": emotions
    }

    # -------------------------
    # 4. Load history (last 2 days)
    # -------------------------
    history = store.get_last_n(req.user_id, n=2)
    full_window = history + [today_record]

    # -------------------------
    # 5. Burnout analysis (3-day window)
    # -------------------------
    burnout_score, wellbeing_status, dominant_signals = (
        wellbeing_trend_engine.analyze_trend(full_window)
    )

    allowed_categories = (
        wellbeing_trend_engine.allowed_suggestion_categories(wellbeing_status)
    )

    today_record["burnout_score"] = burnout_score
    today_record["wellbeing_status"] = wellbeing_status

    # Determine emotional pattern
    if dominant_emotion in ["joy", "love"] and emotions.get(dominant_emotion, 0) > 0.4:
        pattern = "positive"
    elif wellbeing_status == "low":
        pattern = "stable"
    else:
        pattern = "draining"

    



    # -------------------------
    # 6. Persist record
    # -------------------------
    store.add_record(today_record)

    # -------------------------
    # 7. Base (rule-based) message
    # -------------------------
    base_message = wellbeing_trend_engine.base_recommendation(wellbeing_status)

    # -------------------------
    # 8. LLM suggestions (only if allowed)
    # -------------------------
    if allowed_categories:
        llm_context = {
            "wellbeing_status": wellbeing_status,
            "signals": dominant_signals,
            "allowed_categories": allowed_categories
        }
        suggestions = llm.generate_suggestions(llm_context)
    else:
        suggestions = []

    # Trend awareness
    if len(full_window) >= 2:
        prev = full_window[-2]["emotions"]
        curr = emotions

        if curr.get("joy", 0) > prev.get("joy", 0):
            trend = "improving"
        elif curr.get("sadness", 0) > prev.get("sadness", 0):
            trend = "declining"
        else:
            trend = "stable"
    else:
        trend = "unknown"

    chat_context = {
    "state": wellbeing_status,              # internal only
    "dominant_emotion": dominant_emotion,
    "pattern": pattern
    }

    
    chat_context["trend"] = trend



    assistant_message = llm.generate_conversational_message(chat_context)

    # -------------------------
    # 9. RESPONSE SPLIT
    # -------------------------
    if x_source.lower() == "user":
        return UserResponse(
        assistant_message=assistant_message,
        suggestions=suggestions
        )


    else:  # org / analytics
        return OrgResponse(
            user_id=req.user_id,
            wellbeing_status=wellbeing_status,
            burnout_score=burnout_score,
            dominant_signals=dominant_signals
        )