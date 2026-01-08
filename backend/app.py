from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from datetime import date
from pathlib import Path
from typing import List, Union, Optional

from backend.services.db import JsonStore
from backend.services.predictor import Predictor
from backend.services.wellbeing_trend_engine import BurnoutEngine
from backend.services.llm_recommender import LLMRecommender

# =====================================================
# App Init
# =====================================================

app = FastAPI(title="Wellbeing API")

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "best_emotion_model.keras"
TOKENIZER_PATH = BASE_DIR / "models" / "tokenizer.pkl"
CONFIG_PATH = BASE_DIR / "models" / "config.json"

store = JsonStore()

predictor = Predictor(
    model_path=str(MODEL_PATH),
    tokenizer_path=str(TOKENIZER_PATH),
    config_path=str(CONFIG_PATH)
)

engine = BurnoutEngine()
llm = LLMRecommender()

# =====================================================
# Schemas
# =====================================================

class AnalyzeRequest(BaseModel):
    org_id: str
    employee_id: str
    text: str


class UserResponse(BaseModel):
    assistant_message: str
    suggestions: List[dict]


class OrgResponse(BaseModel):
    employee_id: str
    wellbeing_status: str
    wellbeing_score: float
    dominant_signals: List[str]


# =====================================================
# Employee API
# =====================================================

@app.post("/analyze-day", response_model=Union[UserResponse, OrgResponse])
def analyze_day(
    req: AnalyzeRequest,
    x_source: Optional[str] = Header(default="user")
):
    if len(req.text.strip()) < 3:
        raise HTTPException(status_code=400, detail="Text too short")

    # 1. Emotion prediction
    emotions, dominant_emotion = predictor.predict(req.text)

    # 2. Load history
    history = store.get_last_n(req.employee_id, 2)

    today = {
        "org_id": req.org_id,
        "employee_id": req.employee_id,
        "date": date.today().isoformat(),
        "emotions": emotions
    }

    full_window = history + [today]

    # 3. Trend analysis
    score, status, signals = engine.analyze_trend(full_window)

    today.update({
        "wellbeing_score": score,
        "wellbeing_status": status,
        "dominant_signals": signals
    })

    store.add_record(today)

    # =====================================================
    # ✅ CORRECT emotional pattern logic (USED later)
    # =====================================================

    sadness = emotions.get("sadness", 0)
    joy = emotions.get("joy", 0)
    fear = emotions.get("fear", 0)
    anger = emotions.get("anger", 0)

    if joy >= 0.45:
        pattern = "positive"
    elif sadness + fear + anger >= 0.6:
        pattern = "heavy"
    else:
        pattern = "neutral"

    # =====================================================
    # USER RESPONSE
    # =====================================================

    if x_source == "user":
        message = llm.generate_conversational_message({
            "state": status,                 # internal
            "dominant_emotion": dominant_emotion,
            "pattern": pattern,              # ✅ FIXED
            "trend": "stable"
        })

        suggestions = llm.generate_suggestions({
            "wellbeing_status": status,
            "signals": signals,
            "allowed_categories": engine.allowed_suggestion_categories(status)
        })

        return UserResponse(
            assistant_message=message,
            suggestions=suggestions
        )

    # =====================================================
    # ORG RESPONSE
    # =====================================================

    return OrgResponse(
        employee_id=req.employee_id,
        wellbeing_status=status,
        wellbeing_score=score,
        dominant_signals=signals
    )


# =====================================================
# ORG DASHBOARD APIs
# =====================================================

@app.get("/org/summary")
def org_summary(x_org_id: str = Header(...)):
    records = store.get_records_by_org(x_org_id)

    if not records:
        return {"avg_score": 0, "overall_status": "no_data", "employee_count": 0}

    scores = [r["wellbeing_score"] for r in records]
    avg = sum(scores) / len(scores)

    status = (
        "healthy" if avg < 0.3 else
        "watch" if avg < 0.55 else
        "elevated"
    )

    return {
        "avg_score": round(avg, 2),
        "overall_status": status,
        "employee_count": len(set(r["employee_id"] for r in records))
    }



