# backend/app.py
# uvicorn backend.app:app --reload

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import date
from pathlib import Path
from typing import List, Union, Optional, Dict, Any
import jwt
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
from pydantic import BaseModel, EmailStr

from services.db import JsonStore
from services.predictor import Predictor
from services.wellbeing_trend_engine import BurnoutEngine
from services.llm_recommender import LLMRecommender

# =====================================================
# App Init
# =====================================================

app = FastAPI(title="Wellbeing API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For development, allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "best_emotion_model.keras"
TOKENIZER_PATH = BASE_DIR / "models" / "tokenizer.pkl"
CONFIG_PATH = BASE_DIR / "models" / "config.json"

load_dotenv()
SECRET_KEY = os.getenv("SECRET_KEY", "dummy_secret_for_jwt")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7 # 1 week

store = JsonStore()

predictor = Predictor(
    model_path=str(MODEL_PATH),
    tokenizer_path=str(TOKENIZER_PATH),
    config_path=str(CONFIG_PATH)
)

engine = BurnoutEngine()
llm = LLMRecommender()

# =====================================================
# Mock Database
# =====================================================

MOCK_USERS = [
    {"email": "employee1@company.com", "password": "password123", "role": "employee", "employee_id": "E101", "org_id": "ORG_001", "name": "Ravi", "department": "Engineering"},
    {"email": "employee2@company.com", "password": "password123", "role": "employee", "employee_id": "E102", "org_id": "ORG_001", "name": "Anita", "department": "Sales"},
    {"email": "employee3@company.com", "password": "password123", "role": "employee", "employee_id": "E103", "org_id": "ORG_001", "name": "Suresh", "department": "Marketing"},
    {"email": "employee4@company.com", "password": "password123", "role": "employee", "employee_id": "E104", "org_id": "ORG_001", "name": "Meena", "department": "Engineering"},
    {"email": "employee5@company.com", "password": "password123", "role": "employee", "employee_id": "E105", "org_id": "ORG_001", "name": "Arjun", "department": "Sales"},
    {"email": "hr@company.com", "password": "password123", "role": "hr", "employee_id": "H001", "org_id": "ORG_001", "name": "HR Admin", "department": "Human Resources"}
]

# =====================================================
# Schemas
# =====================================================

class AnalyzeRequest(BaseModel):
    org_id: str
    employee_id: str
    text: str

class LoginRequest(BaseModel):
    email: str # Could use EmailStr but keeping it simple
    password: str

class LoginResponse(BaseModel):
    access_token: str
    role: str
    employee_id: str
    org_id: str
    name: str


class AnalyzeResponse(BaseModel):
    emotions: Dict[str, float]
    wellbeing_score: float
    wellbeing_status: str
    dominant_emotion: str
    suggestions: List[dict]

class UserResponse(BaseModel):
    assistant_message: str
    suggestions: List[dict]
    emotions: Dict[str, float]
    wellbeing_score: float
    wellbeing_status: str
    dominant_emotion: str


class OrgResponse(BaseModel):
    employee_id: str
    wellbeing_status: str
    wellbeing_score: float
    dominant_signals: List[str]

# =====================================================
# Auth API
# =====================================================

def create_access_token(data: dict, expires_delta: timedelta):
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

@app.post("/auth/login", response_model=LoginResponse)
def login(req: LoginRequest):
    user = next((u for u in MOCK_USERS if u["email"] == req.email and u["password"] == req.password), None)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["email"], "role": user["role"], "employee_id": user["employee_id"], "org_id": user["org_id"]},
        expires_delta=access_token_expires
    )
    
    return LoginResponse(
        access_token=access_token,
        role=user["role"],
        employee_id=user["employee_id"],
        org_id=user["org_id"],
        name=user["name"]
    )

# =====================================================
# Employee API
# =====================================================

@app.get("/employee/today-summary")
def get_today_summary(employee_id: str):
    records = store.get_records_by_org("ORG_001") # Using default ORG_001 for now, in a real app read from token
    employee_records = [r for r in records if r.get("employee_id") == employee_id]
    if not employee_records:
        raise HTTPException(status_code=404, detail="No data found for today")
    
    # Sort by date descending and get the latest
    employee_records.sort(key=lambda x: x.get("date", ""), reverse=True)
    latest = employee_records[0]
    
    # Check if it's actually today (for mock data it might not be, so we just return the latest)
    
    return {
        "wellbeing_score": latest.get("wellbeing_score", 0),
        "status": latest.get("wellbeing_status", "unknown"),
        "emotions": latest.get("emotions", {})
    }

@app.post("/analyze-day", response_model=Union[UserResponse, OrgResponse])
def analyze_day(
    req: AnalyzeRequest,
    x_source: Optional[str] = Header(default="user")
):
    if len(req.text.strip()) < 3:
        raise HTTPException(status_code=400, detail="Text too short")

    emotions, dominant_emotion = predictor.predict(req.text)
    history = store.get_last_n(req.employee_id, 2)

    today = {
        "org_id": req.org_id,
        "employee_id": req.employee_id,
        "date": date.today().isoformat(),
        "emotions": emotions
    }

    full_window = history + [today]
    score, status, signals = engine.analyze_trend(full_window)

    today.update({
        "wellbeing_score": score,
        "wellbeing_status": status,
        "dominant_signals": signals
    })

    store.add_record(today)

    # ---- FIXED emotional pattern logic ----
    sadness = emotions.get("sadness", 0)
    fear = emotions.get("fear", 0)
    anger = emotions.get("anger", 0)
    joy = emotions.get("joy", 0)

    if joy >= 0.45:
        pattern = "positive"
    elif sadness + fear + anger >= 0.6:
        pattern = "heavy"
    else:
        pattern = "neutral"

    if x_source == "user":
        message = llm.generate_conversational_message({
            "state": status,              # internal only
            "dominant_emotion": dominant_emotion,
            "pattern": pattern,
            "trend": "stable"
        })

        suggestions = llm.generate_suggestions({
            "wellbeing_status": status,
            "signals": signals,
            "allowed_categories": engine.allowed_suggestion_categories(status)
        })

        return UserResponse(
            assistant_message=message,
            suggestions=suggestions,
            emotions=emotions,
            wellbeing_score=score,
            wellbeing_status=status,
            dominant_emotion=dominant_emotion
        )

    return OrgResponse(
        employee_id=req.employee_id,
        wellbeing_status=status,
        wellbeing_score=score,
        dominant_signals=signals
    )

@app.get("/employee/wellbeing-history")
def get_wellbeing_history(employee_id: str, x_org_id: str = Header(default="ORG_001")):
    records = store.get_records_by_org(x_org_id)
    employee_records = [r for r in records if r.get("employee_id") == employee_id]
    
    # Sort chronologically ascending for the chart
    employee_records.sort(key=lambda x: x.get("date", ""))
    
    history = [
        {"date": r.get("date"), "score": r.get("wellbeing_score")}
        for r in employee_records if "date" in r and "wellbeing_score" in r
    ]
    return history

@app.get("/employee/emotion-history")
def get_emotion_history(employee_id: str, x_org_id: str = Header(default="ORG_001")):
    records = store.get_records_by_org(x_org_id)
    employee_records = [r for r in records if r.get("employee_id") == employee_id]
    
    # Sort descending (newest first) for the table
    employee_records.sort(key=lambda x: x.get("date", ""), reverse=True)
    
    history = []
    for r in employee_records:
        if "date" in r and "emotions" in r:
            entry = {"date": r.get("date")}
            entry.update(r.get("emotions", {}))
            history.append(entry)
            
    return history

# =====================================================
# ORG DASHBOARD APIs
# =====================================================

@app.get("/org/summary")
def org_summary(x_org_id: str = Header(...)):
    records = store.get_records_by_org(x_org_id)

    if not records:
        return {
            "avg_score": 0,
            "overall_status": "no_data",
            "employee_count": 0
        }

    scores = [r["wellbeing_score"] for r in records]
    avg = sum(scores) / len(scores)

    if avg < 0.3:
        status = "healthy" # wait, original logic seems flipped or maybe lower is better here? No, higher should be better? Let's fix this based on original code. If avg < 0.3 healthy? Let's keep original for now to avoid breaking existing frontend if any, but add risk employees. Wait, score_to_status in seed_dummy_data says < 0.3 is low. Let's fix it to match.
        status = "low"
    elif avg < 0.55:
        status = "moderate"
    else:
        status = "high"

    # Get latest log for each employee to determine current risk
    latest_logs = {}
    for r in records:
        emp_id = r.get("employee_id")
        if emp_id not in latest_logs or r.get("date", "") > latest_logs[emp_id].get("date", ""):
            latest_logs[emp_id] = r
            
    risk_employees = len([l for l in latest_logs.values() if l.get("wellbeing_status") == "low"])
    
    # Dynamic checkin rate: (employees who checked in today / total employees)
    total_employees = len([u for u in MOCK_USERS if u["role"] == "employee"])
    today_str = date.today().isoformat()
    checked_in_today = len({r["employee_id"] for r in records if r.get("date") == today_str})
    
    checkin_rate = int((checked_in_today / total_employees) * 100) if total_employees > 0 else 0

    return {
        "avg_score": round(avg, 2),
        "overall_status": status,
        "employee_count": total_employees,
        "risk_employees": risk_employees,
        "checkin_rate": checkin_rate
    }

@app.get("/org/emotion-distribution")
def org_emotion_distribution(x_org_id: str = Header(default="ORG_001")):
    records = store.get_records_by_org(x_org_id)
    if not records:
        return []
        
    totals = {"joy": 0, "sadness": 0, "fear": 0, "anger": 0, "love": 0, "neutral": 0}
    count = 0
    
    for r in records:
        if "emotions" in r:
            count += 1
            for k, v in r["emotions"].items():
                if k in totals:
                    totals[k] += v
                    
    if count == 0:
        return []
        
    return [{"name": k.capitalize(), "value": round(v / count, 2)} for k, v in totals.items()]

@app.get("/org/wellbeing-trend")
def org_wellbeing_trend(x_org_id: str = Header(default="ORG_001")):
    records = store.get_records_by_org(x_org_id)
    daily_scores = {}
    
    for r in records:
        date_str = r.get("date")
        score = r.get("wellbeing_score")
        if date_str and score is not None:
            if date_str not in daily_scores:
                daily_scores[date_str] = []
            daily_scores[date_str].append(score)
            
    trend = []
    for date_str in sorted(daily_scores.keys()):
        avg = sum(daily_scores[date_str]) / len(daily_scores[date_str])
        trend.append({"date": date_str, "score": round(avg, 2)})
        
    return trend

@app.get("/org/department-insights")
def org_department_insights(x_org_id: str = Header(default="ORG_001")):
    records = store.get_records_by_org(x_org_id)
    dept_stats = {}
    
    # Map employees to departments via MOCK_USERS
    emp_to_dept = {u["employee_id"]: u.get("department", "General") for u in MOCK_USERS if u["role"] == "employee"}
    
    for r in records:
        emp_id = r.get("employee_id")
        dept = emp_to_dept.get(emp_id, "General")
        score = r.get("wellbeing_score")
        status = r.get("wellbeing_status")
        
        if score is not None:
            if dept not in dept_stats:
                dept_stats[dept] = {"scores": [], "risk_count": 0, "employees": set()}
            dept_stats[dept]["scores"].append(score)
            dept_stats[dept]["employees"].add(emp_id)
            if status in ["low"]:
                # Only count risk per employee, not per log. Let's simplify and just say if any log is low recently
                pass
                
    # Recalculate risk per employee based on latest log
    latest_logs = {}
    for r in records:
        emp_id = r.get("employee_id")
        if emp_id not in latest_logs or r.get("date", "") > latest_logs[emp_id].get("date", ""):
            latest_logs[emp_id] = r
            
    for emp_id, log in latest_logs.items():
        dept = emp_to_dept.get(emp_id, "General")
        if log.get("wellbeing_status") in ["low"]:
            if dept in dept_stats:
                dept_stats[dept]["risk_count"] += 1
                
    res = []
    for dept, stats in dept_stats.items():
        avg = sum(stats["scores"]) / len(stats["scores"]) if stats["scores"] else 0
        res.append({
            "department": dept,
            "avg_wellbeing": round(avg, 2),
            "employees_at_risk": stats["risk_count"]
        })
        
    return res

@app.get("/org/risk-employees")
def org_risk_employees(x_org_id: str = Header(default="ORG_001")):
    records = store.get_records_by_org(x_org_id)
    
    # Group logs by employee
    emp_logs = {}
    for r in records:
        emp_id = r.get("employee_id")
        if emp_id not in emp_logs:
            emp_logs[emp_id] = []
        emp_logs[emp_id].append(r)
            
    emp_to_dept = {u["employee_id"]: u.get("department", "General") for u in MOCK_USERS if u["role"] == "employee"}
    
    res = []
    for emp_id, logs in emp_logs.items():
        # Sort by date descending
        sorted_logs = sorted(logs, key=lambda x: x.get("date", ""), reverse=True)
        latest = sorted_logs[0]
        
        if latest.get("wellbeing_status") == "low":
            # Compute trend by comparing with previous log
            trend = "Stable"
            if len(sorted_logs) > 1:
                prev_score = sorted_logs[1].get("wellbeing_score", 0)
                curr_score = latest.get("wellbeing_score", 0)
                if curr_score < prev_score - 0.05:
                    trend = "Declining"
                elif curr_score > prev_score + 0.05:
                    trend = "Improving"
            
            # Compute risk level based on score
            score = latest.get("wellbeing_score", 0)
            risk_level = "Critical" if score < 0.25 else "High"

            res.append({
                "employee": latest.get("employee_name", emp_id),
                "department": emp_to_dept.get(emp_id, "General"),
                "trend": trend,
                "risk_level": risk_level,
                "employee_id": emp_id # Useful for linking
            })
            
    return res
