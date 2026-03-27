from datetime import date, timedelta
import random
import json
import os
import sys
from pathlib import Path

# Add the backend directory to sys.path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.db import JsonStore
from services.predictor import Predictor
from services.wellbeing_trend_engine import BurnoutEngine

# --- CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_emotion_model.keras"
TOKENIZER_PATH = BASE_DIR / "models" / "tokenizer.pkl"
CONFIG_PATH = BASE_DIR / "models" / "config.json"
ORG_ID = "ORG_001"

# --- DATA ---
employees = [
    {"employee_id": "E101", "name": "Ravi", "profile": "stressed"},
    {"employee_id": "E102", "name": "Anita", "profile": "positive"},
    {"employee_id": "E103", "name": "Suresh", "profile": "neutral"},
    {"employee_id": "E104", "name": "Meena", "profile": "at_risk"},
    {"employee_id": "E105", "name": "Arjun", "profile": "stable"},
]

phrases = {
    "positive": [
        "Had a great day today, finished my tasks early.",
        "Feeling motivated and productive.",
        "Great team meeting, feeling supported.",
        "Excited about the new project phase.",
        "I love the collaborative environment here.",
        "Feeling content with my progress this week."
    ],
    "neutral": [
        "Routine day, caught up on emails.",
        "Work is steady, nothing special to report.",
        "In the office, standard tasks today.",
        "Meetings all day, but productive overall.",
        "Just a regular day at the desk.",
        "Focusing on the backlog today."
    ],
    "negative": [
        "I am completely burned out and can't handle this stress anymore.",
        "I feel terrible and my mental health is suffering.",
        "I am so depressed and anxious about the workload.",
        "Everything is failing and I feel constant fear.",
        "I am extremely angry and frustrated with the management.",
        "I feel hopeless and overwhelmed by these tasks.",
        "The stress is unbearable and I want to quit."
    ]
}

def get_text_for_profile(profile, day_offset):
    if profile == "stressed":
        return random.choice(phrases["negative"] if day_offset < 3 else phrases["neutral"])
    elif profile == "positive":
        return random.choice(phrases["positive"])
    elif profile == "at_risk":
        # Downward trend
        if day_offset < 2:
            return random.choice(phrases["negative"])
        elif day_offset < 5:
            return random.choice(phrases["negative"] + phrases["neutral"])
        else:
            return random.choice(phrases["neutral"])
    elif profile == "stable":
        return random.choice(phrases["neutral"] + phrases["positive"])
    else: # neutral
        return random.choice(phrases["neutral"])

# --- INIT ---
print("Initializing model...")
predictor = Predictor(
    model_path=str(MODEL_PATH),
    tokenizer_path=str(TOKENIZER_PATH),
    config_path=str(CONFIG_PATH)
)
engine = BurnoutEngine()
store = JsonStore()

# Clean existing data for a fresh start
if os.path.exists("backend/data/emotion_logs.json"):
    os.remove("backend/data/emotion_logs.json")
elif os.path.exists("data/emotion_logs.json"):
    os.remove("data/emotion_logs.json")

# Re-init store to create empty file
store = JsonStore() 

all_records = []
today = date.today()

print(f"Generating 7 days of logs for {len(employees)} employees...")

for emp in employees:
    emp_history = []
    for i in range(7):
        day = today - timedelta(days=(6-i)) # Generate chronologically (oldest to newest)
        text = get_text_for_profile(emp["profile"], 6-i)
        
        # 1. Predict emotions
        emotions, dominant = predictor.predict(text)
        
        # 2. Add to history for scoring (using chronological window)
        current_entry = {
            "org_id": ORG_ID,
            "employee_id": emp["employee_id"],
            "employee_name": emp["name"],
            "date": day.isoformat(),
            "raw_text": text,
            "emotions": emotions
        }
        
        emp_history.append(current_entry)
        
        # 3. Score using the engine (engine expects a window of history)
        # We send up to the last 3 days for scoring context
        window = emp_history[-3:]
        score, status, signals = engine.analyze_trend(window)
        
        current_entry.update({
            "wellbeing_score": score,
            "wellbeing_status": status,
            "dominant_signals": signals
        })
        
        all_records.append(current_entry)

# Sort all records by date descending (standard for the store)
all_records.sort(key=lambda x: x["date"], reverse=True)

for r in all_records:
    store.add_record(r)

print(f"✅ Successfully generated {len(all_records)} realistic records using the emotion model.")
