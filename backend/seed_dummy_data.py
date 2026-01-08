from datetime import date, timedelta
import random
from services.db import JsonStore

store = JsonStore()

ORG_ID = "ORG_001"

employees = [
    {"employee_id": "E101", "name": "Ravi"},
    {"employee_id": "E102", "name": "Anita"},
    {"employee_id": "E103", "name": "Suresh"},
    {"employee_id": "E104", "name": "Meena"},
    {"employee_id": "E105", "name": "Arjun"},
]

base_emotions = ["joy", "sadness", "fear", "anger", "love", "neutral"]

def random_emotions():
    values = [random.random() for _ in base_emotions]
    total = sum(values)
    return {k: v / total for k, v in zip(base_emotions, values)}

def score_to_status(score):
    if score < 0.3:
        return "low"
    elif score < 0.55:
        return "moderate"
    else:
        return "high"

today = date.today()

for emp in employees:
    for i in range(7):  # last 7 days
        day = today - timedelta(days=i)

        burnout_score = round(random.uniform(0.15, 0.75), 2)
        wellbeing_status = score_to_status(burnout_score)

        record = {
            "org_id": ORG_ID,
            "employee_id": emp["employee_id"],
            "employee_name": emp["name"],
            "user_id": emp["employee_id"],  # backward compatibility
            "date": day.isoformat(),
            "emotions": random_emotions(),
            "wellbeing_score": burnout_score,
            "wellbeing_status": wellbeing_status
        }

        store.add_record(record)

print("✅ Dummy organization wellbeing data seeded successfully.")
