# backend/services/risk_engine.py
from typing import Dict, List, Tuple

class RiskEngine:
    """
    Computes an explainable risk score based on emotion probabilities across a short window.
    This is NOT a clinical diagnosis — only a simple heuristic score and categories.
    """

    def __init__(self, store=None):
        self.store = store

    def compute_risk_score(self, probs: Dict[str, float], duration_factor: float = 1.0) -> float:
        sadness = probs.get("sadness", 0.0)
        loneliness = probs.get("loneliness", 0.0)
        fear = probs.get("fear", 0.0)
        # Weighted mixture (tune to your preference / dataset)
        score = 0.45 * sadness + 0.45 * loneliness + 0.10 * fear
        score = score * duration_factor
        return max(0.0, min(1.0, score))

    def categorize(self, score: float) -> str:
        if score < 0.4:
            return "normal"
        elif score < 0.65:
            return "mild"
        else:
            return "elevated"

    def compute_with_window(self, today_record: Dict, previous_records: List[Dict]) -> Tuple[float, str]:
        # Build window with previous (newest->oldest) then today at end
        window = []
        # previous_records may contain entries with 'emotions' maps
        if previous_records:
            # ensure we order oldest->newest for averaging; incoming get_last_n returns newest first
            prev_ordered = list(reversed(previous_records))
            window.extend(prev_ordered)
        window.append(today_record)

        n = len(window)
        agg = {}
        for r in window:
            for k, v in r.get("emotions", {}).items():
                agg[k] = agg.get(k, 0.0) + float(v)
        avg = {k: v / n for k, v in agg.items()}

        # duration factor: count days where (sadness + loneliness)/2 >= 0.6
        count_bad = 0
        for r in window:
            s = r.get("emotions", {}).get("sadness", 0.0)
            l = r.get("emotions", {}).get("loneliness", 0.0)
            if (s + l) / 2 >= 0.6:
                count_bad += 1
        duration_factor = 1.0 + 0.15 * max(0, (count_bad - 1))

        score = self.compute_risk_score(avg, duration_factor=duration_factor)
        return score, self.categorize(score)

    def make_recommendation(self, record: Dict) -> str:
        emotions = record.get("emotions", {})
        if not emotions:
            return "Thanks for sharing — keep journaling."

        dominant, prob = max(emotions.items(), key=lambda x: x[1])
        if dominant == "loneliness" and prob >= 0.5:
            return "You seem to be feeling lonely. Consider reaching out to someone you trust or scheduling a short meet-up."
        if dominant == "sadness" and prob >= 0.5:
            return "You seemed low today. A short walk or writing down three small positives could help."
        if dominant == "fear" and prob >= 0.5:
            return "If you're feeling anxious, try a short breathing exercise or grounding technique."
        if record.get("risk_level") == "elevated":
            return "We've noticed a pattern of low mood. Consider talking to a trusted person or a professional if it persists."
        return "Thanks for sharing. Consider a small uplifting activity like listening to a favorite song."
