from typing import Dict, List, Tuple

class BurnoutEngine:
    """
    Detects early burnout risk using multi-day emotion trends.
    This is NOT a clinical diagnosis.
    """

    # Emotion thresholds
    EMOTION_THRESHOLD = 0.45

    def is_burnout_signal_day(self, emotions: Dict[str, float]) -> bool:
        """
        A day is considered a burnout-signal day if
        negative emotions cross a threshold.
        """
        return (
            emotions.get("sadness", 0) > self.EMOTION_THRESHOLD or
            emotions.get("anger", 0) > self.EMOTION_THRESHOLD or
            emotions.get("fear", 0) > self.EMOTION_THRESHOLD
        )

    def compute_burnout_score(self, emotions: Dict[str, float], persistence_days: int) -> float:
        """
        Compute burnout score using weighted emotions
        and persistence factor.
        """
        sadness = emotions.get("sadness", 0)
        anger = emotions.get("anger", 0)
        fear = emotions.get("fear", 0)

        base_score = (
            0.45 * sadness +
            0.30 * anger +
            0.25 * fear
        )

        # Increase score if pattern persists
        score = base_score * (1 + 0.15 * max(0, persistence_days - 1))
        return min(1.0, score)
    
    def analyze_trend(self, history: List[Dict]):
        """
        Analyze burnout trend over multiple days.
        Returns:
        - burnout_score
        - wellbeing_status
        - dominant_signals
        """
        if not history:
            return 0.0, "low", []

        signal_days = 0
        avg_emotions = {}

        for h in history:
            emotions = h["emotions"]

            if self.is_burnout_signal_day(emotions):
                signal_days += 1

            for k, v in emotions.items():
                avg_emotions[k] = avg_emotions.get(k, 0) + v

        avg_emotions = {k: v / len(history) for k, v in avg_emotions.items()}

        # ✅ Always compute base burnout score
        burnout_score = self.compute_burnout_score(avg_emotions, signal_days)

        # ✅ History-aware interpretation
        if len(history) < 3:
            wellbeing_status = "low"   # not enough evidence yet
        else:
            if burnout_score < 0.4:
                wellbeing_status = "low"
            elif burnout_score < 0.65:
                wellbeing_status = "moderate"
            else:
                wellbeing_status = "high"

        dominant_signals = [
            k for k, v in avg_emotions.items()
            if k in ["sadness", "anger", "fear"] and v > self.EMOTION_THRESHOLD
        ]

        return burnout_score, wellbeing_status, dominant_signals


    def base_recommendation(self, wellbeing_status: str) -> str:
        if wellbeing_status == "low":
            return (
                "It sounds like you're managing things reasonably well. "
                "Keep maintaining balance and taking care of yourself."
            )

        elif wellbeing_status == "moderate":
            return (
                "It seems like the last few days have been mentally tiring. "
                "Taking short pauses or easing your workload might help."
            )

        else:  # high
            return (
                "It sounds like you've been carrying a lot lately. "
                "Making space for rest and small moments of recovery could be helpful."
            )


    def allowed_suggestion_categories(self, wellbeing_status: str):
        if wellbeing_status == "low":
            return []
        elif wellbeing_status == "moderate":
            return ["relaxation", "activity"]
        else:  # high
            return ["relaxation", "activity", "trend"]
