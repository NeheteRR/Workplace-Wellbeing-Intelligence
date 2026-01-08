import os
import json
from typing import List, Dict
from dotenv import load_dotenv
from google import genai

load_dotenv()


class LLMRecommender:
    """
    LLM layer responsible ONLY for language generation.
    It must NEVER break the core application flow.
    """

    def __init__(self, model_name: str = "gemini-1.5-flash"):
        self.api_key = os.getenv("GEMINI_API_KEY")

        if not self.api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Please define it as an environment variable."
            )

        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name

    # =====================================================
    # LOW-LEVEL LLM CALL (SAFE)
    # =====================================================

    def _generate(self, prompt: str) -> str:
        """
        Calls Gemini safely.
        Raises NO exceptions upward.
        """
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            return response.text.strip() if response.text else ""
        except Exception as e:
            print("LLM generation failed:", e)
            return ""

    # =====================================================
    # CONVERSATIONAL USER MESSAGE
    # =====================================================

    def generate_conversational_message(self, context: dict) -> str:
        """
        Generates a natural, conversational assistant message.
        Never mentions burnout, scores, or diagnosis.
        """

        prompt = f"""
You are a friendly wellbeing assistant.

Context:
- Emotional pattern: {context.get('pattern')}
- Dominant emotion: {context.get('dominant_emotion')}
- Emotional trend: {context.get('trend')}

Rules:
1. Do NOT mention burnout, risk, scores, or diagnosis.
2. Use warm, natural, human language.
3. Keep it to 1–2 short sentences.
4. Reinforce positive days.
5. Gently acknowledge heavy days.
6. Do NOT give advice unless very soft and optional.

Task:
Write a conversational response to the user.
"""

        response = self._generate(prompt)

        if response:
            return response

        # 🔐 SAFE FALLBACK
        return self._fallback_conversational_message(context)

    def _fallback_conversational_message(self, context: dict) -> str:
        """
        Deterministic fallback to ensure no 500 errors.
        """
        pattern = context.get("pattern")
        trend = context.get("trend")

        if pattern == "positive":
            return (
                "That sounds like a really positive day. "
                "It’s great that you’re feeling good — enjoy those moments."
            )

        if trend == "declining":
            return (
                "It sounds like the last few days have been a bit heavy. "
                "Thanks for taking a moment to reflect and share."
            )

        return (
            "Thanks for sharing how your day went. "
            "Taking time to check in with yourself like this really matters."
        )

    # =====================================================
    # STRUCTURED SUGGESTIONS (LLM)
    # =====================================================

    def generate_suggestions(self, context: dict) -> List[Dict]:
        """
        Uses Gemini to generate structured, non-medical suggestions.
        Returns [] on failure (never crashes).
        """

        prompt = f"""
You are a wellbeing assistant.

Context:
- Internal state: {context['wellbeing_status']}
- Detected signals: {context['signals']}
- Allowed categories: {context['allowed_categories']}

Rules:
1. Do NOT provide medical advice.
2. Do NOT diagnose.
3. Suggestions must be optional, everyday actions.
4. Use ONLY these categories: {context['allowed_categories']}

Task:
Generate 2–3 suggestions in JSON format.

Each item must contain:
- type
- title
- description

Return ONLY valid JSON.
"""

        raw = self._generate(prompt)

        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []

    # =====================================================
    # LEGACY SUPPORT (OPTIONAL)
    # =====================================================

    def recommend(
        self,
        history: List[Dict],
        wellbeing_status: str,
        base_message: str
    ) -> str:
        """
        Legacy method retained for backward compatibility.
        """
        return base_message
