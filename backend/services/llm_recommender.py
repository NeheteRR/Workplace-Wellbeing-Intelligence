# backend/services/llm_recommender.py
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from typing import Dict
import threading

class LLMRecommender:
    """
    Uses an open-source seq2seq LLM (Flan-T5) to produce an enhanced recommendation sentence.
    We keep this lightweight: flan-t5-small is recommended.
    """

    def __init__(self, model_name: str = "google/flan-t5-small"):
        # Load tokenizer + model and create a text2text-generation pipeline
        self.model_name = model_name
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.pipe = pipeline("text2text-generation", model=self.model, tokenizer=self.tokenizer, device=-1)
        except Exception as e:
            # if loading fails, create a fallback that returns the base message
            self.pipe = None

    def _build_prompt(self, text: str, emotions: Dict[str, float], base_message: str) -> str:
        # Build a short prompt including the detected emotions and base message.
        # Keep prompt short for speed.
        top_em = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]
        em_str = ", ".join([f"{k}:{v:.2f}" for k, v in top_em if v > 0.05]) or "none"
        prompt = (
            f"User journal: \"{text}\"\n"
            f"Detected emotions: {em_str}\n"
            f"Base suggestion: {base_message}\n\n"
            "Provide a single concise empathetic recommendation (2-3 sentences). Avoid medical claims.\n"
            "Be friendly, practical, and short."
        )
        return prompt

    def enhance_recommendation(self, text: str, emotions: Dict[str, float], base_message: str) -> str:
        if not self.pipe:
            return base_message  # fallback

        prompt = self._build_prompt(text, emotions, base_message)
        try:
            # Generate with small beam or sampling; short output expected
            out = self.pipe(prompt, max_length=120, do_sample=False, num_beams=2)
            if out and isinstance(out, list):
                return out[0].get("generated_text", base_message)
            return base_message
        except Exception:
            return base_message
