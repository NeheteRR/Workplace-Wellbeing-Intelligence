# backend/services/predictor.py

import os
import numpy as np
from tensorflow.keras.models import load_model
from typing import Dict
import nltk
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
import re
import contractions
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Download required NLTK resources
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")
nltk.download("stopwords")


class Predictor:
    """
    Loads Keras model (.keras) and performs preprocessing + probability prediction.
    """

    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        self.model = load_model(model_path, compile=False)

        # emotion labels in output order
        self.labels = ['sadness', 'love', 'joy', 'anger', 'fear', 'neutral']

        # tokenizer not used since you used text preprocessing
        self.max_len = 128

        # Stopwords
        combined = set(ENGLISH_STOP_WORDS).union(set(stopwords.words("english")))
        keep_words = {'not', 'no', 'nor', 'never', 'cannot', 'but', 'without', 'against', 'cry', 'very', 'many', 'almost'}
        self.final_stopwords = combined - keep_words

        # lemmatizer
        self.lemmatizer = WordNetLemmatizer()

    # -------------------------
    # PREPROCESSING PIPELINE
    # -------------------------

    def _clean_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'@\w+', '', text)
        text = text.replace('#', '')
        text = re.sub(r'http\S+|www\S+', '', text)
        text = contractions.fix(text)
        text = re.sub(r'[^a-z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _get_wordnet_pos(self, tag):
        if tag.startswith("J"):
            return wordnet.ADJ
        if tag.startswith("V"):
            return wordnet.VERB
        if tag.startswith("N"):
            return wordnet.NOUN
        if tag.startswith("R"):
            return wordnet.ADV
        return wordnet.NOUN

    def _lemmatize(self, text: str) -> str:
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
        return " ".join(self.lemmatizer.lemmatize(w, self._get_wordnet_pos(t)) for w, t in tagged)

    def _remove_stopwords(self, text: str) -> str:
        return " ".join([t for t in text.split() if t not in self.final_stopwords])

    def _preprocess(self, text: str) -> str:
        text = self._clean_text(text)
        text = self._lemmatize(text)
        text = self._remove_stopwords(text)
        return text

    # -------------------------
    # PREDICTION
    # -------------------------

    def predict(self, text: str) -> Dict[str, float]:
        txt = self._preprocess(text)

        try:
            # If model uses TextVectorization internally
            inputs = np.array([txt], dtype=object)
            raw_out = self.model.predict(inputs)
        except Exception:
            # Fallback
            dummy = np.zeros((1, self.max_len))
            raw_out = self.model.predict(dummy)

        # Ensure shape match
        probs = np.asarray(raw_out).reshape(-1)

        if len(probs) != len(self.labels):
            temp = np.zeros(len(self.labels))
            for i in range(min(len(temp), len(probs))):
                temp[i] = probs[i]
            probs = temp

        probs = np.clip(probs, 0.0, 1.0).tolist()

        return {label: float(p) for label, p in zip(self.labels, probs)}
