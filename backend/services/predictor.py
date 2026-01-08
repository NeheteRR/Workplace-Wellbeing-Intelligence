import os
import json
import pickle
import re
import contractions
import numpy as np
import tensorflow as tf

from typing import Dict, Tuple
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

import nltk
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# -------------------------------------------------
# NLTK setup (run once)
# -------------------------------------------------
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")
nltk.download("stopwords")


# -------------------------------------------------
# Attention Layer (needed for model loading)
# -------------------------------------------------
class AttentionLayer(tf.keras.layers.Layer):
    def build(self, shape):
        self.W = self.add_weight(
            name="att_weight",
            shape=(shape[-1], 1),
            initializer="normal"
        )
        self.b = self.add_weight(
            name="att_bias",
            shape=(shape[1], 1),
            initializer="zeros"
        )
        super().build(shape)

    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        return tf.keras.backend.sum(x * a, axis=1)


# -------------------------------------------------
# Predictor
# -------------------------------------------------
class Predictor:
    """
    Loads trained emotion model and predicts emotion probabilities.
    """

    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        config_path: str
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")

        # Load model
        self.model = load_model(
            model_path,
            custom_objects={"AttentionLayer": AttentionLayer},
            compile=False
        )

        # Load tokenizer
        with open(tokenizer_path, "rb") as f:
            self.tokenizer = pickle.load(f)

        # Load config
        with open(config_path, "r") as f:
            cfg = json.load(f)

        self.max_len = cfg["max_len"]
        self.emotions = cfg["emotions"]

        # Stopwords (same logic as training)
        combined = set(ENGLISH_STOP_WORDS).union(set(stopwords.words("english")))
        keep = {'not','no','nor','never','cannot','but','against','without','cry','very','many'}
        self.stopwords = combined - keep

        self.lemmatizer = WordNetLemmatizer()

    # -------------------------------------------------
    # Preprocessing (MUST match training)
    # -------------------------------------------------

    def _clean_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'@\w+', '', text)
        text = text.replace("#", "")
        text = re.sub(r'http\S+|www\S+', '', text)
        text = contractions.fix(text)
        text = re.sub(r'[^a-z\s]', '', text)
        return re.sub(r'\s+', ' ', text).strip()

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
        return " ".join(
            self.lemmatizer.lemmatize(w, self._get_wordnet_pos(t))
            for w, t in tagged
        )

    def _remove_stopwords(self, text: str) -> str:
        return " ".join(t for t in text.split() if t not in self.stopwords)

    def preprocess(self, text: str) -> str:
        text = self._clean_text(text)
        text = self._lemmatize(text)
        text = self._remove_stopwords(text)
        return text

    # -------------------------------------------------
    # Prediction
    # -------------------------------------------------

    def predict(self, text: str) -> Tuple[Dict[str, float], str]:
        clean = self.preprocess(text)

        seq = self.tokenizer.texts_to_sequences([clean])
        padded = pad_sequences(seq, maxlen=self.max_len, padding="post")

        probs = self.model.predict(padded)[0]

        prob_dict = {
            self.emotions[i]: float(probs[i])
            for i in range(len(self.emotions))
        }

        top_emotion = self.emotions[int(np.argmax(probs))]

        return prob_dict, top_emotion
