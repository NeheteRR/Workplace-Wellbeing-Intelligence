import re
import json
import pickle
import numpy as np
import tensorflow as tf

from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

import nltk
import contractions
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# --------------------------------------------------
# NLTK (download once)
# --------------------------------------------------
# nltk.download("punkt")
# nltk.download("wordnet")
# nltk.download("averaged_perceptron_tagger")
# nltk.download("stopwords")

# --------------------------------------------------
# PATHS
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "best_emotion_model.h5"
TOKENIZER_PATH = MODEL_DIR / "tokenizer.pkl"
CONFIG_PATH = MODEL_DIR / "config.json"

# --------------------------------------------------
# CUSTOM ATTENTION LAYER (KERAS 3 SAFE)
# --------------------------------------------------
class AttentionLayer(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.W = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True
        )
        self.b = self.add_weight(
            name="att_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        e = tf.keras.backend.tanh(
            tf.keras.backend.dot(inputs, self.W) + self.b
        )
        a = tf.keras.backend.softmax(e, axis=1)
        return tf.keras.backend.sum(inputs * a, axis=1)

    def get_config(self):
        return super().get_config()

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
print("MODEL PATH:", MODEL_PATH)
print("MODEL EXISTS:", MODEL_PATH.exists())

model = load_model(
    MODEL_PATH,
    custom_objects={"AttentionLayer": AttentionLayer},
    compile=False
)

print("MODEL LOADED SUCCESSFULLY")

# --------------------------------------------------
# LOAD TOKENIZER & CONFIG
# --------------------------------------------------
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

MAX_LEN = config["max_len"]
EMOTIONS = config["emotions"]

# --------------------------------------------------
# PREPROCESSING (MATCH TRAINING PIPELINE)
# --------------------------------------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'@\w+', '', text)
    text = text.replace("#", "")
    text = re.sub(r'http\S+|www\S+', '', text)
    text = contractions.fix(text)
    text = re.sub(r'[^a-z\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def get_wordnet_pos(tag):
    if tag.startswith("J"): return wordnet.ADJ
    if tag.startswith("V"): return wordnet.VERB
    if tag.startswith("N"): return wordnet.NOUN
    if tag.startswith("R"): return wordnet.ADV
    return wordnet.NOUN

lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    return " ".join(
        lemmatizer.lemmatize(w, get_wordnet_pos(t))
        for w, t in tagged
    )

combined_sw = set(ENGLISH_STOP_WORDS).union(set(stopwords.words("english")))
keep = {'not','no','nor','never','cannot','but','against','without','cry','very','many'}
final_stopwords = combined_sw - keep

def remove_stopwords(text):
    return " ".join(w for w in text.split() if w not in final_stopwords)

def full_preprocess(text):
    text = preprocess_text(text)
    text = lemmatize_text(text)
    text = remove_stopwords(text)
    return text

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
def predict_emotion(text):
    clean_text = full_preprocess(text)
    seq = tokenizer.texts_to_sequences([clean_text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post")

    probs = model.predict(padded, verbose=0)[0]

    prob_dict = {EMOTIONS[i]: float(probs[i]) for i in range(len(EMOTIONS))}
    top_emotion = EMOTIONS[int(np.argmax(probs))]

    return prob_dict, top_emotion

# --------------------------------------------------
# TEST
# --------------------------------------------------
if __name__ == "__main__":
    text = "I feel lonely and empty today"
    probs, top = predict_emotion(text)

    print("\nInput:", text)
    print("Probabilities:", probs)
    print("Top Emotion:", top)
