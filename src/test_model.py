
import tensorflow as tf
import numpy as np
import pickle, json, re, contractions, nltk

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# nltk.download("punkt")
# nltk.download("wordnet")
# nltk.download("averaged_perceptron_tagger")
# nltk.download("stopwords")


# Attention Layer
class AttentionLayer(tf.keras.layers.Layer):
    def build(self, shape):
        self.W = self.add_weight("att_weight", shape=(shape[-1],1),
                                 initializer="normal")
        self.b = self.add_weight("att_bias", shape=(shape[1],1),
                                 initializer="zeros")
    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x,self.W)+self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        return tf.keras.backend.sum(x*a, axis=1)


# Load model + tokenizer
model = load_model("C:\\Project\\models\\best_emotion_model.keras",
                   custom_objects={"AttentionLayer": AttentionLayer})


with open("C:\\Project\\models\\tokenizer.pkl","rb") as f:
    tokenizer = pickle.load(f)

with open("C:\\Project\\models\\config.json","r") as f:
    config = json.load(f)

MAX_LEN = config["max_len"]
EMOTIONS = config["emotions"]


# Preprocessing functions (exact training pipeline)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'@\w+', '', text)
    text = text.replace("#","")
    text = re.sub(r'http\S+|www\S+','', text)
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
    return " ".join(lemmatizer.lemmatize(w, get_wordnet_pos(t)) for w,t in tagged)

combined_sw = set(ENGLISH_STOP_WORDS).union(set(stopwords.words("english")))
keep = {'not','no','nor','never','cannot','but','against','without','cry','very','many'}
final_stopwords = combined_sw - keep

def remove_stopwords(text):
    return " ".join(t for t in text.split() if t not in final_stopwords)

def full_preprocess(text):
    t = preprocess_text(text)
    t = lemmatize_text(t)
    t = remove_stopwords(t)
    return t


# Prediction
def predict_emotion(text):
    clean = preprocess_text(text)
    seq = tokenizer.texts_to_sequences([clean])
    pad = pad_sequences(seq, maxlen=MAX_LEN, padding='post')

    probs = model.predict(pad)[0]

    prob_dict = {EMOTIONS[i]: float(probs[i]) for i in range(len(EMOTIONS))}
    top_emotion = EMOTIONS[np.argmax(probs)]

    return prob_dict, top_emotion


# Example test
probs, top = predict_emotion("I feel lonely and empty today")
print("Probabilities:", probs)
print("Top Emotion:", top)
