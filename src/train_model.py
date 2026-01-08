import os
import json
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, Embedding,
    Bidirectional, Layer
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --------------------------------------------------
# PATHS (SINGLE SOURCE OF TRUTH)
# --------------------------------------------------
BASE_DIR = r"D:\\Projects_Final\\Sentiment_Analysis"
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
GLOVE_PATH = os.path.join(BASE_DIR, "resources", "glove", "glove.6B.100d.txt")

os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_KERAS_PATH = os.path.join(MODEL_DIR, "best_emotion_model.keras")
MODEL_H5_PATH = os.path.join(MODEL_DIR, "best_emotion_model.h5")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
df = pd.read_csv(os.path.join(DATA_DIR, "goemotions_bal_removing_stopwords.csv"))

texts = df["clean_text"].astype(str).tolist()

emotion_cols = ['sadness', 'love', 'joy', 'anger', 'fear', 'neutral']
Y = df[emotion_cols].values

# --------------------------------------------------
# TOKENIZER
# --------------------------------------------------
MAX_VOCAB_SIZE = 25_000
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
seq_lengths = [len(s) for s in sequences]
MAX_LEN = int(np.percentile(seq_lengths, 95))

X = pad_sequences(sequences, maxlen=MAX_LEN, padding="post")

X_train, X_val, y_train, y_val = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# --------------------------------------------------
# LOAD GLOVE EMBEDDINGS
# --------------------------------------------------
EMBEDDING_DIM = 100
embeddings_index = {}

with open(GLOVE_PATH, encoding="utf8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        embeddings_index[word] = np.asarray(values[1:], dtype="float32")

VOCAB_SIZE = min(len(tokenizer.word_index) + 1, MAX_VOCAB_SIZE)
embedding_matrix = np.zeros((VOCAB_SIZE, EMBEDDING_DIM))

for word, i in tokenizer.word_index.items():
    if i < VOCAB_SIZE and word in embeddings_index:
        embedding_matrix[i] = embeddings_index[word]

embedding_layer = Embedding(
    input_dim=VOCAB_SIZE,
    output_dim=EMBEDDING_DIM,
    weights=[embedding_matrix],
    trainable=True
)

# --------------------------------------------------
# CUSTOM ATTENTION LAYER (KERAS 3 SAFE)
# --------------------------------------------------
class AttentionLayer(Layer):
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

# --------------------------------------------------
# MODEL
# --------------------------------------------------
inp = Input(shape=(MAX_LEN,))
emb = embedding_layer(inp)

lstm = Bidirectional(
    LSTM(128, return_sequences=True, dropout=0.3)
)(emb)

att = AttentionLayer()(lstm)
x = Dense(64, activation="relu")(att)
x = Dropout(0.5)(x)

out = Dense(6, activation="sigmoid")(x)

model = Model(inp, out)
model.compile(
    optimizer=Adam(1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# --------------------------------------------------
# CALLBACKS
# --------------------------------------------------
checkpoint = ModelCheckpoint(
    MODEL_KERAS_PATH,
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=False
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

# --------------------------------------------------
# TRAIN
# --------------------------------------------------
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=15,
    batch_size=64,
    callbacks=[checkpoint, early_stop]
)

# --------------------------------------------------
# SAVE FINAL ARTIFACTS
# --------------------------------------------------
# Extra safety backup
model.save(MODEL_H5_PATH)

with open(TOKENIZER_PATH, "wb") as f:
    pickle.dump(tokenizer, f)

with open(CONFIG_PATH, "w") as f:
    json.dump(
        {
            "max_len": MAX_LEN,
            "emotions": emotion_cols
        },
        f,
        indent=4
    )

print("ALL MODELS & FILES SAVED TO:")
print(MODEL_DIR)
