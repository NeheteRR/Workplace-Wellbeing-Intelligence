
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Embedding, Bidirectional, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle
import json
import tensorflow as tf

df = pd.read_csv("data\\goemotions_bal_removing_stopwords.csv")

texts = df["clean_text"].astype(str).tolist()

MAX_VOCAB_SIZE = 25000
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
seq_lengths = [len(s) for s in sequences]
MAX_LEN = int(np.percentile(seq_lengths, 95))

X = pad_sequences(sequences, maxlen=MAX_LEN, padding='post')

emotion_cols = ['sadness','love','joy','anger','fear','neutral']
Y = df[emotion_cols].values

X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)


# Load GloVe
EMBEDDING_DIM = 100
embeddings_index = {}

with open("./glove.6B.100d.txt", encoding="utf8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        embeddings_index[word] = np.asarray(values[1:], dtype="float32")

VOCAB_SIZE = min(len(tokenizer.word_index)+1, MAX_VOCAB_SIZE)
embedding_matrix = np.zeros((VOCAB_SIZE, EMBEDDING_DIM))

for w, i in tokenizer.word_index.items():
    if i < VOCAB_SIZE and w in embeddings_index:
        embedding_matrix[i] = embeddings_index[w]

embedding_layer = Embedding(
    VOCAB_SIZE, EMBEDDING_DIM, weights=[embedding_matrix],
    input_length=MAX_LEN, trainable=True
)


# Attention Laye
class AttentionLayer(Layer):
    def build(self, shape):
        self.W = self.add_weight(name="att_weight", shape=(shape[-1],1),
                                 initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(shape[1],1),
                                 initializer="zeros")
    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        return tf.keras.backend.sum(x * a, axis=1)


# Model
inp = Input(shape=(MAX_LEN,))
emb = embedding_layer(inp)
lstm = Bidirectional(LSTM(128, return_sequences=True,
                          dropout=0.3, recurrent_dropout=0.3))(emb)
att = AttentionLayer()(lstm)
dense = Dense(64, activation="relu")(att)
dense = Dropout(0.5)(dense)
out = Dense(6, activation="sigmoid")(dense)

model = Model(inp, out)
model.compile(loss="binary_crossentropy", optimizer=Adam(1e-4), metrics=["accuracy"])

checkpoint = ModelCheckpoint("./models/best_emotion_model.h5",
                             save_best_only=True, monitor="val_loss")
early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

model.fit(X_train, y_train, validation_data=(X_val,y_val),
          epochs=15, batch_size=64, callbacks=[checkpoint,early_stop])


# Save tokenizer + confi
with open("./models/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open("./models/config.json", "w") as f:
    json.dump({"max_len": MAX_LEN,
               "emotions": emotion_cols}, f)

print("MODEL + TOKENIZER SAVED!")
