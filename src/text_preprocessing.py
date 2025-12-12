
import pandas as pd
import re
import contractions
import nltk
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from wordcloud import WordCloud

nltk.download("punkt")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")
nltk.download("stopwords")

df = pd.read_csv("data\\pre_processed\\goemotions_bal.csv")


# Preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'@\w+', '', text)
    text = text.replace('#', '')
    text = re.sub(r'http\S+|www\S+', '', text)
    text = contractions.fix(text)
    text = re.sub(r'[^a-z\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

df['clean_text'] = df['text'].apply(preprocess_text)


# Lemmatization
def get_wordnet_pos(tag):
    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("N"):
        return wordnet.NOUN
    if tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN

lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    return " ".join(lemmatizer.lemmatize(w, get_wordnet_pos(t)) for w, t in tagged)

df['clean_text'] = df['clean_text'].apply(lemmatize_text)


# Stopword removal
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

combined = set(ENGLISH_STOP_WORDS).union(set(stopwords.words("english")))

keep_words = {'not','no','nor','never','cannot','but','without','against','cry','very','many','almost'}

final_stopwords = combined - keep_words

def remove_stopwords(text):
    return " ".join([t for t in text.split() if t not in final_stopwords])

df['clean_text'] = df['clean_text'].apply(remove_stopwords)

df.to_csv("./data/goemotions_bal_removing_stopwords.csv", index=False)
print("SAVED CLEAN DATA → data/goemotions_bal_removing_stopwords.csv")
