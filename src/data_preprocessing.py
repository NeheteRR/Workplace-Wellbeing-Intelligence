
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import contractions
from sklearn.utils import resample
from wordcloud import WordCloud
import nltk


# Load Dataset
df1 = pd.read_csv("data\\raw\\goemotions_1.csv")
df2 = pd.read_csv("data\\raw\\goemotions_2.csv")
df3 = pd.read_csv("data\\raw\\goemotions_3.csv")

df = pd.concat([df1, df2, df3], ignore_index=True)
print("Total samples:", len(df))


# Define emotion columns
emotion_columns = [
    'admiration','amusement','anger','annoyance','approval','caring','confusion',
    'curiosity','desire','disappointment','disapproval','disgust','embarrassment',
    'excitement','fear','gratitude','grief','joy','love','nervousness','optimism',
    'pride','realization','relief','remorse','sadness','surprise','neutral'
]


# Keep selected emotions
emotion_columns = [
    'admiration','amusement','anger','annoyance','caring','disappointment','disgust',
    'embarrassment','excitement','fear','gratitude','grief','joy',
    'love','nervousness','optimism','remorse','sadness','neutral'
]

df = df[['text'] + emotion_columns]


# Merge emotion categories
df['anger_final'] = df[['anger','annoyance','disgust']].max(axis=1)
df['fear_final'] = df[['fear','nervousness','embarrassment']].max(axis=1)
df['love_final'] = df[['love','admiration','caring']].max(axis=1)
df['joy_final'] = df[['joy','amusement','excitement','gratitude','optimism']].max(axis=1)
df['sadness_final'] = df[['sadness','grief','disappointment','remorse']].max(axis=1)
df['neutral_final'] = df['neutral']

final_df = df[['text','sadness_final','love_final','joy_final',
               'anger_final','fear_final','neutral_final']]

final_df.columns = ['text','sadness','love','joy','anger','fear','neutral']
final_df = final_df[final_df[['sadness','love','joy','anger','fear','neutral']].sum(axis=1) > 0]


# Downsample & Balance
neutral_df = final_df[final_df['neutral'] == 1]
non_neutral_df = final_df[final_df['neutral'] == 0]

neutral_downsampled = resample(neutral_df, replace=False,
                               n_samples=int(0.5 * len(neutral_df)),
                               random_state=42)

final_df_balanced = pd.concat([neutral_downsampled, non_neutral_df])

joy_dominant = final_df_balanced[
    (final_df_balanced['joy'] == 1) &
    (final_df_balanced[['sadness','anger','fear','love']] == 0).all(axis=1)
]

other_rows = final_df_balanced.drop(joy_dominant.index)

joy_downsampled = resample(joy_dominant, replace=False,
                           n_samples=int(0.8 * len(joy_dominant)),
                           random_state=42)

final_df_balanced = pd.concat([joy_downsampled, other_rows])
final_df_balanced.to_csv("./data/goemotions_bal.csv", index=False)

print("BALANCED DATA SAVED → data/goemotions_bal.csv")
