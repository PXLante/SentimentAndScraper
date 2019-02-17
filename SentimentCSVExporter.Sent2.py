from IPython import display
import math
from pprint import pprint
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import praw
#Only need to download ocne
#nltk.download('vader_lexicon')

sns.set(style='darkgrid', context='talk', palette='Dark2')



reddit = praw.Reddit(client_id='R_ksv9g9qY946Q',
                     client_secret='iqzS7w_QvNdcWfm5mwkL5xcF0WU',
                     user_agent='subSentiment')

headlines = set()
for submission in reddit.subreddit('TheBluePill').new(limit=None):
    headlines.add(submission.title)
    display.clear_output()
    print(len(headlines))

from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA

sia = SIA()
results = []

for line in headlines:
    pol_score = sia.polarity_scores(line)
    pol_score['headline'] = line
    results.append(pol_score)

pprint(results[:3], width=100)

df = pd.DataFrame.from_records(results)
df.head()
df['label'] = 0
df.loc[df['compound'] > 0.2, 'label'] = 1
df.loc[df['compound'] < -0.2, 'label'] = -1
df.head()
df2 = df[['headline', 'label']]
df2.to_csv('reddit_headlines_labels.csv', mode='a', encoding='utf-8', index=False)