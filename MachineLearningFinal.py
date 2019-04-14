from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import nltk
#from ntlk.corpus import stopwords
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import os
import re

                # Compiling reviews into lists

reviews_train = []
df2 = pd.read_csv("D:/Projects/ProperSentiment/Datasets/BroadFinalPill.csv")
for index, row in df2.iterrows():
    reviews_train.append(row['headline'])

reviews_test = []
df = pd.read_csv("D:/Projects/ProperSentiment/Datasets/BroadFinalPill.csv")
for index, row in df.iterrows():
    reviews_test.append(row['headline'])




                # Preprocessing using RegEx

REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)|(\x97)")


def preprocess_reviews(reviews):
    reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]

    return reviews

#print(reviews_test[0])
reviews_train_clean = preprocess_reviews(reviews_train)
reviews_test_clean = preprocess_reviews(reviews_test)

print(reviews_test_clean[5])



#english_stop_words = stopwords.words('english')
# def remove_stop_words(corpus):
#     removed_stop_words = []
#     for review in corpus:
#         removed_stop_words.append(
#             ' '.join([word for word in review.split()
#                       if word not in english_stop_words])
#         )
#     return removed_stop_words
#
# no_stop_words = remove_stop_words(reviews_train_clean)






                # Vectorization (Changing each word into numeric representations)

wc_vectorizer = CountVectorizer(binary=False)
wc_vectorizer.fit(reviews_train_clean)
X = wc_vectorizer.transform(reviews_train_clean)
X_test = wc_vectorizer.transform(reviews_test_clean)

                # Building classifier using Logistic Regression. First 12500 are positive(1) and last 12500 are negative (0)

target = [1 if i < 488 else 0 for i in range(1060)]

X_train, X_val, y_train, y_val = train_test_split(
    X, target, train_size=0.75
)

for c in [0.01, 0.05, 0.25, 0.5, 1]:
    lr = LogisticRegression(C=c)
    lr.fit(X_train, y_train)
    print("Accuracy for C=%s: %s"
          % (c, accuracy_score(y_val, lr.predict(X_val))))

                # Training final model

final_wc = LogisticRegression(C=0.05)
final_wc.fit(X, target)
print("Final Accuracy: %s"
       % accuracy_score(target, final_wc.predict(X_test)))

# print(refsets)
# print(testsets)
# accuracy = accuracy_score(refsets, testsets)
# print('SKLearn Accuracy')
# print(accuracy)
# print()
# coefficients = matthews_corrcoef(refsets, testsets)
# print(coefficients)
# print()


cm = nltk.ConfusionMatrix(target, final_wc.predict(X_test))
print(cm.pretty_format(sort_by_count=True, show_percents=False, truncate=9))
print()
print(cm.pretty_format(sort_by_count=True, show_percents=True, truncate=9))

precision, recall, _ = precision_recall_curve(target, final_wc.predict(X_test), pos_label=1)
pr_auc = auc(recall, precision)
print('Precision')
print(precision)
print('Recall')
print(recall)
print('PR AUC')
print(pr_auc)








                # Final Checks

feature_to_coef = {
    word: coef for word, coef in zip(
        wc_vectorizer.get_feature_names(), final_wc.coef_[0]
    )
}
for best_positive in sorted(
    feature_to_coef.items(),
    key=lambda x: x[1],
    reverse=True)[:5]:
    print(best_positive)


for best_negative in sorted(
    feature_to_coef.items(),
    key=lambda x: x[1])[:5]:
    print(best_negative)