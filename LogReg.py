# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
# from scipy.sparse import hstack
# import numpy as np

# # 1. Load dataframe
# df = pd.read_csv("merged.csv", usecols=['text', 'rating', 'gmap_id'])

# # 2. Clean
# df = df.dropna(subset=['text', 'rating', 'gmap_id'])
# df['text'] = df['text'].astype(str)

# # Feature + label
# X = df['text']
# y = df['rating']

# # 3. Train/Val split
# df_train, df_test = train_test_split(
#     df, test_size=0.2, random_state=42, stratify=df['rating']
# )

# # ============================
# # Add avg_rating feature
# # ============================

# # Compute avg rating using ONLY train data
# train_avg = df_train.groupby('gmap_id')['rating'].mean()

# # Map average rating to train data
# df_train['avg_rating'] = df_train['gmap_id'].map(train_avg)

# # Map train averages to test data
# df_test['avg_rating'] = df_test['gmap_id'].map(train_avg)

# # Assign global avg for unseen gmap_id in test
# global_avg = df_train['rating'].mean()
# df_test['avg_rating'] = df_test['avg_rating'].fillna(global_avg)

# # ============================
# # 4. TF-IDF (baseline)
# # ============================

# tfidf = TfidfVectorizer(
#     max_features=50000,
#     ngram_range=(1, 1),
# )

# X_train_tfidf = tfidf.fit_transform(df_train['text'])
# X_test_tfidf = tfidf.transform(df_test['text'])

# # ============================
# # Combine TF-IDF + avg_rating
# # ============================

# X_train_num = df_train[['avg_rating']].values
# X_test_num = df_test[['avg_rating']].values

# # Combine sparse TF-IDF with dense numerical feature
# X_train_final = hstack([X_train_tfidf, X_train_num])
# X_test_final = hstack([X_test_tfidf, X_test_num])

# # 5. Logistic Regression
# model = LogisticRegression(
#     max_iter=2000,
#     C=1.0
# )

# model.fit(X_train_final, df_train['rating'])

# # 6. Predict
# y_pred = model.predict(X_test_final)

# # 7. Accuracy
# print("Accuracy:", accuracy_score(df_test['rating'], y_pred))

# Basline w/ avg Accuracy: 0.7495957333547603

##########################################################

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.sparse import hstack
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np

# 1. Load dataframe
df = pd.read_csv("merged.csv", usecols=['text', 'rating', 'gmap_id'])

# 2. Clean
df = df.dropna(subset=['text', 'rating', 'gmap_id'])
df['text'] = df['text'].astype(str)

# 3. Train/Val split
df_train, df_test = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df['rating']
)

# ============================
# Add avg_rating feature
# ============================

# Compute avg rating from training set only
train_avg = df_train.groupby('gmap_id')['rating'].mean()

df_train['avg_rating'] = df_train['gmap_id'].map(train_avg)
df_test['avg_rating'] = df_test['gmap_id'].map(train_avg)

global_avg = df_train['rating'].mean()
df_test['avg_rating'] = df_test['avg_rating'].fillna(global_avg)

# ============================
# NEW: Add sentiment feature
# ============================

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Compute compound sentiment score
df_train['sentiment'] = df_train['text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
df_test['sentiment'] = df_test['text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

# ============================
# 4. TF-IDF (baseline)
# ============================

tfidf = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 1),
)

X_train_tfidf = tfidf.fit_transform(df_train['text'])
X_test_tfidf = tfidf.transform(df_test['text'])

# ============================
# Combine TF-IDF + avg_rating + sentiment
# ============================

# Combine numeric features into array
num_features_train = df_train[['avg_rating', 'sentiment']].values
num_features_test = df_test[['avg_rating', 'sentiment']].values

# hstack with TF-IDF sparse matrix
X_train_final = hstack([X_train_tfidf, num_features_train])
X_test_final = hstack([X_test_tfidf, num_features_test])

# 5. Logistic Regression
model = LogisticRegression(
    max_iter=2000,
    C=1.0
)

# Train
model.fit(X_train_final, df_train['rating'])

# 6. Predict
y_pred = model.predict(X_test_final)

# 7. Accuracy
print("Accuracy:", accuracy_score(df_test['rating'], y_pred))
