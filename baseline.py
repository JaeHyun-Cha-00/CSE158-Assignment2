import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Load dataframe
df = pd.read_csv("merged.csv", usecols=['text', 'rating'])

# 2. Clean
df = df.dropna(subset=['text', 'rating'])
df['text'] = df['text'].astype(str)

# Feature + label
X = df['text']
y = df['rating']

# 3. Train/Val split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. TF-IDF (baseline)
tfidf = TfidfVectorizer(
    max_features=50000,
    #unigram
    ngram_range=(1, 1)
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# 5. Logistic Regression
model = LogisticRegression(
    max_iter=1000,
    C=1.0
)

model.fit(X_train_tfidf, y_train)

# 6. Predict
y_pred = model.predict(X_test_tfidf)

# 7. Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Accuracy: 0.7489227926036781