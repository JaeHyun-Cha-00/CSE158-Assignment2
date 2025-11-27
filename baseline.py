import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataframe
df = pd.read_csv("merged.csv", usecols=['user_id', 'rating'])

# Clean
df = df.dropna(subset=['user_id', 'rating'])
df['rating'] = df['rating'].astype(float)

# Train/Test split
train, test = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df['rating']
)

# ========== Build baseline averages ==========
allRatings = []
userRatings = defaultdict(list)

for u, r in zip(train['user_id'], train['rating']):
    r = float(r)
    allRatings.append(r)
    userRatings[u].append(r)

globalAverage = sum(allRatings) / len(allRatings)

userAverage = {}
for u in userRatings:
    userAverage[u] = sum(userRatings[u]) / len(userRatings[u])

# ========== Predict ==========
preds = []

for u in test['user_id']:
    if u in userAverage:
        preds.append(userAverage[u])
    else:
        preds.append(globalAverage)

# ========== Convert predictions to integers 1–5 ==========
def clamp_rating(x):
    return min(5, max(1, int(round(x))))

preds_clamped = [clamp_rating(p) for p in preds]

# ========== Evaluate ==========
accuracy = accuracy_score(test['rating'], preds_clamped)
print("User-Average Baseline Accuracy:", accuracy)
