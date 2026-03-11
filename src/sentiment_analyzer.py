import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# 1. Load Data
df = pd.read_csv('data/IMDB-Dataset.csv')

# 2. Simple Preprocessing
df['review'] = df['review'].str.lower()
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# 3. The Most Optimized Pipeline (No Fluff)
# This setup is the "sweet spot" for IMDb accuracy.
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 3),      # Captures phrases like "not very good"
        max_features=40000,      # High capacity for detail
        sublinear_tf=True,       # Best for long movie reviews
        stop_words='english'
    )),
    ('lr', LogisticRegression(C=2.0, max_iter=1000))
])

# 4. Train
print("Training High-Accuracy ML Model...")
pipeline.fit(df['review'], df['sentiment'])

# 5. Save to Root
joblib.dump(pipeline, "imdb_pipeline.pkl")
print("✅ Success! 'imdb_pipeline.pkl' is ready.")