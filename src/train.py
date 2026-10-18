import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import joblib
from preprocess import clean_text

# Load dataset
df = pd.read_csv("../data/reviews.csv")
df["clean_review"] = df["review"].apply(clean_text)

# Features & labels
X = df["clean_review"]
y = df["sentiment"]

# Vectorize text
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Save model & vectorizer
joblib.dump(model, "../model.pkl")
joblib.dump(vectorizer, "../vectorizer.pkl")

print("✅ Model trained and saved!")
