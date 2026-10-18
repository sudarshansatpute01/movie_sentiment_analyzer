import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
import joblib
from preprocess import clean_text

# Load Kaggle IMDb dataset
df = pd.read_csv("../data/IMDB Dataset.csv")
df["clean_review"] = df["review"].apply(clean_text)

X = df["clean_review"]
y = df["sentiment"]

# Vectorize text (limit features for speed)
vectorizer = CountVectorizer(max_features=10000)
X_vec = vectorizer.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Models
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": LinearSVC()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n📊 {name} Evaluation:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    joblib.dump(model, f"../{name.replace(' ', '_').lower()}.pkl")

# Save vectorizer
joblib.dump(vectorizer, "../vectorizer.pkl")

print("✅ Models trained on Kaggle IMDb dataset!")
