import joblib
from preprocess import clean_text
from transformers import pipeline

# Load classical ML model (default Naive Bayes)
model = joblib.load("../naive_bayes.pkl")
vectorizer = joblib.load("../vectorizer.pkl")

def predict_sentiment_ml(text):
    clean = clean_text(text)
    vec = vectorizer.transform([clean])
    prediction = model.predict(vec)
    return prediction[0]

# Transformer model
sentiment_pipeline = pipeline("sentiment-analysis")

def predict_sentiment_transformer(text):
    result = sentiment_pipeline(text)[0]
    return result["label"], result["score"]

if __name__ == "__main__":
    sample = "The movie was amazing and thrilling!"
    print("Review:", sample)
    print("ML Prediction:", predict_sentiment_ml(sample))
    label, score = predict_sentiment_transformer(sample)
    print(f"Transformer Prediction: {label} (confidence: {score:.2f})")
