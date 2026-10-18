import joblib
from preprocess import clean_text

# Load model & vectorizer
model = joblib.load("../model.pkl")
vectorizer = joblib.load("../vectorizer.pkl")

def predict_sentiment(text):
    clean = clean_text(text)
    vec = vectorizer.transform([clean])
    prediction = model.predict(vec)
    return prediction[0]

# Example usage
if __name__ == "__main__":
    sample = "The movie was amazing and thrilling!"
    print("Review:", sample)
    print("Predicted Sentiment:", predict_sentiment(sample))
