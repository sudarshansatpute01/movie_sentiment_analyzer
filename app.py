from flask import Flask, request, render_template, send_file
import joblib
from src.preprocess import clean_text
from transformers import pipeline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import csv

app = Flask(__name__)

# Load classical ML models
models = {
    "Naive Bayes": joblib.load("naive_bayes.pkl"),
    "Logistic Regression": joblib.load("logistic_regression.pkl"),
    "SVM": joblib.load("svm.pkl")
}
vectorizer = joblib.load("vectorizer.pkl")

# Load Transformer pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

def predict_ml(text, model):
    clean = clean_text(text)
    vec = vectorizer.transform([clean])
    return model.predict(vec)[0]

def predict_transformer(text):
    result = sentiment_pipeline(text)[0]
    return result["label"], result["score"]

def generate_chart(predictions, transformer_confidence):
    labels = list(predictions.keys())
    values = [1 if v.lower() == "positive" else 0 for v in predictions.values()[:-1]]
    values.append(transformer_confidence)

    plt.figure(figsize=(6,4))
    plt.bar(labels, values, color=["blue","green","red","purple"])
    plt.ylim(0,1)
    plt.ylabel("Confidence / Sentiment Score")
    plt.title("Model Predictions Comparison")

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    chart_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    buf.close()
    return chart_base64

@app.route("/", methods=["GET", "POST"])
def home():
    predictions = {}
    chart = None
    review = None
    if request.method == "POST":
        review = request.form["review"]
        
        for name, model in models.items():
            predictions[name] = predict_ml(review, model)
        
        label, score = predict_transformer(review)
        predictions["Transformer"] = f"{label} (confidence: {score:.2f})"
        
        chart = generate_chart(predictions, score)
    
    return render_template("index.html", predictions=predictions, chart=chart, review=review)

@app.route("/download_chart")
def download_chart():
    # Example chart download (fixed sample text)
    sample_text = "The movie was amazing and thrilling!"
    predictions = {name: predict_ml(sample_text, model) for name, model in models.items()}
    label, score = predict_transformer(sample_text)
    predictions["Transformer"] = f"{label} (confidence: {score:.2f})"

    labels = list(predictions.keys())
    values = [1 if v.lower() == "positive" else 0 for v in predictions.values()[:-1]]
    values.append(score)

    plt.figure(figsize=(6,4))
    plt.bar(labels, values, color=["blue","green","red","purple"])
    plt.ylim(0,1)
    plt.ylabel("Confidence / Sentiment Score")
    plt.title("Model Predictions Comparison")

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    return send_file(buf, mimetype="image/png", as_attachment=True, download_name="sentiment_chart.png")

@app.route("/download_csv", methods=["POST"])
def download_csv():
    review = request.form["review"]
    predictions = {}
    for name, model in models.items():
        predictions[name] = predict_ml(review, model)
    label, score = predict_transformer(review)
    predictions["Transformer"] = f"{label} (confidence: {score:.2f})"

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Review"] + list(predictions.keys()))
    writer.writerow([review] + list(predictions.values()))
    buf.seek(0)

    return send_file(io.BytesIO(buf.getvalue().encode("utf-8")),
                     mimetype="text/csv",
                     as_attachment=True,
                     download_name="sentiment_predictions.csv")

if __name__ == "__main__":
    app.run(debug=True)
