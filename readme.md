\# Movie Review Sentiment Analyzer 🎬



\---





movie-sentiment-analyzer/

│

├── data/

│   └── IMDB Dataset.csv       # Kaggle dataset (50,000 reviews)

│

├── src/

│   ├── preprocess.py

│   ├── train.py

│   ├── predict.py

│

├── app.py						# Flask web app

├── templates/

│   └── index.html				# Web UI

│

├── requirements.txt

├── README.md







\---



A simple AI/ML project that classifies movie reviews as \*\*Positive\*\* or \*\*Negative\*\* using NLP and Naive Bayes.



\## 🚀 How to Run



1\. Install dependencies:

&#x20;  ```bash

&#x20;  pip install -r requirements.txt



2\. kaggle datasets download -d lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

unzip imdb-dataset-of-50k-movie-reviews.zip -d data/





3\. python src/train.py



4\. python src/predict.py



5\. python app.py





\---



📚 Features

Huge Kaggle dataset (50K reviews)



Multiple ML models (Naive Bayes, Logistic Regression, SVM)



Transformer model (Hugging Face)



Evaluation metrics (accuracy, precision, recall, F1)



Flask web app showing all model predictions side by side



\---



🔄 How It Works

After you submit a review:



All ML models predict Positive/Negative.



Transformer predicts with a confidence score.



A bar chart is generated:



ML models → shown as 1 (Positive) or 0 (Negative).



Transformer → shows actual confidence score.



Chart is embedded directly in the results page.



\---



🔄 How It Works

After analyzing a review:



Predictions are shown for all models.



Chart is displayed inline.



Download Chart → saves PNG image.



Download Predictions CSV → saves a CSV file with: Review,Naive Bayes,Logistic Regression,SVM,Transformer

"Your review text",Positive,Positive,Negative,POSITIVE (confidence: 0.98)



\---



app can export both visuals and structured data — perfect for building datasets, reports, or presentations.



\---



✅ Features Added (Commit Log Style)

* Initial Project Setup



Added preprocess.py for text cleaning (lowercasing, stopword removal).



Added train.py with Naive Bayes model training.



Added predict.py for single review prediction.



Created requirements.txt and README.md.



* Evaluation Metrics



Integrated classification\_report and accuracy\_score in train.py.



Prints precision, recall, F1, and accuracy after training.



* Multiple Models



Added Logistic Regression and SVM alongside Naive Bayes.



Saved each model separately (naive\_bayes.pkl, logistic\_regression.pkl, svm.pkl).



* Transformer Integration



Added Hugging Face pipeline("sentiment-analysis") in predict.py.



Allows switching between classical ML and Transformer predictions.



* Flask Web App



Created app.py with a simple form.



Added index.html template for user input and results display.



* Model Selection in Web App



Dropdown menu to choose Naive Bayes, Logistic Regression, SVM, or Transformer.



Predictions shown based on selected model.



* Compare All Models at Once



Web app updated to run all models simultaneously.



Results displayed side by side in a list.



\*Kaggle Dataset Integration



Updated train.py to load IMDb 50K reviews dataset.



Vectorizer limited to 10,000 features for efficiency.



Models trained on large-scale data.



* Confidence Chart Visualization



Added Matplotlib bar chart to compare predictions.



Chart embedded in results page (index.html).



* Download Chart Feature



Added /download\_chart route in Flask.



Button to download chart as sentiment\_chart.png.



* Download Predictions CSV



Added /download\_csv route in Flask.



Button to export predictions (review + outputs from all models) as sentiment\_predictions.csv.



💡 Suggested Git Commit Messages -



feat: add text preprocessing and Naive Bayes training



feat: integrate evaluation metrics in train.py



feat: add Logistic Regression and SVM models



feat: add Hugging Face Transformer sentiment analysis



feat: create Flask web app with input form



feat: add model selection dropdown in web app



feat: run all models simultaneously and display results



feat: integrate Kaggle IMDb dataset for training



feat: add Matplotlib confidence chart visualization



feat: add chart download feature in Flask app



feat: add CSV download for predictions



\---

