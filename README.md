# Kindle Review Sentiment Analysis

An end-to-end NLP project that classifies Amazon Kindle reviews into Positive and Negative sentiments using Machine Learning.

## 🚀 Features
- **Preprocessing:** BeautifulSoup for HTML removal, NLTK for Lemmatization.
- **Vectorization:** TF-IDF with Unigrams and Bigrams.
- **Model:** Logistic Regression with Hyperparameter Tuning via GridSearchCV.
- **UI:** Interactive web app built with Streamlit.

## 🛠️ How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Train the model: `python sentiment_analyzer.py`
3. Launch the app: `streamlit run app.py`

## 📊 Results
- **Accuracy:** ~88% (depending on dataset subset)
- **F1-Score:** Optimized via GridSearchCV