# 🎬 IMDb Sentiment Analysis Benchmark Suite

A **multi-architecture sentiment analysis platform** built with **Machine Learning, Deep Learning, and Transformer models** to compare prediction accuracy, confidence, and latency in real time.

The application analyzes movie reviews using **three different NLP paradigms** and benchmarks their performance through an interactive web interface built with Streamlit.

---

## 🚀 Features

* **Triple-Model Benchmarking**

  * Logistic Regression (Statistical Machine Learning)
  * Custom LSTM Neural Network (Deep Learning)
  * BERT Transformer (State-of-the-Art NLP)

* **Real-time Sentiment Prediction**

  * Positive / Negative classification
  * Confidence scoring

* **Performance Benchmarking**

  * Model latency comparison
  * Accuracy confidence comparison

* **Visual Insights**

  * Word importance visualization using WordCloud
  * Aggregate sentiment star rating
  * Model comparison table

* **Interactive Web Application**

  * Built with Streamlit
  * Fully interactive UI

---

## 🧠 Models Used

### 1️⃣ Logistic Regression (Statistical ML)

Pipeline:

* TF-IDF Vectorization
* N-gram range (1–3)
* Logistic Regression classifier

Advantages:

* Very fast inference
* Low computational cost

---

### 2️⃣ LSTM Neural Network (Deep Learning)

Architecture:

Embedding Layer → LSTM Layer → Dense Layers → Sigmoid Output

Features:

* Captures sequential dependencies
* Context-aware sentiment detection
* Custom tokenizer trained on IMDb dataset

---

### 3️⃣ BERT Transformer (State-of-the-Art NLP)

Model:
DistilBERT fine-tuned for sentiment classification.

Capabilities:

* Contextual language understanding
* Transformer attention mechanism
* High semantic accuracy

---

## 📊 Benchmark Output

The system compares:

* Sentiment prediction
* Confidence score
* Inference latency

Example output:

| Model               | Sentiment | Confidence | Latency |
| ------------------- | --------- | ---------- | ------- |
| Logistic Regression | Positive  | 92%        | 0.002s  |
| LSTM                | Positive  | 94%        | 0.014s  |
| BERT                | Positive  | 97%        | 0.087s  |

---

## 📦 Tech Stack

### Languages

* Python

### Machine Learning

* Scikit-learn
* Logistic Regression
* TF-IDF

### Deep Learning

* TensorFlow
* Keras
* LSTM Networks

### NLP

* NLTK
* Transformers
* DistilBERT

### Visualization

* Matplotlib
* WordCloud

### Deployment

* Streamlit

---

## 📂 Project Structure

```
IMDb-Sentiment-Analysis
│
├── app.py
├── requirements.txt
├── imdb_pipeline.pkl
├── imdb_lstm_model.h5
├── imdb_tokenizer.pkl
│
├── src
│   ├── sentiment_analyzer.py
│   └── deep_analyzer.py
│
└── data
    └── IMDB-Dataset.csv
```

---

## ⚙️ Installation

Clone repository:

```bash
git clone https://github.com/yourusername/imdb-sentiment-analysis.git
cd imdb-sentiment-analysis
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Application

```bash
streamlit run app.py
```

The application will open at:

```
http://localhost:8501
```

---

## 📊 Dataset

Dataset used:
IMDb Movie Reviews Dataset (50,000 reviews)

Features:

* Balanced positive/negative labels
* Long-form natural language reviews

---

## 🎯 Key Highlights

* Built an **end-to-end NLP pipeline**
* Compared **three AI paradigms**
* Implemented **real-time benchmarking**
* Integrated **deep learning and transformer models**

---


## 👨‍💻 Author

Akshay Gupta

Computer Science Student | Machine Learning & AI Enthusiast
