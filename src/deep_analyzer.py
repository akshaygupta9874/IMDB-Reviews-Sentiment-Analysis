import pandas as pd
import numpy as np
import joblib
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from transformers import pipeline

class DeepSentimentAnalyzer:
    def __init__(self, max_words=5000, max_len=100):
        # 1. BERT Pipeline (Zero-Shot)
        # Using DistilBERT for better speed on your laptop
        self.bert_analyzer = pipeline("sentiment-analysis", 
                                     model="distilbert-base-uncased-finetuned-sst-2-english")
        
        # 2. LSTM Configuration
        self.tokenizer = Tokenizer(num_words=max_words)
        self.max_len = max_len

    def build_lstm(self):
        """Builds a Custom LSTM Neural Network"""
        model = Sequential([
            Embedding(5000, 128, input_length=self.max_len),
            LSTM(64, dropout=0.2, recurrent_dropout=0.2),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid') # Binary Output (0 or 1)
        ])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def clean_text(self, text):
        """Removes HTML and standardizes text for training"""
        text = re.sub(r'<.*?>', '', text)
        return text.lower()

# ---------- Execution Logic (The part that forms the files) ----------

if __name__ == "__main__":
    # 1. Setup
    print("🚀 Initializing Deep Analyzer...")
    analyzer = DeepSentimentAnalyzer()
    
    # 2. Data Loading
    print("📂 Loading IMDb Dataset...")
    try:
        df = pd.read_csv('data/IMDB-Dataset.csv')
    except FileNotFoundError:
        print("❌ Error: data/IMDB-Dataset.csv not found!")
        exit()

    # 3. Preprocessing
    df['review'] = df['review'].apply(analyzer.clean_text)
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

    # 4. Fit Tokenizer on actual data
    print("📝 Fitting Tokenizer on 50,000 reviews...")
    analyzer.tokenizer.fit_on_texts(df['review'])
    
    # 5. Prepare Sequences for LSTM
    X = pad_sequences(analyzer.tokenizer.texts_to_sequences(df['review']), maxlen=100)
    y = df['sentiment'].values

    # 6. Train LSTM
    print("🧠 Training LSTM Model (using 10,000 samples for speed)...")
    lstm_model = analyzer.build_lstm()
    # Using a subset so your laptop doesn't freeze
    lstm_model.fit(X[:10000], y[:10000], epochs=2, batch_size=64, validation_split=0.1)

    # 7. Save EVERYTHING to Root
    print("💾 Saving Model & Tokenizer to root directory...")
    
    # Save the LSTM Weights
    lstm_model.save("imdb_lstm_model.h5")
    
    # Save the Tokenizer (Critical for app.py)
    joblib.dump(analyzer.tokenizer, "imdb_tokenizer.pkl")
    
    print("✅ Done! Files generated: 'imdb_lstm_model.h5' and 'imdb_tokenizer.pkl'")