
import streamlit as st
import joblib
import re
import nltk
import time
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from transformers import pipeline as bert_pipeline
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------- Page Configuration ----------
st.set_page_config(page_title="IMDb Sentiment Benchmarking", page_icon="🎬", layout="wide")

# ---------- Resources & Downloads ----------
@st.cache_resource
def download_nltk_data():
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)

@st.cache_resource
def load_ml_pipeline():
    return joblib.load("imdb_pipeline.pkl") 

@st.cache_resource
def load_lstm_assets():
    model = tf.keras.models.load_model("imdb_lstm_model.h5")
    tokenizer = joblib.load("imdb_tokenizer.pkl")
    return model, tokenizer

@st.cache_resource
def load_bert_pipeline():
    return bert_pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

download_nltk_data()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# ---------- Text Cleaning Logic ----------
def clean_input(text):
    text = re.sub(r'<.*?>', '', text) 
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    words = text.split()
    negation_words = {"not", "no", "never", "neither", "nor"}
    cleaned = [lemmatizer.lemmatize(w) for w in words if w not in stop_words or w in negation_words]
    return " ".join(cleaned)

# ---------- UI Content ----------
st.title("🎬 IMDb Triple-Model Sentiment Suite")
st.markdown("""
Enter a movie review below. This app will process your input through **three different AI architectures** simultaneously to compare their accuracy and speed.
""")

user_input = st.text_area("Review Input:", placeholder="The cinematography was brilliant, but the plot felt rushed...", height=150)

if st.button("Analyze with All Models"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        results = []
        
        # --- 1. Logistic Regression (ML) ---
        with st.spinner("Running Statistical ML..."):
            t0 = time.time()
            ml_model = load_ml_pipeline()
            cleaned = clean_input(user_input)
            pred = ml_model.predict([cleaned])[0]
            prob = ml_model.predict_proba([cleaned])[0]
            score = prob[1] * 100 if pred == 1 else prob[0] * 100
            t_ml = time.time() - t0
            results.append({"Model": "Logistic Regression", "Sentiment": "Positive" if pred == 1 else "Negative", "Confidence": f"{score:.1f}%", "Latency": f"{t_ml:.4f}s"})

        # --- 2. LSTM (Deep Learning) ---
        with st.spinner("Running Custom LSTM..."):
            t0 = time.time()
            lstm_model, tokenizer = load_lstm_assets()
            cleaned = clean_input(user_input)
            seq = tokenizer.texts_to_sequences([cleaned])
            padded = pad_sequences(seq, maxlen=100)
            score_raw = float(lstm_model.predict(padded)[0][0])
            t_lstm = time.time() - t0
            sentiment = "Positive" if score_raw > 0.5 else "Negative"
            conf = score_raw * 100 if score_raw > 0.5 else (1-score_raw) * 100
            results.append({"Model": "Custom LSTM (RNN)", "Sentiment": sentiment, "Confidence": f"{conf:.1f}%", "Latency": f"{t_lstm:.4f}s"})

        # --- 3. BERT (Transformer) ---
        with st.spinner("Running BERT Transformer..."):
            t0 = time.time()
            bert = load_bert_pipeline()
            res = bert(user_input[:512])[0]
            t_bert = time.time() - t0
            results.append({"Model": "BERT Transformer", "Sentiment": res['label'].capitalize(), "Confidence": f"{res['score']*100:.1f}%", "Latency": f"{t_bert:.4f}s"})

        # ---------- Display Comparison Table ----------
        st.divider()
        st.subheader("📊 Model Benchmark Comparison")
        st.table(pd.DataFrame(results))

        # ---------- Visual Breakdown ----------
        st.subheader("💡 Visual Insights")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Word Importance (Cloud)**")
            cloud_text = clean_input(user_input)
            wordcloud = WordCloud(background_color="white", width=800, height=400, colormap='viridis').generate(cloud_text)
            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)
            
        with col2:
            st.write("**Star Rating (Based on Average Confidence)**")
            avg_pos = (score if results[0]['Sentiment'] == "Positive" else (100-score)) / 100
            stars = int(avg_pos * 5)
            if stars == 0 and avg_pos > 0: stars = 1
            st.title(" ".join(["⭐"] * stars))
            st.write(f"Aggregate Score: {avg_pos*100:.1f}%")

st.sidebar.markdown("### 🏆 Internship Spotlight")
st.sidebar.info("""
**Architecture Comparison:**
- **ML:** High speed, low resource.
- **LSTM:** Context-aware memory.
- **BERT:** State-of-the-art attention.
""")