

import streamlit as st
import joblib
import re
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from transformers import pipeline

# ---------- Page Configuration ----------
st.set_page_config(page_title="NLP Sentiment Suite", page_icon="📚", layout="wide")

# ---------- Resources & Downloads ----------
@st.cache_resource
def download_nltk_data():
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)

@st.cache_resource
def load_ml_pipeline():
    """Loads the Scikit-Learn Pipeline (Logistic Regression)"""
    return joblib.load("sentiment_pipeline.pkl")

@st.cache_resource
def load_bert_pipeline():
    """Loads DistilBERT for Deep Learning inference"""
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

download_nltk_data()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# ---------- Text Cleaning Logic ----------
def clean_input(text):
    """Processes text for ML model and Word Cloud"""
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    words = text.split()
    # Preserving negation words is a 'best practice' for sentiment nuance
    negation_words = {"not", "no", "never", "neither", "nor"}
    cleaned = [lemmatizer.lemmatize(w) for w in words if w not in stop_words or w in negation_words]
    return " ".join(cleaned)

# ---------- Sidebar & Info ----------
st.sidebar.title("🛠️ Model Architecture")
model_choice = st.sidebar.radio(
    "Select Model Type:",
    ("Logistic Regression (85% Accuracy)", "BERT Transformer (Deep Learning)")
)

st.sidebar.divider()
st.sidebar.info("""
**Project Details:**
- **Developer:** BCSE Student, JU
- **Backend:** Python/Scikit-Learn/HuggingFace
- **Frontend:** Streamlit
""")

# ---------- Main UI ----------
st.title("🚀 Advanced Sentiment Analysis Suite")
st.markdown("Compare a optimized **Logistic Regression** baseline against a **BERT Transformer** model.")

# Example Reviews for Demo
examples = [
    "This book changed my life. Absolutely incredible writing!",
    "Total waste of time. The formatting is terrible and full of bugs.",
    "Not the best I've read, but the plot was interesting enough."
]
example_choice = st.selectbox("Quick Demo Examples:", [""] + examples)

user_input = st.text_area("Enter Kindle Review Text:", value=example_choice, height=180)

# ---------- Execution Logic ----------
if st.button("Run Full Analysis"):
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner(f"Processing with {model_choice}..."):
            
            # --- Model Inference ---
            if model_choice == "Logistic Regression (85% Accuracy)":
                model = load_ml_pipeline()
                cleaned_text = clean_input(user_input)
                prediction = model.predict([cleaned_text])[0]
                probs = model.predict_proba([cleaned_text])[0]
                
                pos_score, neg_score = probs[1] * 100, probs[0] * 100
                sentiment_label = "POSITIVE 😊" if prediction == 1 else "NEGATIVE 😞"
            
            else:
                # BERT handles internal tokenization (Transfer Learning)
                bert = load_bert_pipeline()
                result = bert(user_input[:512])[0]
                label_raw = result['label']
                conf = result['score'] * 100
                
                sentiment_label = "POSITIVE 😊" if label_raw == "POSITIVE" else "NEGATIVE 😞"
                pos_score = conf if label_raw == "POSITIVE" else (100 - conf)
                neg_score = conf if label_raw == "NEGATIVE" else (100 - conf)

            # --- Results Display ---
            st.divider()
            col_metric1, col_metric2 = st.columns(2)
            col_metric1.metric("Predicted Sentiment", sentiment_label)
            col_metric2.metric("Confidence Level", f"{max(pos_score, neg_score):.2f}%")

            # --- Score Visuals ---
            st.write("### 📊 Probability Breakdown")
            c1, c2 = st.columns(2)
            c1.write(f"**Positivity:** {pos_score:.1f}%")
            c1.progress(int(pos_score))
            c2.write(f"**Negativity:** {neg_score:.1f}%")
            c2.progress(int(neg_score))

            # --- Word Cloud ---
            st.subheader("🔠 Semantic Word Cloud")
            cloud_text = clean_input(user_input)
            if len(cloud_text.split()) > 1:
                
                wordcloud = WordCloud(background_color="white", width=800, height=400, colormap='coolwarm').generate(cloud_text)
                fig, ax = plt.subplots()
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig)
            else:
                st.info("Word Cloud requires more descriptive text.")